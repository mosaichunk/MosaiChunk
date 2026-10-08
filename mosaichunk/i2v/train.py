"""Train the I2V router by matching frozen-teacher velocity predictions."""

import argparse
import hashlib
import math
import os
import random
import shutil
import time
from datetime import timedelta
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import cfg
from .compose import Composer
from .compose import self_test as rope_self_test
from .dit import FrozenDiT
from .heads import make_selector
from .store import KVStore
from .teacher import _oracle_far
from .traj_palindrome import build_rotation_hssd_style, chunk_poses


def clip_seed(clip, epoch, base=0):
    """Pure function of the clip NAME and the epoch. Never python hash() -- it is salted per process,
    so base and oracle would get different trajectories for the same clip."""
    h = hashlib.sha1(os.path.basename(clip).encode()).hexdigest()[:8]
    return base + epoch * 7919 + int(h, 16) % 1000003


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sections", type=int, default=48, help="Equal-size sections per chunk.")
    ap.add_argument(
        "--budget_chunks",
        type=float,
        default=cfg.BUDGET_CHUNKS,
        help="Far-memory budget in video-chunk equivalents.",
    )
    ap.add_argument(
        "--d_desc", type=int, default=1024, help="Descriptor width per layer/statistic view."
    )
    ap.add_argument("--sel_tau", type=float, default=8.0)
    ap.add_argument(
        "--mmr_lam",
        type=float,
        default=1.0,
        help="Redundancy penalty used by the released checkpoint.",
    )
    ap.add_argument(
        "--explore", type=float, default=0.15, help="Maximum training-only exploration fraction."
    )
    ap.add_argument("--n_local", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.0002)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--w_dist", type=float, default=100.0)
    ap.add_argument(
        "--n_ch", type=int, default=cfg.N_CH, help="Chunks per clip; reduce only for smoke tests."
    )
    ap.add_argument(
        "--traj_dir",
        default="",
        help="Optional directory of precomputed per-clip camera trajectories.",
    )
    ap.add_argument("--clip_list", default=cfg.CLIP_LIST)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--tag", default="ptr0")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    is_dist = world > 1
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if is_dist:
        dist.init_process_group(
            "nccl", rank=rank, world_size=world, timeout=timedelta(minutes=120), device_id=dev
        )
    p0 = rank == 0
    log = (lambda *a: print(*a, flush=True)) if p0 else lambda *a: None
    out_dir = os.path.join(cfg.RUNS, args.tag)
    if p0:
        os.makedirs(out_dir, exist_ok=True)
    dit = FrozenDiT(args.ckpt_dir, device_id=local)
    rope_self_test(dit.freqs, device=dev)
    log("[ptr] rope_rows == causal_rope_apply (bit-exact)")
    if is_dist:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        blk_cls = type(dit.pipe.model.blocks[0])
        apply_activation_checkpointing(
            dit.pipe.model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda m: isinstance(m, blk_cls),
        )
        mesh = init_device_mesh("cuda", (world,))
        for blk in dit.pipe.model.blocks:
            fully_shard(blk, mesh=mesh, reshard_after_forward=True)
        fully_shard(dit.pipe.model, mesh=mesh, reshard_after_forward=True)
        log(f"[ptr] FSDP2 over {world} ranks + activation checkpointing")
    R = cfg.CHUNK_TOKENS // args.n_sections
    budget_rows = int(args.budget_chunks * cfg.CHUNK_TOKENS)
    budget_sections = budget_rows // R
    topk = budget_sections // args.n_sections
    assert args.budget_chunks < len(cfg.F_FAR), (
        "Student far memory must be smaller than the teacher's far memory."
    )
    assert budget_sections > 0, "The student needs a positive section budget."
    log(
        f"[ptr] budget {budget_rows} rows = {args.budget_chunks}x chunk = {budget_sections} sections; topk={topk} per query section; teacher {cfg.ORACLE_TOKENS} rows ({len(cfg.F_FAR)}x)"
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sel = make_selector(
        "model",
        args.n_sections,
        topk,
        budget_sections,
        d=args.d_desc,
        tau=args.sel_tau,
        explore=args.explore,
        train_alpha=False,
    )
    sel.lam = float(args.mmr_lam)
    sel = sel.to(dev).to(torch.float32)
    sel.train()
    sel_params = list(sel.parameters())
    from torch.distributed.tensor import DTensor

    assert not any((isinstance(p, DTensor) for p in sel_params)), (
        "the Selector got sharded -- it must be built AFTER fully_shard and stay top-level"
    )
    log(f"[ptr] selector {sel.n_params() / 1000000.0:.2f}M params")
    if is_dist:
        flat = torch.cat([p.data.reshape(-1) for p in sel_params])
        dist.broadcast(flat, src=0)
        off = 0
        for p in sel_params:
            n = p.numel()
            p.data.copy_(flat[off : off + n].view_as(p))
            off += n
        del flat
    groups = [{"params": sel_params, "lr": args.lr, "initial_lr": args.lr, "name": "sel"}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0, eps=1e-12)
    gstep = 0
    resume_ep, resume_ci = (0, 0)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        if ck.get("sel"):
            ADDITIVE_OK = {"alpha"}
            r = sel.load_state_dict(ck["sel"], strict=False)
            miss, extra = (set(r.missing_keys), set(r.unexpected_keys))
            if miss or extra:
                log(f"[ptr] resume key diff -- missing={sorted(miss)} unexpected={sorted(extra)}")
            bad = miss - ADDITIVE_OK | extra
            if bad:
                raise SystemExit(
                    f"REFUSING to resume: checkpoint does not match the model on {sorted(bad)}. Only {sorted(ADDITIVE_OK)} may be absent (they default sensibly). Either point --resume at a matching checkpoint or start a fresh --tag."
                )
            if miss:
                log(f"[ptr] {sorted(miss)} kept at its default (older checkpoint predates it)")
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        gstep = int(ck.get("gstep", 0))
        resume_ep, resume_ci = (int(ck.get("epoch", 0)), int(ck.get("ci", 0)))
        log(f"[ptr] resumed {args.resume} @ gstep={gstep} epoch={resume_ep} clip_idx={resume_ci}")
    clips = cfg.split(args.clip_list, "train")
    mine = clips[rank::world][: len(clips) // world] if is_dist else clips
    if args.max_clips:
        mine = mine[: args.max_clips]
    n_ch = args.n_ch
    lat_f = n_ch * cfg.CHUNK
    log(f"[ptr] {len(mine)} clips/rank  n_ch={n_ch}  sections={args.n_sections} x R={R} rows")
    total_steps = max(1, args.epochs * len(mine) * max(1, n_ch - args.n_local - 2))
    kv_dtype = torch.bfloat16

    def set_lr(step):
        f = (
            (step + 1) / max(1, args.warmup)
            if step < args.warmup
            else 0.05
            + 0.95
            * 0.5
            * (
                1
                + math.cos(
                    math.pi * min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
                )
            )
        )
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * f

    ema, t0 = ({}, time.time())
    for epoch in range(resume_ep, args.epochs):
        for ci, clip in enumerate(mine):
            if epoch == resume_ep and ci < resume_ci:
                continue
            seed = clip_seed(clip, epoch, args.seed)
            adir = os.path.join(out_dir, "actions", f"r{rank}", os.path.basename(clip))
            if args.traj_dir:
                src = os.path.join(args.traj_dir, os.path.basename(clip))
                need = ["poses.npy", "intrinsics.npy", "prompt.txt"]
                missing = [f for f in need if not os.path.exists(os.path.join(src, f))]
                if missing:
                    raise SystemExit(
                        f"REFUSING: {src} is missing {missing}. Generate the trajectory for the FULL train split before training, not per-rank on the fly."
                    )
                os.makedirs(adir, exist_ok=True)
                for f in need:
                    shutil.copyfile(os.path.join(src, f), os.path.join(adir, f))
            else:
                th = cfg.THETA_DEG * 1.0
                build_rotation_hssd_style(clip, adir, frames=cfg.FRAMES, theta_deg=th)
            poses = np.load(os.path.join(adir, "poses.npy")).astype(np.float64)
            yaw = chunk_poses(poses, n_ch, cfg.CHUNK * 4)
            c_ = dit.cond(clip, adir, lat_f)
            c_["ckv"] = dit.cross_cache()
            gen = torch.Generator(device=dev).manual_seed(seed)
            noise = list(
                torch.randn(
                    16, lat_f, cfg.LAT_H, cfg.LAT_W, generator=gen, device=dev, dtype=torch.float32
                ).split(cfg.CHUNK, 1)
            )
            mem = KVStore(args.n_sections, "kmeans", kv_dtype, keep_last=None)
            comp = Composer(dit, mem)
            whole = {}
            first_cross = True
            for c in range(n_ch):
                lids = list(range(max(0, c - args.n_local), c))
                extra = [(whole[0], cfg.F_SINK)] if c > args.n_local else []
                extra += [
                    (whole[i], f) for i, f in zip(lids, cfg.F_LOCAL[len(cfg.F_LOCAL) - len(lids) :])
                ]
                cand = mem.candidates(c, args.n_local, keep_sink=True)
                if c < cfg.FAR_FROM_CHUNK:
                    cand = []
                train_here = len(cand) >= 1 and c - 1 in mem.feats
                graded = (
                    random.Random(gstep * 2654435761 ^ (c + 1) * 40503).randrange(cfg.N_STEPS)
                    if train_here
                    else -1
                )
                cur = noise[c].clone()
                for i in range(cfg.N_STEPS):
                    t = float(dit.ts[i])
                    if train_here:
                        cand_index = [(j, s) for j in cand for s in range(args.n_sections)]
                        _red = [
                            j
                            for j in [0] + list(range(max(0, c - args.n_local), c - 1))
                            if j in mem.feats and j != c - 1
                        ]
                        _rf = mem.all_feats(_red) if _red else None
                        picks, ranks, gates, logits = sel(
                            mem.feats[c - 1],
                            mem.all_feats(cand),
                            cand_index,
                            gen=gen,
                            red_feats=_rf,
                        )
                    if i == graded:
                        with torch.no_grad():
                            far = _oracle_far(c, cand, yaw)
                            mk_t = comp.build_whole_chunks(
                                extra[:1]
                                + [(mem.whole_of(j), cfg.F_FAR[r]) for r, j in enumerate(far)]
                                + extra[1:]
                            )
                            v_t = dit.forward(cur, t, c_, c, mk_t, first_cross=first_cross).float()
                            del mk_t
                        opt.zero_grad(set_to_none=True)
                        mk_s = comp.build(picks, ranks, gates=gates, extra_slots=extra)
                        v_s = dit.forward(cur, t, c_, c, mk_s)
                        loss = F.mse_loss(v_s.float(), v_t)
                        (args.w_dist * loss).backward()
                        v = v_s.detach()
                        del mk_s, v_s, v_t
                        _reduce_and_step(sel_params, is_dist, opt, set_lr, gstep)
                        gstep += 1
                        lv = float(loss.detach())
                        ema["tr"] = lv if "tr" not in ema else 0.9 * ema["tr"] + 0.1 * lv
                        if gstep % args.log_every == 0:
                            n_src = len({j for j, _ in picks})
                            log(
                                f"[ptr g{gstep}] tr {ema['tr']:.3e} lr {opt.param_groups[0]['lr']:.1e} cand {len(cand)} picks {len(picks)} src {n_src} gate[{float(gates.min()):.2f},{float(gates.max()):.2f}] "
                                + f"tau {float(sel.log_tau.exp()):.2f} "
                                + f"host {mem.bytes_resident() / 2**30:.0f}GiB peak {torch.cuda.max_memory_allocated() / 2**30:.0f}GiB {gstep / (time.time() - t0):.3f} st/s"
                            )
                        if gstep % args.save_every == 0:
                            if p0:
                                _save(
                                    out_dir,
                                    gstep,
                                    sel,
                                    dit,
                                    opt,
                                    vars(args),
                                    total_steps,
                                    epoch=epoch,
                                    ci=ci,
                                )
                        del loss
                    else:
                        with torch.no_grad():
                            mk = (
                                comp.build(picks, ranks, gates=gates.detach(), extra_slots=extra)
                                if train_here
                                else dit.mem_kv(extra)
                                if extra
                                else None
                            )
                            v = dit.forward(cur, t, c_, c, mk, first_cross=first_cross)
                            del mk
                    first_cross = False
                    x0 = cur.float() - t / 1000.0 * v.float()
                    cur = dit.step(cur, x0, i, gen)
                    del v
                x0_c = cur.detach()
                cap = dit.capture(
                    x0_c, c_, c, dit.mem_kv(extra) if extra else None, first_cross=first_cross
                )
                first_cross = False
                whole[c] = {
                    L: {
                        kv: cap[L][kv]
                        .view(1, cfg.CHUNK_TOKENS, cfg.N_HEADS, cfg.HEAD_DIM)
                        .to(dit.pdtype)
                        for kv in ("k", "v")
                    }
                    for L in dit.layers
                }
                for k_ in [k_ for k_ in whole if k_ != 0 and k_ < c - args.n_local]:
                    del whole[k_]
                mem.add(c, cap, seed=seed)
                del cap, x0_c
                torch.cuda.empty_cache()
            del mem, comp, whole
            torch.cuda.empty_cache()
    if p0:
        _save(
            out_dir,
            gstep,
            sel,
            dit,
            opt,
            vars(args),
            total_steps,
            final=True,
            epoch=args.epochs,
            ci=0,
        )
        log(f"[ptr] DONE gstep={gstep} ({time.time() - t0:.0f}s)")
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


def _reduce_and_step(sel_params, is_dist, opt, set_lr, gstep):
    """One fused all_reduce for the TOP-LEVEL selector only -- it is the one thing FSDP does not
    reduce, because it is deliberately kept outside the sharded module."""
    if is_dist and sel_params:
        for p in sel_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat = torch._utils._flatten_dense_tensors([p.grad for p in sel_params])
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        for p, g in zip(
            sel_params, torch._utils._unflatten_dense_tensors(flat, [p.grad for p in sel_params])
        ):
            p.grad.copy_(g)
        del flat
    gn = torch.nn.utils.clip_grad_norm_(sel_params, 1.0) if sel_params else torch.tensor(0.0)
    if torch.isfinite(gn):
        set_lr(gstep)
        opt.step()


def _save(out_dir, gstep, sel, dit, opt, args, total_steps, final=False, epoch=0, ci=0):
    sd = {
        "sel": sel.state_dict() if hasattr(sel, "state_dict") else None,
        "gstep": gstep,
        "args": args,
        "total_steps": total_steps,
        "epoch": epoch,
        "ci": ci,
    }
    for path, obj in (
        (os.path.join(out_dir, "last.pt"), {**sd, "opt": opt.state_dict()}),
        (os.path.join(out_dir, f"g{gstep}.pt"), sd),
    ):
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(obj, tmp)
        os.replace(tmp, path)


if __name__ == "__main__":
    main()
