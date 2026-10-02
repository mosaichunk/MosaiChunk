"""Generate Base or MosaiChunk rollouts with the same conditioning and noise."""

import argparse
import json
import os
import shutil

import imageio.v2 as imageio
import numpy as np
import torch

from mosaichunk.checkpoints import load_i2v_router

from . import cfg
from .compose import Composer
from .compose import self_test as rope_self_test
from .dit import FrozenDiT
from .heads import Selector, make_selector
from .store import KVStore
from .traj_palindrome import build_rotation_hssd_style


def _alpha_of(sel):
    """The selector's amplitude scale as a plain float, 1.0 for selectors that have none."""
    a = getattr(sel, "alpha", 1.0)
    return a.detach() if hasattr(a, "detach") else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ours", choices=["base", "ours"])
    ap.add_argument(
        "--pick",
        default="model",
        choices=["model"],
        help="model  = the trained Selector (needs --ckpt).",
    )
    ap.add_argument(
        "--n_sections",
        type=int,
        default=48,
        help="only used when there is no --ckpt to read it from",
    )
    ap.add_argument(
        "--budget_chunks",
        type=float,
        default=cfg.BUDGET_CHUNKS,
        help="row budget in chunks, used ONLY when there is no --ckpt to read it from",
    )
    ap.add_argument(
        "--budget_override",
        type=float,
        default=0.0,
        help="Override the checkpoint far-memory budget; 0 keeps the saved budget.",
    )
    ap.add_argument(
        "--mmr_lam",
        type=float,
        default=-1.0,
        help="-1 = take it from the checkpoint (or 0 if none). >=0 overrides.",
    )
    ap.add_argument(
        "--ckpt", default="", help="Exported model.safetensors; required for MosaiChunk."
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_ch", type=int, default=cfg.N_CH)
    ap.add_argument("--n_local", type=int, default=2)
    ap.add_argument(
        "--traj_dir",
        default="",
        help="Read complete input camera paths from <traj_dir>/<clip>/poses.npy.",
    )
    ap.add_argument(
        "--skip_done",
        action="store_true",
        help="Skip clips that already have saved latents.",
    )
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip clips with missing trajectories and report the count.",
    )
    ap.add_argument("--split", default="test", choices=["test", "val", "all"])
    ap.add_argument("--clip_list", default=cfg.CLIP_LIST)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument(
        "--only",
        default="",
        help="Clip basenames separated by | or comma, filtered after scene splitting. Unknown names are errors.",
    )
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    dit = FrozenDiT(args.ckpt_dir, device_id=0)
    rope_self_test(dit.freqs, device=dev)
    sel = None
    sa = {}
    if args.ckpt:
        ck = load_i2v_router(args.ckpt)
        sa = ck.get("args", {})
    elif args.arm == "ours":
        raise SystemExit(f"--arm ours --pick {args.pick} needs --ckpt")
    n_sections = int(sa.get("n_sections", args.n_sections))
    R = cfg.CHUNK_TOKENS // n_sections
    _bc = (
        args.budget_override
        if args.budget_override > 0
        else float(sa.get("budget_chunks", args.budget_chunks))
    )
    budget_sections = int(_bc * cfg.CHUNK_TOKENS) // R
    topk = max(1, budget_sections // n_sections)
    if args.arm == "ours":
        sel = make_selector(
            args.pick,
            n_sections,
            topk,
            budget_sections,
            d=int(sa.get("d_desc", 1024)),
            tau=float(sa.get("sel_tau", 8.0)),
            explore=0.0,
            train_alpha=bool(int(sa.get("train_alpha", 0))),
        )
        sel.lam = float(args.mmr_lam) if args.mmr_lam >= 0 else float(sa.get("mmr_lam", 0.0))
        _av = _alpha_of(sel)
        print(f"[eval] mmr_lam={sel.lam} alpha={float(_av):.3f} (pre-load)", flush=True)
        if isinstance(sel, Selector):
            sel = sel.to(dev).to(torch.float32)
            ADDITIVE_OK = {"alpha"}
            _r = sel.load_state_dict(ck["sel"], strict=False)
            _miss, _extra = (set(_r.missing_keys), set(_r.unexpected_keys))
            if _miss or _extra:
                print(
                    f"[eval] ckpt key diff -- missing={sorted(_miss)} unexpected={sorted(_extra)}",
                    flush=True,
                )
            _bad = _miss - ADDITIVE_OK | _extra
            if _bad:
                raise SystemExit(f"REFUSING: checkpoint does not match the model on {sorted(_bad)}")
            sel.eval()
        print(
            f"[eval] pick={args.pick} sections={n_sections} topk={topk} budget={budget_sections} sections = {budget_sections * R} rows (teacher {cfg.ORACLE_TOKENS}) part=kmeans"
            + (f" ckpt={args.ckpt} @ g{ck.get('gstep')}" if args.ckpt else " NO TRAINING"),
            flush=True,
        )
    n_ch, lat_f = (args.n_ch, args.n_ch * cfg.CHUNK)
    mem_from = args.n_local + 2
    geom = {
        "traj": "precomputed" if args.traj_dir else "rotation_120",
        "theta_deg": None if args.traj_dir else cfg.THETA_DEG,
        "frames": lat_f * 4 - 3,
        "seed": args.seed,
        "n_ch": n_ch,
        "n_local": args.n_local,
        "split": args.split,
        "split_seed": cfg.SPLIT_SEED,
        "clips_per_scene": cfg.CLIPS_PER_SCENE,
    }
    clips = cfg.split(args.clip_list, args.split)
    if args.only:
        want = [w.strip() for w in args.only.replace("|", ",").split(",") if w.strip()]
        have = {os.path.basename(c): c for c in clips}
        miss = [w for w in want if w not in have]
        if miss:
            raise SystemExit(
                f"REFUSING: --only names not in split {args.split}: {miss}\n  split has {len(have)} clips, e.g. {sorted(have)[:3]}"
            )
        clips = [have[w] for w in want]
    clips = clips[args.shard :: args.nshard]
    if args.max_clips:
        clips = clips[: args.max_clips]
    os.makedirs(args.out, exist_ok=True)
    if cfg.F_LOCAL[0] - cfg.CHUNK < cfg.F_FAR[-1] and args.arm != "base":
        raise SystemExit(
            f"REFUSING: PTR_N_LOCAL_SLOTS={cfg.N_LOCAL_SLOTS} collapses the far region to frames {cfg.F_FAR[0]}..{cfg.F_LOCAL[0] - cfg.CHUNK} (F_LOCAL={cfg.F_LOCAL}), which no longer holds the oracle's slots {cfg.F_FAR}. Only --arm base is meaningful there, not {args.arm!r}"
        )
    _sel_chunks = budget_sections * R / cfg.CHUNK_TOKENS if args.arm == "ours" else 0
    _nmem = 1 + args.n_local + _sel_chunks
    print(
        f"[eval sh{args.shard}/{args.nshard}] arm={args.arm} pick={args.pick} {len(clips)} clips -> {args.out}",
        flush=True,
    )
    print(
        f"[eval sh{args.shard}/{args.nshard}] memory: sink+{args.n_local} local{(' + %g selected (%d sections)' % (_sel_chunks, budget_sections) if args.arm == 'ours' else '')} = {_nmem:g} chunks = {_nmem * cfg.CHUNK_TOKENS:.0f} rows | F_SINK={cfg.F_SINK} F_FAR={cfg.F_FAR} F_LOCAL={cfg.F_LOCAL} F_CUR={cfg.F_CUR}",
        flush=True,
    )
    import atexit

    atexit.register(
        lambda: (
            print(
                f"[eval sh{args.shard}/{args.nshard}] already done, skipped {len(n_done)} clip(s)",
                flush=True,
            )
            if n_done
            else None
        )
    )
    atexit.register(
        lambda: (
            print(
                f"[eval sh{args.shard}/{args.nshard}] skipped {len(n_skip)} clip(s) with no trajectory: {n_skip[:4]}{('...' if len(n_skip) > 4 else '')}",
                flush=True,
            )
            if n_skip
            else None
        )
    )
    n_skip, n_done = ([], [])
    for n, clip in enumerate(clips):
        name = os.path.basename(clip)
        odir = os.path.join(args.out, name)
        adir = os.path.join(odir, "action")
        _done = os.path.exists(os.path.join(odir, "x0.pt")) and os.path.exists(
            os.path.join(odir, "rollout.mp4")
        )
        if args.skip_done and _done:
            n_done.append(name)
            continue
        need = ["poses.npy", "intrinsics.npy", "prompt.txt"]
        src = os.path.join(args.traj_dir, name) if args.traj_dir else ""
        if args.traj_dir:
            missing = [f for f in need if not os.path.exists(os.path.join(src, f))]
            if missing and args.skip_missing:
                n_skip.append(name)
                continue
            if missing:
                raise SystemExit(
                    f"REFUSING: {src} is missing {missing}. A silently regenerated trajectory would make this arm incomparable to the others. Pass --skip_missing if the set omits scenes on purpose."
                )
        os.makedirs(odir, exist_ok=True)
        if args.traj_dir:
            os.makedirs(adir, exist_ok=True)
            for f in need:
                shutil.copyfile(os.path.join(src, f), os.path.join(adir, f))
        else:
            build_rotation_hssd_style(clip, adir, frames=cfg.FRAMES, theta_deg=cfg.THETA_DEG)
        c_ = dit.cond(clip, adir, lat_f)
        c_["ckv"] = dit.cross_cache()
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        noise = list(
            torch.randn(
                16, lat_f, cfg.LAT_H, cfg.LAT_W, generator=gen, device=dev, dtype=torch.float32
            ).split(cfg.CHUNK, 1)
        )
        mem = KVStore(n_sections, "kmeans", torch.bfloat16)
        comp = Composer(dit, mem)
        x0s, whole, picks_log = ({}, {}, [])
        first_cross = True
        with torch.no_grad():
            for c in range(n_ch):
                lids = list(range(max(0, c - args.n_local), c))
                extra = [(whole[0], cfg.F_SINK)] if c > args.n_local else []
                if len(lids) > len(cfg.F_LOCAL):
                    raise SystemExit(
                        f"--n_local {args.n_local} needs {len(lids)} local slots but F_LOCAL has {len(cfg.F_LOCAL)} ({cfg.F_LOCAL}); set PTR_N_LOCAL_SLOTS={args.n_local}"
                    )
                _slots = cfg.F_LOCAL[len(cfg.F_LOCAL) - len(lids) :]
                extra += [
                    (whole[i] if i in whole else mem.whole_of(i), f) for i, f in zip(lids, _slots)
                ]
                cand = mem.candidates(c, args.n_local, keep_sink=True)
                if c < cfg.FAR_FROM_CHUNK:
                    cand = []
                live = bool(cand) and c - 1 in mem.feats
                _staged = any((i not in whole for i in lids))
                mk_near = (
                    (comp.build_whole_chunks(extra) if _staged else dit.mem_kv(extra))
                    if extra
                    else None
                )
                mk_ours = None
                if args.arm == "ours" and live:
                    cand_index = [(j, s) for j in cand for s in range(n_sections)]
                    _red_src = [0] + list(range(max(0, c - args.n_local), c - 1))
                    _red = [j for j in _red_src if j in mem.feats and j != c - 1]
                    _rf = mem.all_feats(_red) if _red else None
                    q_from = c - 1
                    pk, rk, gt, lg = sel(
                        mem.feats[q_from], mem.all_feats(cand), cand_index, red_feats=_rf
                    )
                    mk_ours = comp.build(pk, rk, gates=gt, extra_slots=extra)
                    picks_log.append(
                        {
                            "chunk": c,
                            "picked": [[int(a), int(b)] for a, b in pk],
                            "rows": [mem.R] * len(pk),
                            "q_from": q_from,
                        }
                    )
                mk = {"base": mk_near, "ours": mk_ours if mk_ours is not None else mk_near}[
                    args.arm
                ]
                cur = noise[c].clone()
                for i in range(cfg.N_STEPS):
                    t = float(dit.ts[i])
                    v = dit.forward(cur, t, c_, c, mk, first_cross=first_cross)
                    first_cross = False
                    x0 = cur.float() - t / 1000.0 * v.float()
                    cur = dit.step(cur, x0, i, gen)
                    del v
                x0s[c] = cur.detach()
                cap = dit.capture(x0s[c], c_, c, mk_near, first_cross=first_cross)
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
                for k_ in [k_ for k_ in whole if k_ != 0 and k_ < c - min(args.n_local, 2)]:
                    del whole[k_]
                mem.add(c, cap, seed=args.seed)
                del cap, mk, mk_near, mk_ours
                torch.cuda.empty_cache()
            torch.save(
                {
                    "x0": {c: x0s[c].float().cpu() for c in sorted(x0s)},
                    "n_pre": mem_from,
                    "mem_from": mem_from,
                    "n_ch": n_ch,
                    "arm": args.arm,
                    "pick": args.pick,
                    "geom": geom,
                },
                os.path.join(odir, "x0.pt"),
            )
            frames = dit.decode(x0s)
        if picks_log:
            json.dump(picks_log, open(os.path.join(odir, "picks.json"), "w"))
        try:
            np.savez_compressed(
                os.path.join(odir, "feats.npz"),
                **{str(j): mem.feats[j].float().cpu().numpy() for j in sorted(mem.feats)},
            )
        except Exception as e:
            print(f"[eval] WARN feats.npz failed: {e}", flush=True)
        try:
            np.savez_compressed(
                os.path.join(odir, "sections.npz"),
                **{
                    str(j): mem.sections[j].to(torch.int16).cpu().numpy()
                    for j in sorted(mem.sections)
                },
            )
        except Exception as e:
            print(f"[eval] WARN sections.npz failed: {e}", flush=True)
        w = imageio.get_writer(os.path.join(odir, "rollout.mp4"), fps=16, macro_block_size=8)
        for f in frames:
            w.append_data(f)
        w.close()
        print(
            f"[eval sh{args.shard}] {n + 1}/{len(clips)} {name} chunks={len(x0s)} picks={len(picks_log)}",
            flush=True,
        )
        del x0s, mem, comp, whole
        torch.cuda.empty_cache()
    dit.close()


if __name__ == "__main__":
    main()
