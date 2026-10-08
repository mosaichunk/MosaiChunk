"""Portable launchers for the two frozen backbones."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .data import prepare_i2v, prepare_t2v, read_samples
from .runtime import ROOT, revision


def distributed_args(args):
    result = ["-m", "torch.distributed.run", f"--nproc_per_node={args.nproc}"]
    if args.nodes == 1:
        return result + ["--standalone"]
    return result + [
        f"--nnodes={args.nodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
    ]


def environment(split, budget, method, training=False):
    # Experimental flags must not silently change a published recipe.
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PTR_", "RAVEN_PIN_", "RAVEN_KV_"))
    }
    env.update(
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    if split == "i2v":
        cudnn = ROOT / "runtime-libs/cudnn97/nvidia/cudnn/lib"
        env["LD_LIBRARY_PATH"] = str(cudnn) + ":" + env.get("LD_LIBRARY_PATH", "")
        env.update(
            PTR_N_FAR="3",
            PTR_F_PAD="0",
            PTR_FAR_FROM="0",
            PTR_N_LOCAL_SLOTS=str(2 + budget if method == "base" else 2),
            PTR_DESC_LAYERS="10,20,30",
        )
    else:
        libs = ROOT / "runtime-libs/cudnn914/nvidia"
        env["LD_LIBRARY_PATH"] = (
            f"{libs / 'cudnn/lib'}:{libs / 'cu13/lib'}:" + env.get("LD_LIBRARY_PATH", "")
        )
        env.update(
            PTR_TEACHER_SINK="6",
            PTR_STUDENT_SINK="1",
            PTR_N_LOCAL="2",
            PTR_FAR_BUDGET="2.0",
            PTR_EVAL_FAR_BUDGET=str(budget),
            PTR_N_SECTIONS="40",
            PTR_DESC_LAYERS="12,25,38",
            PTR_PART_MODE="kmeans",
            PTR_EXPLORE="0.15",
            PTR_ROLLOUT="student",
            PTR_MMR_LAM="1.0",
            PTR_SEG_BLEND="1",
            PTR_SEG_SPLIT="3:6:11" if training else "5:9:16",
        )
    return env


def t2v_config(args, inputs, env, training):
    import yaml

    text = (ROOT / "configs/t2v.yaml").read_text()
    text = text.replace("${assets}", str(args.assets)).replace("${train_data}", str(inputs))
    config = yaml.safe_load(text)
    config["persistence"] = {
        "proj_name": "mosaichunk",
        "exp_name": "run",
        "output_dir": str(args.output),
    }
    env["PTR_TOKENIZER"] = str(args.assets / "MiniMax-H3/FL2VA/tokenizer")
    env["PTR_SEG_TABLE"] = str(inputs / "train2000_segments.json")
    if training:
        config["engine"]["training_steps"] = args.steps
        config["engine"]["val_interval"] = 0
        env["PTR_STEPLOG"] = str(args.output / "steps.jsonl")
    else:
        config["entry"] = {"module": "mosaichunk.t2v.engine", "class_name": "InferenceEngine"}
        config["engine"]["eval_only"] = True
        config["validation"].update(
            num_frames=379,
            num_prompts=args.count,
            prompt_path=str(inputs / "prompts.txt"),
            segment_file=str(inputs / "segments.json"),
        )
        config["data"]["args"].update(
            max_seqlen=122000,
            max_seqlen_per_sample=122000,
            paths=[str(inputs / "prompts.txt")],
            window_size=2 + args.budget if args.method == "base" else 2,
        )
        config["meta_model"] = {
            "module": "mosaichunk.t2v.seg_meta"
            if args.method == "base"
            else "mosaichunk.t2v.ptr_eval",
            "class_name": "SegmentedPromptMetaModel"
            if args.method == "base"
            else "PtrEvalMetaModel",
        }
        if args.method == "base":
            del config["models"]["selector"]
        else:
            selector = config["models"]["selector"]
            selector["weight"] = {"path": str(args.checkpoint)}
            selector["runtime"].update(training=False, requires_grad=False)
            selector.pop("optimizer")
        env.update(
            PTR_SEED_TABLE=str(inputs / "seeds.json"),
            PTR_EVAL_PROMPTS=str(inputs / "prompts.txt"),
            MOSAICHUNK_LATENTS=str(args.output / "latents"),
        )
    dest = args.output / "config.yaml"
    dest.write_text(yaml.safe_dump(config, sort_keys=False))
    return dest


def main(training=False):
    parser = argparse.ArgumentParser(
        description="Train the router." if training else "Generate RememBench rollouts."
    )
    parser.add_argument("--split", choices=["t2v", "i2v"], required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nproc", type=int, default=None, help="GPUs per node; T2V requires 8.")
    parser.add_argument("--nodes", type=int, default=1, help="Number of training nodes.")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", help="Rendezvous host for multi-node training.")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write inputs/configuration without loading models."
    )
    if training:
        parser.add_argument(
            "--train-data", type=Path, help="T2V prompt directory or I2V prepared clip directory."
        )
        parser.add_argument("--steps", type=int, default=4000, help="T2V optimizer steps.")
        parser.add_argument("--epochs", type=int, default=2, help="I2V epochs.")
        parser.add_argument(
            "--max-clips", type=int, default=0, help="I2V smoke test limit per rank."
        )
        parser.add_argument(
            "--chunks", type=int, default=20, help="I2V rollout length (paper: 20)."
        )
        parser.add_argument(
            "--resume",
            type=Path,
            help="I2V optimizer checkpoint; T2V resumes its output directory automatically.",
        )
    else:
        parser.add_argument("--method", choices=["base", "mc"], required=True)
        parser.add_argument(
            "--checkpoint",
            type=Path,
            help="Exported router model.safetensors (default: released checkpoint).",
        )
        parser.add_argument(
            "--budget",
            type=int,
            choices=[1, 2],
            default=2,
            help="Far-memory budget; Base gets 2 + budget recent chunks.",
        )
        parser.add_argument(
            "--data", type=Path, help="RememBench checkout (default: assets/RememBench)."
        )
        parser.add_argument(
            "--setting",
            choices=[
                "rotation_90",
                "rotation_180",
                "rotation_360",
                "translation_90",
                "translation_180",
                "translation_360",
            ],
            default="rotation_180",
        )
        parser.add_argument("--scene-ids", nargs="+")
        parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.nodes < 1 or not 0 <= args.node_rank < args.nodes:
        parser.error("Invalid node count or node rank.")
    if args.nodes > 1 and (not training or not args.master_addr):
        parser.error("Multi-node launch requires training mode and --master-addr.")
    args.assets, args.output = args.assets.resolve(), args.output.resolve()
    if not training:
        args.checkpoint = (
            args.checkpoint or args.assets / "MosaiChunk" / args.split / "model.safetensors"
        ).resolve()
    args.nproc = args.nproc or (8 if args.split == "t2v" else 1)
    if args.split == "t2v" and args.nproc != 8:
        parser.error(
            "The released T2V recipe uses 8 GPUs, sequence parallel size 4 and FSDP shard size 8."
        )
    if args.output.exists() and not training and any(args.output.iterdir()):
        parser.error(
            "Use an empty output directory to avoid mixing configurations or old rollouts."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if training:
        args.budget, args.method = 2, "mc"
        inputs = (args.train_data or ROOT / "data/train" / args.split).resolve()
        if args.split == "i2v":
            names = (ROOT / "data/train/i2v/clips.txt").read_text().splitlines()
            clips = [str(inputs / name) for name in names]
            missing = [p for p in clips if not Path(p).is_dir()]
            if missing:
                parser.error(
                    f"Missing {len(missing)} training clips; first: {missing[0]}. See docs/training.md."
                )
            clip_list = args.output / "clips.txt"
            clip_list.write_text("\n".join(clips) + "\n")
    else:
        args.data = (args.data or args.assets / "RememBench").resolve()
        rows = read_samples(args.data, args.split, args.scene_ids, args.setting, args.limit)
        args.count = len(rows)
        inputs = args.output / "inputs"
        if args.split == "i2v":
            clip_list = prepare_i2v(args.data, rows, inputs)
        else:
            prepare_t2v(rows, inputs)
        (args.output / "samples.json").write_text(json.dumps(rows, indent=2) + "\n")
    env = environment(args.split, args.budget, args.method, training)
    if not args.dry_run:
        cudnn = ROOT / (
            "runtime-libs/cudnn97/nvidia/cudnn/lib/libcudnn_graph.so.9"
            if args.split == "i2v"
            else "runtime-libs/cudnn914/nvidia/cudnn/lib/libcudnn_graph.so.9.14.0"
        )
        if not cudnn.exists():
            parser.error(
                f"Missing pinned cuDNN engines. Run python scripts/setup.py --split {args.split}."
            )
    env["MOSAICHUNK_OUTPUT"] = str(args.output)
    if args.split == "t2v":
        config = t2v_config(args, inputs, env, training)
        command = (
            [sys.executable]
            + distributed_args(args)
            + [
                "-m",
                "common.launch",
                "--config",
                str(config),
            ]
        )
        cwd = ROOT / "third_party/raven"
    else:
        env["MOSAICHUNK_BACKBONE"] = str(args.assets / "LingBot")
        env["MOSAICHUNK_OUTPUT"] = str(args.output)
        command = [sys.executable]
        if training and args.nproc * args.nodes > 1:
            command += distributed_args(args)
        command += [
            "-m",
            "mosaichunk.i2v.worker",
            "train" if training else "test",
            "--ckpt_dir",
            str(args.assets / "LingBot"),
            "--clip_list",
            str(clip_list),
        ]
        if training:
            command += [
                "--epochs",
                str(args.epochs),
                "--max_clips",
                str(args.max_clips),
                "--n_ch",
                str(args.chunks),
                "--tag",
                "run",
            ]
            if args.resume:
                command += ["--resume", str(args.resume.resolve())]
        else:
            command += [
                "--arm",
                "base" if args.method == "base" else "ours",
                "--split",
                "all",
                "--n_ch",
                "16",
                "--n_local",
                str(2 + args.budget if args.method == "base" else 2),
                "--seed",
                "0",
                "--traj_dir",
                str(inputs),
                "--out",
                str(args.output / "videos"),
                "--budget_override",
                str(args.budget),
            ]
            if args.method == "mc":
                command += ["--ckpt", str(args.checkpoint)]
        cwd = ROOT
    manifest = {
        "commit": revision(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": command,
        "recipe": {k: v for k, v in env.items() if k.startswith(("PTR_", "MOSAICHUNK_"))},
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)
