"""
목표 성능 달성을 위한 하이퍼파라미터 스윕 실행기.
여러 조합을 순차 실행하고 결과를 각 run 폴더에 저장한다.
"""
import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="학습에 사용할 파이썬 실행 파일 경로",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sweep_runs"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    base_args = [
        "train.py",
        "--use-task-adapters",
        "--use-uncertainty-weighting",
        "--use-weighted-sampler",
        "--aug-advanced",
        "--icm-loss",
        "focal",
        "--te-loss",
        "focal",
        "--icm-weight",
        "1.5",
        "--te-weight",
        "1.5",
    ]

    grid = {
        "focal_gamma": ["1.5", "2.0", "2.5"],
        "aug_strength": ["0.2", "0.3", "0.4"],
        "shared_dim": ["1024", "1536"],
        "head_hidden_dim": ["512", "768"],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    indexed = list(enumerate(combos, start=1))
    if args.start_index > 1:
        indexed = [item for item in indexed if item[0] >= args.start_index]
    if args.end_index is not None:
        indexed = [item for item in indexed if item[0] <= args.end_index]

    def score(values: tuple[str, str, str, str]) -> float:
        focal_gamma, aug_strength, shared_dim, head_hidden_dim = values
        s = 0.0
        s += 2.0 if shared_dim == "1536" else 0.5
        s += 2.0 if head_hidden_dim == "768" else 0.5
        s += 1.5 if aug_strength == "0.3" else (1.0 if aug_strength == "0.4" else 0.3)
        s += 1.0 if focal_gamma == "2.0" else (0.7 if focal_gamma == "2.5" else 0.4)
        return s

    if args.top_k is not None:
        indexed = sorted(indexed, key=lambda x: score(x[1]), reverse=True)[: args.top_k]

    for idx, values in indexed:
        run_name = f"run_{timestamp}_{idx:03d}"
        run_dir = os.path.join(args.output_root, run_name)
        os.makedirs(run_dir, exist_ok=True)

        cmd = [args.python, *base_args]
        cmd += ["--focal-gamma", values[keys.index("focal_gamma")]]
        cmd += ["--aug-strength", values[keys.index("aug_strength")]]
        cmd += ["--shared-dim", values[keys.index("shared_dim")]]
        cmd += ["--head-hidden-dim", values[keys.index("head_hidden_dim")]]
        cmd += ["--save-dir", run_dir]

        print(f"[{idx}/{len(combos)}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
