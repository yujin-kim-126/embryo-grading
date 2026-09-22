"""sweep_runs 내 각 run에 대해 eval을 실행하고 eval_result.txt를 생성한 뒤 report_sweep 실행."""
import os
import subprocess
import sys

SWEEP_ROOT = os.path.join(os.path.dirname(__file__), "sweep_runs")


def main() -> None:
    if not os.path.isdir(SWEEP_ROOT):
        print("sweep_runs 폴더가 없습니다.")
        return
    runs = sorted([d for d in os.listdir(SWEEP_ROOT) if os.path.isdir(os.path.join(SWEEP_ROOT, d))])
    python = sys.executable
    proj_dir = os.path.dirname(os.path.abspath(__file__))

    for i, run_name in enumerate(runs, 1):
        run_dir = os.path.join(SWEEP_ROOT, run_name)
        eval_result = os.path.join(run_dir, "eval_result.txt")
        if os.path.exists(eval_result):
            print(f"[{i}/{len(runs)}] {run_name}: 이미 평가됨, 스킵")
            continue
        ckpt = os.path.join(run_dir, "best_finetune.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(run_dir, "best_freeze.pt")
        if not os.path.exists(ckpt):
            print(f"[{i}/{len(runs)}] {run_name}: 체크포인트 없음, 스킵")
            continue
        print(f"[{i}/{len(runs)}] {run_name} 평가 중...")
        subprocess.run(
            [python, "eval.py", "--checkpoint", ckpt, "--output-dir", run_dir],
            cwd=proj_dir,
            check=True,
        )

    print("\n리포트 생성 중...")
    subprocess.run([python, "report_sweep.py"], cwd=proj_dir, check=True)


if __name__ == "__main__":
    main()
