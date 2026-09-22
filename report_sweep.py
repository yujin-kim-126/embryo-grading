"""
Sweep 결과 요약 리포트 생성기.
각 run 폴더의 eval_result.txt를 읽어 성능을 정리한다.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RunResult:
    name: str
    path: str
    loss: Optional[float]
    stage: Optional[float]
    icm: Optional[float]
    te: Optional[float]

    @property
    def min_icm_te(self) -> Optional[float]:
        if self.icm is None or self.te is None:
            return None
        return min(self.icm, self.te)


def _parse_eval_result(path: str) -> RunResult:
    name = os.path.basename(os.path.dirname(path))
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def _find(pattern: str) -> Optional[float]:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    loss = _find(r"Loss:\s+([0-9.]+)")
    stage = _find(r"Stage Acc:\s+([0-9.]+)")
    icm = _find(r"ICM Acc:\s+([0-9.]+)")
    te = _find(r"TE Acc:\s+([0-9.]+)")
    return RunResult(name=name, path=os.path.dirname(path), loss=loss, stage=stage, icm=icm, te=te)


def _scan_runs(root: str) -> List[RunResult]:
    results: List[RunResult] = []
    if not os.path.isdir(root):
        return results
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        if not os.path.isdir(run_dir):
            continue
        eval_path = os.path.join(run_dir, "eval_result.txt")
        if os.path.exists(eval_path):
            results.append(_parse_eval_result(eval_path))
        else:
            results.append(RunResult(entry, run_dir, None, None, None, None))
    return results


def _format_table(results: List[RunResult]) -> str:
    lines = [
        "| Run | Loss | Stage | ICM | TE | Min(ICM,TE) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        def fmt(v: Optional[float]) -> str:
            return f"{v:.4f}" if isinstance(v, float) else "-"

        lines.append(
            "| {run} | {loss} | {stage} | {icm} | {te} | {minv} |".format(
                run=r.name,
                loss=fmt(r.loss),
                stage=fmt(r.stage),
                icm=fmt(r.icm),
                te=fmt(r.te),
                minv=fmt(r.min_icm_te),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sweep_runs"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sweep_report.md"),
    )
    args = parser.parse_args()

    results = _scan_runs(args.runs_root)
    if results:
        ranked = sorted(
            results,
            key=lambda r: (-1 if r.min_icm_te is None else -r.min_icm_te),
        )
    else:
        ranked = []

    lines: List[str] = [
        "# Sweep 성능 요약",
        "",
        f"총 run 수: {len(results)}",
        "",
    ]
    if not results:
        lines.append("sweep_runs 폴더에 결과가 없습니다.\nrun_sweep.py 실행 후 재생성하세요.")
    else:
        lines.append("## 전체 결과")
        lines.append(_format_table(results))
        lines.append("")
        lines.append("## ICM/TE 기준 상위 5개")
        lines.append(_format_table(ranked[:5]))

    out = "\n".join(lines)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
