from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunResult:
    run: str
    loss: float
    stage: float
    icm: float
    te: float

    @property
    def min_icm_te(self) -> float:
        return min(self.icm, self.te)

    @property
    def avg_icm_te(self) -> float:
        return (self.icm + self.te) / 2.0


def parse_eval_file(path: str, run_name: str) -> Optional[RunResult]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def pick(pattern: str) -> Optional[float]:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    loss = pick(r"Loss:\s+([0-9.]+)")
    stage = pick(r"Stage Acc:\s+([0-9.]+)")
    icm = pick(r"ICM Acc:\s+([0-9.]+)")
    te = pick(r"TE Acc:\s+([0-9.]+)")

    if None in (loss, stage, icm, te):
        return None
    return RunResult(run=run_name, loss=loss, stage=stage, icm=icm, te=te)


def scan_runs(runs_root: str) -> List[RunResult]:
    results: List[RunResult] = []
    if not os.path.isdir(runs_root):
        return results

    for run_name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(run_dir):
            continue
        eval_path = os.path.join(run_dir, "eval_result.txt")
        if not os.path.exists(eval_path):
            continue
        parsed = parse_eval_file(eval_path, run_name)
        if parsed is not None:
            results.append(parsed)
    return results


def fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_vals[lower]
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def write_csv(path: str, header: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_accuracy_by_run(results: Sequence[RunResult], out_path: str) -> None:
    runs = [r.run for r in results]
    x = list(range(len(runs)))
    stage = [r.stage for r in results]
    icm = [r.icm for r in results]
    te = [r.te for r in results]

    plt.figure(figsize=(14, 6))
    plt.plot(x, stage, marker="o", label="Stage")
    plt.plot(x, icm, marker="o", label="ICM")
    plt.plot(x, te, marker="o", label="TE")
    plt.xticks(x, runs, rotation=60, ha="right", fontsize=8)
    plt.ylim(0.4, 1.02)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Sweep Run")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_icm_te_scatter(results: Sequence[RunResult], out_path: str, top_n: int = 5) -> None:
    plt.figure(figsize=(7, 6))
    x = [r.icm for r in results]
    y = [r.te for r in results]
    plt.scatter(x, y, s=45, alpha=0.8)

    ranked = sorted(results, key=lambda r: r.min_icm_te, reverse=True)[:top_n]
    top_names = {r.run: i + 1 for i, r in enumerate(ranked)}

    for r in results:
        if r.run in top_names:
            plt.annotate(
                f"#{top_names[r.run]}",
                (r.icm, r.te),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color="darkred",
            )

    plt.axvline(0.7, color="red", linestyle="--", linewidth=1, label="Target ICM=0.70")
    plt.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="Target TE=0.70")
    plt.xlabel("ICM Accuracy")
    plt.ylabel("TE Accuracy")
    plt.title("ICM vs TE (Top-5 annotated)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_metric_boxplot(results: Sequence[RunResult], out_path: str) -> None:
    data = [
        [r.stage for r in results],
        [r.icm for r in results],
        [r.te for r in results],
        [r.min_icm_te for r in results],
    ]
    labels = ["Stage", "ICM", "TE", "Min(ICM,TE)"]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylim(0.4, 1.02)
    plt.ylabel("Score")
    plt.title("Distribution of Metrics Across Runs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_loss_vs_min_score(results: Sequence[RunResult], out_path: str) -> None:
    x = [r.loss for r in results]
    y = [r.min_icm_te for r in results]
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=50, alpha=0.85)
    best = max(results, key=lambda r: r.min_icm_te)
    plt.annotate(
        best.run,
        (best.loss, best.min_icm_te),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=8,
        color="darkgreen",
    )
    plt.xlabel("Validation Loss")
    plt.ylabel("Min(ICM, TE)")
    plt.title("Loss vs Balanced Task Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        default=os.path.join(os.path.dirname(__file__), "sweep_runs"),
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "performance_materials"),
        type=str,
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = scan_runs(args.runs_root)
    if not runs:
        raise RuntimeError("분석 가능한 eval_result.txt가 없습니다.")

    runs_sorted = sorted(runs, key=lambda r: r.run)
    runs_ranked = sorted(runs, key=lambda r: r.min_icm_te, reverse=True)

    # 1) CSV: 전체 run 상세
    all_csv = os.path.join(args.output_dir, "all_runs_metrics.csv")
    write_csv(
        all_csv,
        ["run", "loss", "stage", "icm", "te", "min_icm_te", "avg_icm_te"],
        [
            [
                r.run,
                fmt(r.loss),
                fmt(r.stage),
                fmt(r.icm),
                fmt(r.te),
                fmt(r.min_icm_te),
                fmt(r.avg_icm_te),
            ]
            for r in runs_sorted
        ],
    )

    # 2) CSV: Top-5
    top5 = runs_ranked[:5]
    top5_csv = os.path.join(args.output_dir, "top5_runs.csv")
    write_csv(
        top5_csv,
        ["rank", "run", "loss", "stage", "icm", "te", "min_icm_te"],
        [
            [str(i + 1), r.run, fmt(r.loss), fmt(r.stage), fmt(r.icm), fmt(r.te), fmt(r.min_icm_te)]
            for i, r in enumerate(top5)
        ],
    )

    # 3) CSV: 기술 통계
    def stats_row(metric_name: str, values: Sequence[float]) -> List[str]:
        return [
            metric_name,
            fmt(min(values)),
            fmt(percentile(values, 0.25)),
            fmt(statistics.mean(values)),
            fmt(statistics.median(values)),
            fmt(percentile(values, 0.75)),
            fmt(max(values)),
            fmt(statistics.pstdev(values)),
        ]

    stage_vals = [r.stage for r in runs]
    icm_vals = [r.icm for r in runs]
    te_vals = [r.te for r in runs]
    min_vals = [r.min_icm_te for r in runs]
    loss_vals = [r.loss for r in runs]

    stats_csv = os.path.join(args.output_dir, "descriptive_stats.csv")
    stats_header = ["metric", "min", "q1", "mean", "median", "q3", "max", "std"]
    stats_rows = [
        stats_row("loss", loss_vals),
        stats_row("stage", stage_vals),
        stats_row("icm", icm_vals),
        stats_row("te", te_vals),
        stats_row("min_icm_te", min_vals),
    ]
    write_csv(stats_csv, stats_header, stats_rows)

    # 4) 그림 생성
    fig1 = os.path.join(args.output_dir, "fig_accuracy_by_run.png")
    fig2 = os.path.join(args.output_dir, "fig_icm_te_scatter.png")
    fig3 = os.path.join(args.output_dir, "fig_metric_boxplot.png")
    fig4 = os.path.join(args.output_dir, "fig_loss_vs_min_score.png")
    plot_accuracy_by_run(runs_sorted, fig1)
    plot_icm_te_scatter(runs_sorted, fig2, top_n=5)
    plot_metric_boxplot(runs_sorted, fig3)
    plot_loss_vs_min_score(runs_sorted, fig4)

    # 5) Markdown 리포트 생성
    best = runs_ranked[0]
    worst = min(runs_ranked, key=lambda r: r.min_icm_te)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    top5_table_rows = [
        [
            str(i + 1),
            r.run,
            fmt(r.loss),
            f"{r.stage*100:.2f}%",
            f"{r.icm*100:.2f}%",
            f"{r.te*100:.2f}%",
            f"{r.min_icm_te*100:.2f}%",
        ]
        for i, r in enumerate(top5)
    ]
    stats_table_rows = stats_rows

    report_md = os.path.join(args.output_dir, "performance_evaluation.md")
    report_lines = [
        "# 통합 성능 평가 보고서",
        "",
        f"- 생성 시각: {created_at}",
        f"- 분석 run 수: {len(runs)}",
        "",
        "## 핵심 결론",
        "",
        f"- 최고 균형 성능 run: `{best.run}` (Min(ICM,TE)={best.min_icm_te*100:.2f}%)",
        f"- 최저 균형 성능 run: `{worst.run}` (Min(ICM,TE)={worst.min_icm_te*100:.2f}%)",
        f"- Stage 평균 정확도: {statistics.mean(stage_vals)*100:.2f}%",
        f"- ICM 평균 정확도: {statistics.mean(icm_vals)*100:.2f}%",
        f"- TE 평균 정확도: {statistics.mean(te_vals)*100:.2f}%",
        "",
        "## 최고 성능",
        "",
        f"- Stage: {max(stage_vals)*100:.2f}%",
        f"- ICM: {max(icm_vals)*100:.2f}%",
        f"- TE: {max(te_vals)*100:.2f}%",
        "",
        "## 표 1. Top-5 run (Min(ICM,TE) 기준)",
        "",
        markdown_table(
            ["Rank", "Run", "Loss", "Stage", "ICM", "TE", "Min(ICM,TE)"],
            top5_table_rows,
        ),
        "",
        "## 표 2. 기술 통계",
        "",
        markdown_table(stats_header, stats_table_rows),
        "",
        "## 그림 목록",
        "",
        "- 그림 1: `fig_accuracy_by_run.png` (run별 Stage/ICM/TE 정확도)",
        "- 그림 2: `fig_icm_te_scatter.png` (ICM-TE 산점도 + Top-5 표시)",
        "- 그림 3: `fig_metric_boxplot.png` (지표 분포 박스플롯)",
        "- 그림 4: `fig_loss_vs_min_score.png` (손실-균형점수 관계)",
        "",
        "## 집계 결과 파일",
        "",
        "- `all_runs_metrics.csv`",
        "- `top5_runs.csv`",
        "- `descriptive_stats.csv`",
        "",
    ]
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[완료] 결과 경로: {args.output_dir}")
    print(f"- {os.path.basename(report_md)}")
    print(f"- {os.path.basename(all_csv)}")
    print(f"- {os.path.basename(top5_csv)}")
    print(f"- {os.path.basename(stats_csv)}")
    print(f"- {os.path.basename(fig1)}")
    print(f"- {os.path.basename(fig2)}")
    print(f"- {os.path.basename(fig3)}")
    print(f"- {os.path.basename(fig4)}")


if __name__ == "__main__":
    main()
