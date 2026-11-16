"""Build one auditable comparison figure from the four ablation workbooks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook, load_workbook


MODES = ("no_uncertainty", "ud_only", "um_only", "full")
MODE_LABELS = {
    "no_uncertainty": "None",
    "ud_only": r"$U_d$-only",
    "um_only": r"$U_m$-only",
    "full": "Full",
}
MODE_COLORS = {
    "no_uncertainty": "#7F7F7F",
    "ud_only": "#0072B2",
    "um_only": "#D55E00",
    "full": "#009E73",
}


def _format_rate(rate: float) -> str:
    return np.format_float_positional(float(rate), trim="-")


def _rate_token(miss_rates: list[float]) -> str:
    return "-".join(_format_rate(rate) for rate in miss_rates)


def _find_latest_workbook(
    log_root: Path, dataset: str, mode: str, miss_rates: list[float]
) -> Path:
    workbook_name = f"{dataset}_{_rate_token(miss_rates)}.xlsx"
    candidates = list(log_root.glob(f"{dataset}_{mode}_*/{workbook_name}"))
    if not candidates:
        raise FileNotFoundError(
            f"No workbook found for dataset={dataset}, mode={mode}, "
            f"missrates={_rate_token(miss_rates)} under {log_root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_history(path: Path) -> dict[str, object]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        history = workbook["Uncertainty_History"]
        rows = history.iter_rows(values_only=True)
        headers = list(next(rows))
        index = {str(header): position for position, header in enumerate(headers)}
        required = (
            "Communication_Round",
            "Global_Mean_U_d_Post",
            "Global_Mean_U_m_Post",
            "ACC",
        )
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        values = list(rows)
        if not values:
            raise ValueError(f"{path} has no recorded communication rounds.")

        metadata = {
            str(row[0]): row[1]
            for row in workbook["Metadata"].iter_rows(min_row=2, values_only=True)
            if row[0] is not None
        }
        return {
            "path": path,
            "mode": str(metadata.get("Experiment_Mode", "")),
            "warmup_rounds": int(metadata.get("Warmup_Rounds", 5)),
            "rounds": np.asarray(
                [row[index["Communication_Round"]] for row in values], dtype=int
            ),
            "ud": np.asarray(
                [row[index["Global_Mean_U_d_Post"]] for row in values], dtype=float
            ),
            "um": np.asarray(
                [row[index["Global_Mean_U_m_Post"]] for row in values], dtype=float
            ),
            "acc": np.asarray(
                [row[index["ACC"]] for row in values], dtype=float
            ),
        }
    finally:
        workbook.close()


def _relative(values: np.ndarray, warmup_rounds: int) -> np.ndarray:
    count = min(values.size, max(1, warmup_rounds))
    baseline = float(np.mean(values[:count]))
    if abs(baseline) <= np.finfo(float).eps:
        return np.zeros_like(values)
    return values / baseline


def _moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    radius = max(1, window // 2)
    return np.asarray(
        [
            np.mean(values[max(0, i - radius) : min(values.size, i + radius + 1)])
            for i in range(values.size)
        ]
    )


def _validate_common_warmup(histories: dict[str, dict[str, object]]) -> int:
    """Validate comparable rounds; initial rounds are reference, not shared state."""
    reference = histories[MODES[0]]
    warmup_rounds = int(reference["warmup_rounds"])
    reference_rounds = np.asarray(reference["rounds"])
    for mode in MODES[1:]:
        current = histories[mode]
        if int(current["warmup_rounds"]) != warmup_rounds:
            raise ValueError("The four workbooks use different warm-up lengths.")
        if not np.array_equal(current["rounds"], reference_rounds):
            raise ValueError("The four workbooks contain different communication rounds.")
    return warmup_rounds


def _late_summary(
    history: dict[str, object], warmup_rounds: int, late_window: int
) -> dict[str, float]:
    ud = np.asarray(history["ud"])
    um = np.asarray(history["um"])
    acc = np.asarray(history["acc"])
    warm_count = min(ud.size, max(1, warmup_rounds))
    late_count = min(ud.size, max(1, late_window))
    warm_ud = float(np.mean(ud[:warm_count]))
    warm_um = float(np.mean(um[:warm_count]))
    warm_acc = float(np.mean(acc[:warm_count]))
    late_ud = float(np.mean(ud[-late_count:]))
    late_um = float(np.mean(um[-late_count:]))
    late_acc = float(np.mean(acc[-late_count:]))
    return {
        "warm_ud": warm_ud,
        "late_ud": late_ud,
        "ud_reduction": 1.0 - late_ud / warm_ud if warm_ud else 0.0,
        "warm_um": warm_um,
        "late_um": late_um,
        "um_reduction": 1.0 - late_um / warm_um if warm_um else 0.0,
        "warm_acc": warm_acc,
        "late_acc": late_acc,
        "acc_gain": late_acc - warm_acc,
    }


def _write_comparison_workbook(
    output_path: Path,
    histories: dict[str, dict[str, object]],
    summaries: dict[str, dict[str, float]],
    warmup_rounds: int,
) -> None:
    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Late_Round_Summary"
    summary_sheet.append(
        [
            "Mode",
            "Source_Workbook",
            "Warmup_Mean_U_d",
            "Late_Mean_U_d",
            "U_d_Reduction",
            "Warmup_Mean_U_m",
            "Late_Mean_U_m",
            "U_m_Reduction",
            "Warmup_Mean_ACC",
            "Late_Mean_ACC",
            "ACC_Gain",
        ]
    )
    for mode in MODES:
        summary = summaries[mode]
        summary_sheet.append(
            [
                mode,
                str(histories[mode]["path"]),
                summary["warm_ud"],
                summary["late_ud"],
                summary["ud_reduction"],
                summary["warm_um"],
                summary["late_um"],
                summary["um_reduction"],
                summary["warm_acc"],
                summary["late_acc"],
                summary["acc_gain"],
            ]
        )

    curves = workbook.create_sheet("Comparison_Curves")
    headers = ["Communication_Round", "Phase"]
    for mode in MODES:
        headers.extend(
            [
                f"{mode}_U_d_Relative",
                f"{mode}_U_m_Relative",
                f"{mode}_ACC",
            ]
        )
    curves.append(headers)
    rounds = np.asarray(histories[MODES[0]]["rounds"])
    relative = {
        mode: (
            _relative(np.asarray(histories[mode]["ud"]), warmup_rounds),
            _relative(np.asarray(histories[mode]["um"]), warmup_rounds),
        )
        for mode in MODES
    }
    for index, round_number in enumerate(rounds):
        row: list[object] = [
            int(round_number),
            "Initial reference" if int(round_number) <= warmup_rounds else "Later rounds",
        ]
        for mode in MODES:
            row.extend(
                [
                    float(relative[mode][0][index]),
                    float(relative[mode][1][index]),
                    float(np.asarray(histories[mode]["acc"])[index]),
                ]
            )
        curves.append(row)

    temp_path = output_path.with_suffix(".tmp.xlsx")
    workbook.save(temp_path)
    os.replace(temp_path, output_path)


def _write_comparison_plot(
    output_path: Path,
    dataset: str,
    miss_rates: list[float],
    histories: dict[str, dict[str, object]],
    summaries: dict[str, dict[str, float]],
    warmup_rounds: int,
    late_window: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)
    metric_specs = (
        ("ud", r"Relative data uncertainty $U_d$", True),
        ("um", r"Relative model uncertainty $U_m$", True),
        ("acc", "Server ACC", False),
    )
    for axis, (metric, ylabel, relative_metric) in zip(axes, metric_specs):
        for mode in MODES:
            rounds = np.asarray(histories[mode]["rounds"])
            raw = np.asarray(histories[mode][metric])
            values = _relative(raw, warmup_rounds) if relative_metric else raw
            axis.plot(
                rounds,
                _moving_average(values),
                color=MODE_COLORS[mode],
                linewidth=2.2 if mode == "full" else 1.8,
                linestyle="-" if mode == "full" else "--",
                label=MODE_LABELS[mode],
            )
        if relative_metric:
            axis.axhline(1.0, color="#999999", linewidth=0.9, linestyle=":")
        else:
            axis.set_ylim(0.0, 1.0)
        if warmup_rounds > 0:
            axis.axvspan(
                0.5,
                warmup_rounds + 0.5,
                color="#BDBDBD",
                alpha=0.13,
                linewidth=0,
            )
            axis.axvline(
                warmup_rounds + 0.5,
                color="#777777",
                linewidth=0.9,
                linestyle=":",
            )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Communication round")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    summary_lines = []
    for mode in MODES:
        summary = summaries[mode]
        summary_lines.append(
            f"{MODE_LABELS[mode]}: "
            f"Ud↓ {summary['ud_reduction'] * 100:.1f}%, "
            f"Um↓ {summary['um_reduction'] * 100:.1f}%, "
            f"late ACC {summary['late_acc']:.3f}"
        )
    summary_text = (
        "   |   ".join(summary_lines[:2])
        + "\n"
        + "   |   ".join(summary_lines[2:])
    )
    fig.text(
        0.5,
        0.012,
        f"Last {late_window}-round means\n{summary_text}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    rate_text = (
        f"{_format_rate(miss_rates[0])} (all views)"
        if np.allclose(miss_rates, miss_rates[0])
        else ", ".join(_format_rate(rate) for rate in miss_rates)
    )
    fig.suptitle(
        f"{dataset} | four-mode dual-uncertainty ablation | missing rate: {rate_text}",
        y=0.975,
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.935))
    temp_path = output_path.with_suffix(".tmp.png")
    fig.savefig(temp_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    os.replace(temp_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the latest four uncertainty-ablation workbooks."
    )
    parser.add_argument("--name", required=True, help="Dataset name used by main.py.")
    parser.add_argument("--missrates", type=float, nargs="+", required=True)
    parser.add_argument("--log-root", default="log_new")
    parser.add_argument("--late-window", type=int, default=5)
    args = parser.parse_args()
    if args.late_window <= 0:
        raise ValueError("late-window must be positive.")

    log_root = Path(args.log_root)
    histories = {
        mode: _read_history(
            _find_latest_workbook(log_root, args.name, mode, args.missrates)
        )
        for mode in MODES
    }
    for mode, history in histories.items():
        if history["mode"] != mode:
            raise ValueError(
                f"Workbook mode mismatch: expected {mode}, found {history['mode']}."
            )
    warmup_rounds = _validate_common_warmup(histories)
    summaries = {
        mode: _late_summary(histories[mode], warmup_rounds, args.late_window)
        for mode in MODES
    }

    output_dir = log_root / (
        f"comparison_{args.name}_{_rate_token(args.missrates)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{args.name}_{_rate_token(args.missrates)}_four_mode_comparison"
    plot_path = output_dir / f"{base_name}.png"
    workbook_path = output_dir / f"{base_name}.xlsx"
    _write_comparison_plot(
        plot_path,
        args.name,
        args.missrates,
        histories,
        summaries,
        warmup_rounds,
        args.late_window,
    )
    _write_comparison_workbook(
        workbook_path, histories, summaries, warmup_rounds
    )
    print(f"Comparison plot: {plot_path}")
    print(f"Comparison Excel: {workbook_path}")


if __name__ == "__main__":
    main()
