"""Collect the latest paper-ablation runs into the two requested Excel tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


ITEM2_SETTINGS = (
    ("none", "Full", "✓", "✓", "✓"),
    ("qaa_wo_ud", "w/o U_d", "—", "✓", "✓"),
    ("qaa_wo_um", "w/o U_m", "✓", "—", "✓"),
    ("qaa_wo_r", "w/o R", "✓", "✓", "—"),
    ("average_aggregation", "Average aggregation", "—", "—", "—"),
)

ITEM8_SETTINGS = (
    ("wo_proxy_initialization", "w/o Proxy Initialization"),
    ("wo_semantic_refinement", "w/o Semantic Refinement"),
    ("average_aggregation", "Uniform Aggregation"),
    ("none", "Reliability-aware Aggregation"),
)

SERVER_PATTERN = re.compile(
    r"server fused data\s*:com_epoch\s+(?P<round>\d+):"
    r"acc\s+(?P<acc>[-+0-9.eE]+),\s*"
    r"nmi\s+(?P<nmi>[-+0-9.eE]+),\s*"
    r"ari\s+(?P<ari>[-+0-9.eE]+),\s*"
    r"f1\s+(?P<f1>[-+0-9.eE]+)"
)


def _normalized_rates(values):
    rates = np.asarray(values, dtype=float).reshape(-1)
    if np.any(rates > 1.0):
        rates = rates / 100.0
    return rates


def _read_best_metrics(log_path: Path) -> dict[str, float]:
    matches = list(SERVER_PATTERN.finditer(log_path.read_text(encoding="utf-8")))
    if not matches:
        raise ValueError(f"No server-fusion result found in {log_path}")
    history = [
        {
            "Round": int(match.group("round")) + 1,
            "ACC": float(match.group("acc")),
            "NMI": float(match.group("nmi")),
            "ARI": float(match.group("ari")),
            "F1": float(match.group("f1")),
        }
        for match in matches
    ]
    result = {
        "Recorded_Rounds": len(history),
        "First_Round": min(item["Round"] for item in history),
        "Last_Round": max(item["Round"] for item in history),
    }
    for metric in ("ACC", "NMI", "ARI", "F1"):
        best = max(history, key=lambda item: item[metric])
        result[metric] = best[metric]
        result[f"{metric}_Round"] = best["Round"]
    return result


def collect_latest_runs(log_dir: Path, dataset: str, missrates) -> dict[str, dict]:
    requested_rates = _normalized_rates(missrates)
    candidates: dict[str, list[tuple[float, dict]]] = {}
    for args_path in log_dir.glob("*/args.json"):
        try:
            config = json.loads(args_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Do not mix historical runs (created before this controlled protocol)
        # with the new paper-ablation results.
        if "paper_ablation" not in config:
            continue
        if config.get("name") != dataset or config.get("experiment_mode") != "full":
            continue
        run_rates = _normalized_rates(config.get("missrates", []))
        if run_rates.shape != requested_rates.shape or not np.allclose(
            run_rates, requested_rates, rtol=0.0, atol=1e-12
        ):
            continue
        setting = str(config.get("paper_ablation", "none"))
        logs = list(args_path.parent.glob("*.log"))
        if not logs:
            continue
        log_path = max(logs, key=lambda path: path.stat().st_mtime)
        try:
            metrics = _read_best_metrics(log_path)
        except ValueError:
            continue
        metrics["Run_Folder"] = str(args_path.parent)
        candidates.setdefault(setting, []).append(
            (args_path.parent.stat().st_mtime, metrics)
        )
    return {
        setting: max(entries, key=lambda item: item[0])[1]
        for setting, entries in candidates.items()
    }


def _style_sheet(sheet, widths, metric_start, metric_end):
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for row in sheet.iter_rows(min_row=2, min_col=metric_start, max_col=metric_end):
        for cell in row:
            cell.number_format = "0.0000"
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[chr(64 + index)].width = width
    sheet.freeze_panes = "A2"


def _write_item2_table(workbook, results):
    sheet = workbook.create_sheet("Item2_Reliability")
    sheet.append(
        (
            "Setting", "U_d", "U_m", "R",
            "Max ACC", "ACC Round", "Max NMI", "NMI Round",
        )
    )
    for key, label, use_ud, use_um, use_r in ITEM2_SETTINGS:
        if key not in results:
            raise RuntimeError(
                f"Missing completed run for '{key}'. Run that command before summarizing."
            )
        result = results[key]
        sheet.append(
            (
                label, use_ud, use_um, use_r,
                result["ACC"], result["ACC_Round"],
                result["NMI"], result["NMI_Round"],
            )
        )
    _style_sheet(sheet, (30, 10, 10, 10, 12, 12, 12, 12), 5, 8)


def _write_item8_table(workbook, results):
    sheet = workbook.create_sheet("Item8_Structure")
    sheet.append(("Setting", "Max ACC", "ACC Round", "Max NMI", "NMI Round"))
    for key, label in ITEM8_SETTINGS:
        if key not in results:
            raise RuntimeError(
                f"Missing completed run for '{key}'. Run that command before summarizing."
            )
        result = results[key]
        sheet.append(
            (
                label,
                result["ACC"], result["ACC_Round"],
                result["NMI"], result["NMI_Round"],
            )
        )
    _style_sheet(sheet, (34, 12, 12, 12, 12), 2, 5)


def _write_source_table(workbook, results):
    """Keep every reported value traceable to its completed experiment."""
    sheet = workbook.create_sheet("Run_Sources")
    sheet.append(
        (
            "Ablation Key",
            "Max ACC", "ACC Round",
            "Max NMI", "NMI Round",
            "Max ARI", "ARI Round",
            "Max F1", "F1 Round",
            "Recorded Rounds", "First Round", "Last Round",
            "Run Folder",
        )
    )
    for key in sorted(results):
        result = results[key]
        sheet.append(
            (
                key,
                result["ACC"], result["ACC_Round"],
                result["NMI"], result["NMI_Round"],
                result["ARI"], result["ARI_Round"],
                result["F1"], result["F1_Round"],
                result["Recorded_Rounds"],
                result["First_Round"],
                result["Last_Round"],
                result["Run_Folder"],
            )
        )
    _style_sheet(
        sheet,
        (32, 12, 12, 12, 12, 12, 12, 12, 12, 16, 14, 14, 80),
        2,
        12,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export the latest completed paper ablations to one workbook."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--missrates", type=float, nargs="+", required=True)
    parser.add_argument("--log-dir", default="log_new")
    parser.add_argument("--output", default="paper_ablation_summary.xlsx")
    args = parser.parse_args()

    results = collect_latest_runs(Path(args.log_dir), args.name, args.missrates)
    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_item2_table(workbook, results)
    _write_item8_table(workbook, results)
    _write_source_table(workbook, results)
    output_path = Path(args.output).resolve()
    workbook.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
