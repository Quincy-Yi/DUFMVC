"""Export per-round dual-uncertainty histories to Excel and PNG."""

from __future__ import annotations

import os
import re
import copy
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


_HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
_HEADER_FONT = Font(color="FFFFFF", bold=True)
_LIGHT_FILL = PatternFill("solid", fgColor="D9EAF7")
_SCIENTIFIC_FORMAT = "0.000000E+00"


def _format_rate(rate: float) -> str:
    """Return a compact, stable representation suitable for filenames."""
    return np.format_float_positional(float(rate), trim="-")


def _safe_filename_component(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", str(value)).strip(" .")
    return value or "dataset"


def _relative_to_warmup(values: np.ndarray, warmup_rounds: int) -> np.ndarray:
    """Scale a curve by its initial-reference mean for display only."""
    values = np.asarray(values, dtype=float)
    baseline_count = min(values.size, max(1, int(warmup_rounds)))
    baseline = float(np.mean(values[:baseline_count]))
    if abs(baseline) > np.finfo(float).eps:
        return values / baseline

    # A zero reference mean cannot be used as a denominator. Keep an all-zero
    # curve at zero; otherwise use the largest magnitude as a display fallback.
    max_abs = float(np.max(np.abs(values)))
    if max_abs <= np.finfo(float).eps:
        return np.zeros_like(values)
    return values / max_abs


def _centered_moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Return a short centered moving average while preserving all endpoints."""
    values = np.asarray(values, dtype=float)
    if values.size < 3 or window <= 1:
        return values.copy()
    radius = max(1, int(window) // 2)
    return np.asarray(
        [
            np.mean(values[max(0, index - radius) : min(values.size, index + radius + 1)])
            for index in range(values.size)
        ],
        dtype=float,
    )


class UncertaintyExperimentRecorder:
    """Collect client uncertainties and export an auditable experiment artifact.

    Communication rounds are stored as one-based values in both the workbook and
    the plot because that is the convention normally used in experiment figures.
    """

    def __init__(
        self,
        dataset_name: str,
        miss_rates: Sequence[float],
        output_dir: Union[str, os.PathLike],
        experiment_mode: str,
        warmup_rounds: int = 5,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.miss_rates = tuple(float(rate) for rate in miss_rates)
        if not self.miss_rates:
            raise ValueError("miss_rates must contain at least one client/view rate.")
        if not np.all(np.isfinite(self.miss_rates)):
            raise ValueError("miss_rates contains NaN or infinity.")

        self.client_count = len(self.miss_rates)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_mode = str(experiment_mode)
        self.warmup_rounds = max(0, int(warmup_rounds))
        self._records: list[dict[str, object]] = []

        rate_token = "-".join(_format_rate(rate) for rate in self.miss_rates)
        base_name = (
            f"{_safe_filename_component(self.dataset_name)}_"
            f"{_safe_filename_component(rate_token)}"
        )
        self.excel_path = self.output_dir / f"{base_name}.xlsx"
        self.plot_path = self.output_dir / f"{base_name}.png"

    @property
    def completed_rounds(self) -> int:
        return len(self._records)

    def record_round(
        self,
        communication_round: int,
        client_ud: Iterable[float],
        client_um: Iterable[float],
        acc: float,
        nmi: float,
        ari: float,
        f1: float,
        client_ud_pre: Optional[Iterable[float]] = None,
        client_um_pre: Optional[Iterable[float]] = None,
        client_gate_mean: Optional[Iterable[float]] = None,
        client_gate_max: Optional[Iterable[float]] = None,
        client_um_loss: Optional[Iterable[float]] = None,
        client_reliability: Optional[Iterable[float]] = None,
        qaa_max_weight: float = 0.0,
        qaa_weight_entropy: float = 0.0,
        recovery_kind: str = "unknown",
    ) -> None:
        """Record one completed communication round.

        ``communication_round`` is the one-based round number presented to users.
        """
        round_number = int(communication_round)
        if round_number <= 0:
            raise ValueError("communication_round must be a positive one-based integer.")
        if self._records and round_number <= int(self._records[-1]["round"]):
            raise ValueError("communication_round values must be strictly increasing.")

        ud_post = np.asarray(list(client_ud), dtype=float).reshape(-1)
        um_post = np.asarray(list(client_um), dtype=float).reshape(-1)
        ud_pre = (
            ud_post.copy()
            if client_ud_pre is None
            else np.asarray(list(client_ud_pre), dtype=float).reshape(-1)
        )
        um_pre = (
            um_post.copy()
            if client_um_pre is None
            else np.asarray(list(client_um_pre), dtype=float).reshape(-1)
        )
        uncertainty_vectors = (ud_pre, um_pre, ud_post, um_post)
        if any(vector.size != self.client_count for vector in uncertainty_vectors):
            raise ValueError(
                "client uncertainty length must match the number of missing rates "
                f"({self.client_count})."
            )
        if not all(np.all(np.isfinite(vector)) for vector in uncertainty_vectors):
            raise ValueError("client uncertainty values contain NaN or infinity.")
        gate_mean = (
            np.zeros(self.client_count, dtype=float)
            if client_gate_mean is None
            else np.asarray(list(client_gate_mean), dtype=float).reshape(-1)
        )
        gate_max = (
            np.zeros(self.client_count, dtype=float)
            if client_gate_max is None
            else np.asarray(list(client_gate_max), dtype=float).reshape(-1)
        )
        um_loss = (
            np.zeros(self.client_count, dtype=float)
            if client_um_loss is None
            else np.asarray(list(client_um_loss), dtype=float).reshape(-1)
        )
        reliability = (
            np.zeros(self.client_count, dtype=float)
            if client_reliability is None
            else np.asarray(list(client_reliability), dtype=float).reshape(-1)
        )
        diagnostic_vectors = (gate_mean, gate_max, um_loss, reliability)
        if any(vector.size != self.client_count for vector in diagnostic_vectors):
            raise ValueError(
                "client diagnostic length must match the number of missing rates "
                f"({self.client_count})."
            )
        if not all(np.all(np.isfinite(vector)) for vector in diagnostic_vectors):
            raise ValueError("client diagnostics contain NaN or infinity.")
        performance = np.asarray(
            [acc, nmi, ari, f1, qaa_max_weight, qaa_weight_entropy], dtype=float
        )
        if not np.all(np.isfinite(performance)):
            raise ValueError("performance or aggregation diagnostics contain NaN/infinity.")

        self._records.append(
            {
                "round": round_number,
                "client_ud_pre": ud_pre.copy(),
                "client_um_pre": um_pre.copy(),
                "client_ud_post": ud_post.copy(),
                "client_um_post": um_post.copy(),
                "global_mean_ud_pre": float(np.mean(ud_pre)),
                "global_mean_um_pre": float(np.mean(um_pre)),
                "global_mean_ud_post": float(np.mean(ud_post)),
                "global_mean_um_post": float(np.mean(um_post)),
                "client_gate_mean": gate_mean.copy(),
                "client_gate_max": gate_max.copy(),
                "client_um_loss": um_loss.copy(),
                "client_reliability": reliability.copy(),
                "global_mean_gate": float(np.mean(gate_mean)),
                "global_max_gate": float(np.max(gate_max)),
                "global_mean_um_loss": float(np.mean(um_loss)),
                "global_mean_reliability": float(np.mean(reliability)),
                "qaa_max_weight": float(qaa_max_weight),
                "qaa_weight_entropy": float(qaa_weight_entropy),
                "recovery_kind": str(recovery_kind),
                "acc": float(acc),
                "nmi": float(nmi),
                "ari": float(ari),
                "f1": float(f1),
            }
        )

    def snapshot_records(self) -> list[dict[str, object]]:
        """Return an isolated checkpoint-safe copy of the recorded history."""
        return copy.deepcopy(self._records)

    def restore_records(self, records: Iterable[dict[str, object]]) -> None:
        """Restore validated history saved by the shared warm-up checkpoint."""
        self._records = []
        for record in records:
            self.record_round(
                communication_round=int(record["round"]),
                client_ud=record["client_ud_post"],
                client_um=record["client_um_post"],
                acc=float(record["acc"]),
                nmi=float(record["nmi"]),
                ari=float(record["ari"]),
                f1=float(record["f1"]),
                client_ud_pre=record["client_ud_pre"],
                client_um_pre=record["client_um_pre"],
                client_gate_mean=record["client_gate_mean"],
                client_gate_max=record["client_gate_max"],
                client_um_loss=record["client_um_loss"],
                client_reliability=record.get(
                    "client_reliability",
                    np.zeros(self.client_count, dtype=float),
                ),
                qaa_max_weight=float(record["qaa_max_weight"]),
                qaa_weight_entropy=float(record["qaa_weight_entropy"]),
                recovery_kind=str(record["recovery_kind"]),
            )

    def export(self) -> tuple[Path, Path]:
        """Write/refresh the Excel workbook and the global-mean PNG curve."""
        if not self._records:
            raise RuntimeError("No uncertainty records are available for export.")

        self._write_plot()
        self._write_workbook()
        return self.excel_path, self.plot_path

    def _write_workbook(self) -> None:
        workbook = Workbook()
        history = workbook.active
        history.title = "Uncertainty_History"
        per_client = workbook.create_sheet("Per_Client")
        metadata = workbook.create_sheet("Metadata")

        history_headers = ["Communication_Round", "Phase"]
        for client_index in range(self.client_count):
            history_headers.extend(
                [
                    f"Client_{client_index + 1}_U_d_Pre",
                    f"Client_{client_index + 1}_U_m_Pre",
                    f"Client_{client_index + 1}_U_d_Post",
                    f"Client_{client_index + 1}_U_m_Post",
                ]
            )
        history_headers.extend(
            [
                "Global_Mean_U_d_Pre",
                "Global_Mean_U_m_Pre",
                "Global_Mean_U_d_Post",
                "Global_Mean_U_m_Post",
                "ACC",
                "NMI",
                "ARI",
                "F1",
                "Mean_Recovery_Gate",
                "Max_Recovery_Gate",
                "Mean_Model_Alignment_Loss",
                "QAA_Max_Weight",
                "QAA_Normalized_Entropy",
                "Mean_Reserved_Reliability_Diagnostic",
                "Recovery_Kind",
            ]
        )
        history.append(history_headers)

        for record in self._records:
            round_number = int(record["round"])
            phase = (
                "Initial_Reference"
                if round_number <= self.warmup_rounds
                else "Main_Training"
            )
            row: list[Union[float, int, str]] = [round_number, phase]
            ud_pre = np.asarray(record["client_ud_pre"], dtype=float)
            um_pre = np.asarray(record["client_um_pre"], dtype=float)
            ud_post = np.asarray(record["client_ud_post"], dtype=float)
            um_post = np.asarray(record["client_um_post"], dtype=float)
            for client_index in range(self.client_count):
                row.extend(
                    [
                        float(ud_pre[client_index]),
                        float(um_pre[client_index]),
                        float(ud_post[client_index]),
                        float(um_post[client_index]),
                    ]
                )
            row.extend(
                [
                    float(record["global_mean_ud_pre"]),
                    float(record["global_mean_um_pre"]),
                    float(record["global_mean_ud_post"]),
                    float(record["global_mean_um_post"]),
                    float(record["acc"]),
                    float(record["nmi"]),
                    float(record["ari"]),
                    float(record["f1"]),
                    float(record["global_mean_gate"]),
                    float(record["global_max_gate"]),
                    float(record["global_mean_um_loss"]),
                    float(record["qaa_max_weight"]),
                    float(record["qaa_weight_entropy"]),
                    float(record["global_mean_reliability"]),
                    str(record["recovery_kind"]),
                ]
            )
            history.append(row)

        per_client_headers = [
            "Communication_Round",
            "Phase",
            "Client",
            "Missing_Rate",
            "Client_U_d_Pre",
            "Client_U_m_Pre",
            "Client_U_d_Post",
            "Client_U_m_Post",
            "Recovery_Gate_Mean",
            "Recovery_Gate_Max",
            "Model_Alignment_Loss",
            "Reserved_Reliability_Diagnostic",
        ]
        per_client.append(per_client_headers)
        for record in self._records:
            round_number = int(record["round"])
            phase = (
                "Initial_Reference"
                if round_number <= self.warmup_rounds
                else "Main_Training"
            )
            ud_pre = np.asarray(record["client_ud_pre"], dtype=float)
            um_pre = np.asarray(record["client_um_pre"], dtype=float)
            ud_post = np.asarray(record["client_ud_post"], dtype=float)
            um_post = np.asarray(record["client_um_post"], dtype=float)
            gate_mean = np.asarray(record["client_gate_mean"], dtype=float)
            gate_max = np.asarray(record["client_gate_max"], dtype=float)
            um_loss = np.asarray(record["client_um_loss"], dtype=float)
            reliability = np.asarray(record["client_reliability"], dtype=float)
            for client_index in range(self.client_count):
                per_client.append(
                    [
                        round_number,
                        phase,
                        f"Client_{client_index + 1}",
                        float(self.miss_rates[client_index]),
                        float(ud_pre[client_index]),
                        float(um_pre[client_index]),
                        float(ud_post[client_index]),
                        float(um_post[client_index]),
                        float(gate_mean[client_index]),
                        float(gate_max[client_index]),
                        float(um_loss[client_index]),
                        float(reliability[client_index]),
                    ]
                )

        metadata_rows = [
            ("Field", "Value"),
            ("Dataset", self.dataset_name),
            ("Experiment_Mode", self.experiment_mode),
            ("Client_Count", self.client_count),
            ("Missing_Rates", ", ".join(_format_rate(r) for r in self.miss_rates)),
            ("Round_Numbering", "1-based"),
            ("Warmup_Rounds", self.warmup_rounds),
            ("Client_Numbering", "Client_1 corresponds to internal view index 0"),
            ("Global_Mean_Definition", "Arithmetic mean across all clients/views"),
            (
                "PRE_Measurement",
                "Measured at the start of each communication round before "
                "local training; U_m is evaluated against the previous inertial "
                "global target.",
            ),
            (
                "POST_Measurement",
                "Measured immediately after local training and before server "
                "aggregation; these values drive QAA and the round-end CAR update.",
            ),
            (
                "Exported_Curve_Source",
                "POST synchronized global means and raw server-fusion ACC",
            ),
            (
                "Plot_Uncertainty_Scale",
                "Display only: each POST global uncertainty is divided by its "
                f"first {max(1, self.warmup_rounds)} round(s) reference mean "
                "(zero-baseline fallback: maximum absolute value)",
            ),
            (
                "Plot_Smoothing",
                "Bold curves use a centered 3-round moving average; faint curves "
                "show every raw round and Excel retains all raw values.",
            ),
            ("Plot_ACC_Scale", "Raw ACC on an adaptive right y-axis bounded by [0, 1]"),
            (
                "Recovery_Gate",
                "Paper CAR coefficient g=U_d*(1-U_m) for Full; the two single-factor "
                "ablation modes retain only their named factor. Observed rows are "
                "always preserved.",
            ),
            (
                "Model_Alignment_Loss",
                "Graph-reconstruction loss plus reliability-weighted KL alignment; "
                "sample reliability is stop-gradient (1-U_m). There is no separate "
                "uncertainty, hard-label, or cluster-balance objective.",
            ),
            (
                "Model_Uncertainty_Target",
                "The server Student-t assignment is squared and row-normalized "
                "exactly as specified in the paper, then propagated by the dynamic "
                "inertia update. No frequency balancing or extra sharpening is used.",
            ),
            (
                "Reserved_Reliability_Diagnostic",
                "Compatibility-only column fixed to one; it is not used by the "
                "paper implementation.",
            ),
            (
                "Performance_Source",
                "Server KMeans on the quality-weighted fused representation.",
            ),
            (
                "QAA_Normalized_Entropy",
                "Client-weight entropy divided by log(client_count); 1 means uniform.",
            ),
        ]
        for row in metadata_rows:
            metadata.append(row)

        self._style_header(history, history.max_column)
        self._style_header(per_client, per_client.max_column)
        self._style_header(metadata, metadata.max_column)

        history.freeze_panes = "A2"
        per_client.freeze_panes = "A2"
        metadata.freeze_panes = "A2"
        history.sheet_view.showGridLines = False
        per_client.sheet_view.showGridLines = False
        metadata.sheet_view.showGridLines = False
        history.auto_filter.ref = history.dimensions
        per_client.auto_filter.ref = per_client.dimensions

        history.column_dimensions["A"].width = 22
        history.column_dimensions["B"].width = 14
        uncertainty_last_column = 2 + 4 * self.client_count + 4
        for column_index in range(3, uncertainty_last_column + 1):
            history.column_dimensions[history.cell(1, column_index).column_letter].width = 20
            for row_index in range(2, history.max_row + 1):
                history.cell(row_index, column_index).number_format = _SCIENTIFIC_FORMAT
        performance_first_column = uncertainty_last_column + 1
        performance_last_column = performance_first_column + 3
        for column_index in range(performance_first_column, performance_last_column + 1):
            history.column_dimensions[history.cell(1, column_index).column_letter].width = 12
            for row_index in range(2, history.max_row + 1):
                history.cell(row_index, column_index).number_format = "0.0000"
        for column_index in range(performance_last_column + 1, history.max_column):
            history.column_dimensions[history.cell(1, column_index).column_letter].width = 24
            for row_index in range(2, history.max_row + 1):
                history.cell(row_index, column_index).number_format = "0.000000"
        history.column_dimensions[
            history.cell(1, history.max_column).column_letter
        ].width = 20
        for row_index in range(2, history.max_row + 1):
            history.cell(row_index, 2).alignment = Alignment(horizontal="center")

        per_client_widths = (22, 14, 14, 16, 18, 18, 18, 18, 20, 20, 26, 22)
        for column_index, width in enumerate(per_client_widths, start=1):
            per_client.column_dimensions[
                per_client.cell(1, column_index).column_letter
            ].width = width
        for row_index in range(2, per_client.max_row + 1):
            per_client.cell(row_index, 2).alignment = Alignment(horizontal="center")
            per_client.cell(row_index, 4).number_format = "0.0000"
            for column_index in range(5, 9):
                per_client.cell(row_index, column_index).number_format = _SCIENTIFIC_FORMAT
            for column_index in range(9, 13):
                per_client.cell(row_index, column_index).number_format = "0.000000"

        metadata.column_dimensions["A"].width = 28
        metadata.column_dimensions["B"].width = 82
        for row_index in range(2, metadata.max_row + 1):
            metadata.cell(row_index, 1).fill = _LIGHT_FILL
            metadata.cell(row_index, 1).font = Font(bold=True)
            metadata.cell(row_index, 2).alignment = Alignment(
                vertical="top", wrap_text=True
            )
        for row_index in range(2, metadata.max_row + 1):
            if metadata.cell(row_index, 1).value in {
                "PRE_Measurement",
                "POST_Measurement",
                "Plot_Uncertainty_Scale",
                "Plot_Smoothing",
            }:
                metadata.row_dimensions[row_index].height = 34

        history_table = Table(displayName="UncertaintyHistoryTable", ref=history.dimensions)
        history_table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True, showFirstColumn=False
        )
        history.add_table(history_table)

        per_client_table = Table(displayName="PerClientUncertaintyTable", ref=per_client.dimensions)
        per_client_table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True, showFirstColumn=False
        )
        per_client.add_table(per_client_table)

        plot_image_stream = BytesIO(self.plot_path.read_bytes())
        plot_image = ExcelImage(plot_image_stream)
        plot_image.width = 900
        plot_image.height = 520
        history.add_image(plot_image, history.cell(2, history.max_column + 2).coordinate)

        temp_path = self.excel_path.with_suffix(".tmp.xlsx")
        try:
            workbook.save(temp_path)
        finally:
            plot_image_stream.close()
        os.replace(temp_path, self.excel_path)

    def _write_plot(self) -> None:
        rounds = np.asarray([record["round"] for record in self._records], dtype=int)
        mean_ud = np.asarray(
            [record["global_mean_ud_post"] for record in self._records], dtype=float
        )
        mean_um = np.asarray(
            [record["global_mean_um_post"] for record in self._records], dtype=float
        )
        acc = np.asarray([record["acc"] for record in self._records], dtype=float)

        relative_ud = _relative_to_warmup(mean_ud, self.warmup_rounds)
        relative_um = _relative_to_warmup(mean_um, self.warmup_rounds)
        smooth_ud = _centered_moving_average(relative_ud)
        smooth_um = _centered_moving_average(relative_um)
        smooth_acc = _centered_moving_average(acc)

        fig, axis_uncertainty = plt.subplots(figsize=(9.8, 5.6))
        axis_acc = axis_uncertainty.twinx()
        color_ud = "#0072B2"
        color_um = "#D55E00"
        color_acc = "#009E73"
        marker_interval = max(1, int(np.ceil(rounds.size / 12)))

        # Faint lines retain all raw round-to-round variation.  The bold
        # three-round curves make the scientific trend readable without
        # replacing or altering any exported raw measurement.
        axis_uncertainty.plot(
            rounds, relative_ud, color=color_ud, linewidth=1.0, alpha=0.24, zorder=2
        )
        axis_uncertainty.plot(
            rounds, relative_um, color=color_um, linewidth=1.0, alpha=0.24, zorder=2
        )
        axis_acc.plot(
            rounds, acc, color=color_acc, linewidth=1.0, alpha=0.24, zorder=2
        )
        line_ud = axis_uncertainty.plot(
            rounds,
            smooth_ud,
            color=color_ud,
            marker="o",
            markevery=marker_interval,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.5,
            markersize=5.5,
            label=r"$U_d$ (post, 3-round mean)",
            zorder=3,
        )[0]
        line_um = axis_uncertainty.plot(
            rounds,
            smooth_um,
            color=color_um,
            marker="s",
            markevery=marker_interval,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.5,
            linestyle="--",
            markersize=5.5,
            label=r"$U_m$ (post, 3-round mean)",
            zorder=3,
        )[0]
        line_acc = axis_acc.plot(
            rounds,
            smooth_acc,
            color=color_acc,
            marker="^",
            markevery=marker_interval,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.5,
            linestyle="-.",
            markersize=5.5,
            label="Server ACC (3-round mean)",
            zorder=4,
        )[0]

        axis_uncertainty.set_xlabel("Communication round")
        axis_uncertainty.set_ylabel("Relative uncertainty (initial mean = 1.0)")
        axis_uncertainty.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis_uncertainty.axhline(
            1.0, color="#8C8C8C", linewidth=1.0, linestyle=":", alpha=0.75, zorder=1
        )
        axis_uncertainty.grid(
            axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65, zorder=0
        )

        if self.warmup_rounds > 0:
            warmup_end = min(self.warmup_rounds, int(rounds[-1]))
            axis_uncertainty.axvspan(
                float(rounds[0]) - 0.5,
                float(warmup_end) + 0.5,
                color="#BDBDBD",
                alpha=0.13,
                linewidth=0,
                zorder=0,
            )
            axis_uncertainty.text(
                (float(rounds[0]) + float(warmup_end)) / 2.0,
                0.97,
                "Initial reference",
                transform=axis_uncertainty.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="#666666",
            )
            if int(rounds[-1]) > self.warmup_rounds:
                axis_uncertainty.axvline(
                    self.warmup_rounds + 0.5,
                    color="#777777",
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.75,
                    zorder=1,
                )

        uncertainty_min = float(min(np.min(relative_ud), np.min(relative_um), 1.0))
        uncertainty_max = float(max(np.max(relative_ud), np.max(relative_um), 1.0))
        uncertainty_span = max(uncertainty_max - uncertainty_min, 0.2)
        uncertainty_bottom = max(0.0, uncertainty_min - 0.12 * uncertainty_span)
        axis_uncertainty.set_ylim(
            uncertainty_bottom, uncertainty_max + 0.12 * uncertainty_span
        )
        axis_uncertainty.margins(x=0.025)

        axis_acc.set_ylabel("Server ACC", color=color_acc)
        axis_acc.tick_params(axis="y", labelcolor=color_acc)
        acc_min = float(np.min(acc))
        acc_max = float(np.max(acc))
        acc_span = max(acc_max - acc_min, 0.04)
        acc_bottom = max(0.0, acc_min - 0.20 * acc_span)
        acc_top = min(1.0, acc_max + 0.20 * acc_span)
        if acc_top - acc_bottom < 0.02:
            acc_bottom = max(0.0, acc_min - 0.02)
            acc_top = min(1.0, acc_max + 0.02)
        axis_acc.set_ylim(acc_bottom, acc_top)
        axis_acc.spines["right"].set_color(color_acc)
        axis_uncertainty.spines["top"].set_visible(False)
        axis_acc.spines["top"].set_visible(False)

        legend_lines = [line_ud, line_um, line_acc]
        axis_uncertainty.legend(
            legend_lines,
            [line.get_label() for line in legend_lines],
            loc="upper right",
            frameon=False,
            ncol=1,
            fontsize=9,
        )

        if np.allclose(self.miss_rates, self.miss_rates[0]):
            rate_text = f"{_format_rate(self.miss_rates[0])} (all views)"
            rate_label = "missing rate"
        else:
            rate_text = ", ".join(_format_rate(rate) for rate in self.miss_rates)
            rate_label = "missing rates"
        axis_uncertainty.set_title(
            f"{self.dataset_name} | mode: {self.experiment_mode} | "
            f"{rate_label}: {rate_text}",
            pad=14,
            fontsize=12,
            fontweight="semibold",
        )
        fig.tight_layout()

        temp_path = self.plot_path.with_suffix(".tmp.png")
        fig.savefig(temp_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        os.replace(temp_path, self.plot_path)

    @staticmethod
    def _style_header(worksheet, column_count: int) -> None:
        for column_index in range(1, column_count + 1):
            cell = worksheet.cell(1, column_index)
            cell.fill = _HEADER_FILL
            cell.font = _HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")
        worksheet.row_dimensions[1].height = 24
