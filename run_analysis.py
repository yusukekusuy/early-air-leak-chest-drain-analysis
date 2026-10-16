from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from functions import (
    compute_nagelkerke_r2,
    run_forward_layered_regressions,
    run_layered_residual_lingam,
)


# ============================================================
# Settings
# ============================================================

INPUT_FILE = "data.xlsx"
OUTPUT_FILE = "results.xlsx"

NORMALIZE = True
N_BOOT = 100
EDGE_THRESHOLD = 0.60
MIN_CAUSAL_EFFECT = 0.01
RANDOM_STATE = 0

START_LAYER = 3
END_LAYER = 6


# ============================================================
# Layer definition
# ============================================================

layers = {
    "L1": [
        "Age",
        "Sex",
        "BMI",
        "Smoking index",
        "COPD",
        "Interstitial pneumonia",
        "ASA-PS",
    ],
    "L2": [
        "%VC",
        "FEV1.0%",
        "DLCO",
        "Tumor size",
        "Albumin",
        "Hemoglobin",
        "eGFR",
        "Glucose",
        "HbA1c",
        "CRP",
    ],
    "L3": [
        "Robotic surgery",
        "Lobectomy or more",
        "Lymph-node dissection",
    ],
    "L4": [
        "Pleural adhesion",
        "Incomplete fissure",
        "Intraoperative air leak",
    ],
    "L5": [
        "Early air leak (POD1)",
    ],
    "L6": [
        "Days to drain removal",
    ],
}


# ============================================================
# Helper functions
# ============================================================

def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file."""
    path = Path(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "cp932"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue

    raise ValueError(f"Unsupported or unreadable input file: {path}")


def add_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add missing columns with NaN values."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_unified_edge_table(
    l1_l2_edges: pd.DataFrame,
    l2_l2_edges: pd.DataFrame,
    forward_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all edge tables into a single table."""

    common_columns = [
        "from_layer",
        "to_layer",
        "from",
        "to",
        "model_type",
        "coef",
        "probability",
        "median_direct_effect",
    ]

    l1_l2 = l1_l2_edges.copy()
    l1_l2["model_type"] = "linear"

    l2_l2 = l2_l2_edges.copy()
    l2_l2["model_type"] = "DirectLiNGAM"

    forward = forward_edges.copy()

    unified = pd.concat(
        [
            add_missing_columns(l1_l2, common_columns)[common_columns],
            add_missing_columns(l2_l2, common_columns)[common_columns],
            add_missing_columns(forward, common_columns)[common_columns],
        ],
        axis=0,
        ignore_index=True,
    )

    return unified


# ============================================================
# Main analysis
# ============================================================

def main() -> None:
    df = read_table(INPUT_FILE)
    required_columns = [column for layer in layers.values() for column in layer]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required English columns: {missing_columns}")

    lingam_result = run_layered_residual_lingam(
        df=df,
        L1=layers["L1"],
        L2=layers["L2"],
        normalize=NORMALIZE,
        n_boot=N_BOOT,
        edge_threshold=EDGE_THRESHOLD,
        min_causal_effect=MIN_CAUSAL_EFFECT,
        random_state=RANDOM_STATE,
    )

    forward_result = run_forward_layered_regressions(
        df=df,
        layers=layers,
        start_layer=START_LAYER,
        end_layer=END_LAYER,
        normalize=NORMALIZE,
        n_boot=N_BOOT,
        edge_threshold=EDGE_THRESHOLD,
        min_causal_effect=MIN_CAUSAL_EFFECT,
        random_state=RANDOM_STATE,
    )

    l1_l2_edges = lingam_result["edges"]["L1_to_L2"]
    l2_l2_edges = lingam_result["edges"]["L2_to_L2_selected"]
    forward_edges = forward_result["edges"]["all_forward_regression_edges"]

    all_edges = build_unified_edge_table(
        l1_l2_edges=l1_l2_edges,
        l2_l2_edges=l2_l2_edges,
        forward_edges=forward_edges,
    )

    # Export only the two model-fit values reported in the manuscript.
    leak_model = forward_result["regressions"]["L5"][layers["L5"][0]]
    duration_model = forward_result["regressions"]["L6"][layers["L6"][0]]

    duration_pred = duration_model["y_pred"].to_numpy()
    if duration_model["y_scaler"] is not None:
        duration_pred = duration_model["y_scaler"].inverse_transform(
            duration_pred.reshape(-1, 1)
        ).ravel()

    fit_table = pd.DataFrame([
        {
            "layer": "L5",
            "target": layers["L5"][0],
            "n_samples_used": len(leak_model["used_index"]),
            "metric": "Nagelkerke pseudo-R2",
            "value": compute_nagelkerke_r2(
                leak_model["y_true"].to_numpy(),
                leak_model["y_pred"].to_numpy(),
            ),
        },
        {
            "layer": "L6",
            "target": layers["L6"][0],
            "n_samples_used": len(duration_model["used_index"]),
            "metric": "R2",
            "value": float(r2_score(duration_model["y_true"], duration_pred)),
        },
    ])

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        all_edges.to_excel(writer, sheet_name="All_Edges", index=False)
        fit_table.to_excel(writer, sheet_name="Model_Fit", index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Number of edges: {len(all_edges)}")


if __name__ == "__main__":
    main()
