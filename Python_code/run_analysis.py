from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from function_analysis import run_forward_layered_regressions, run_layered_residual_lingam


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
        "年齢",
        "性別",
        "BMI",
        "喫煙指数",
        "併存症COPD",
        "併存症IP",
        "術前全身評価_ASA PS",
    ],
    "L2": [
        "肺機能.%VC",
        "肺機能.FEV1.0%",
        "肺機能.DLCO",
        "推計腫瘍サイズ",
        "術前_Alb",
        "術前_Hb",
        "術前_eGFR",
        "術前_BS",
        "術前_HbA1c",
        "術前_CRP",
    ],
    "L3": [
        "術式_ロボット操作",
        "術式_lobectomy_or_more",
        "術式_alymph_dissection",
    ],
    "L4": [
        "術中所見_aadhesion",
        "術中所見_aincomplete_fissure",
        "術中所見_air_leak",
    ],
    "L5": [
        "初期リーク（POD1まで）",
    ],
    "L6": [
        "Y=ドレーン抜去日数",
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
        "odds_ratio",
        "probability",
        "count",
        "mean_direct_effect",
        "median_direct_effect",
        "target_r2",
        "accuracy",
        "auc",
        "r2",
    ]

    l1_l2 = l1_l2_edges.copy()
    l1_l2["edge_group"] = "L1_to_L2"
    l1_l2["model_type"] = "linear"

    l2_l2 = l2_l2_edges.copy()
    l2_l2["edge_group"] = "L2_to_L2"
    l2_l2["model_type"] = "DirectLiNGAM"

    forward = forward_edges.copy()
    forward["edge_group"] = "Forward_Edges"

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

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        all_edges.to_excel(writer, sheet_name="All_Edges", index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Number of edges: {len(all_edges)}")


if __name__ == "__main__":
    main()
