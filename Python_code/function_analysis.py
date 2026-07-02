from __future__ import annotations

import pandas as pd
import numpy as np
import lingam

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score


__all__ = [
    "is_binary_series",
    "detect_binary_columns",
    "make_residual_col_name",
    "run_linear_regression",
    "run_auto_regression",
    "run_direct_lingam_with_bootstrap",
    "make_l2_residuals",
    "run_layered_residual_lingam",
    "make_lingam_adjusted_l2_residuals",
    "run_forward_layered_regressions",
]


# ============================================================
# Utility
# ============================================================

def is_binary_series(s: pd.Series) -> bool:
    vals = set(s.dropna().unique())
    return vals.issubset({0, 1, 0.0, 1.0}) and len(vals) <= 2


def detect_binary_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if is_binary_series(df[col])]


def make_residual_col_name(layer_name: str, col: str) -> str:
    return f"{layer_name}__{col}__residual"


# ============================================================
# Linear regression for L1 -> L2 residualization
# ============================================================

def run_linear_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    normalize: bool = False,
):
    """Fit a linear regression model for one target variable.

    Continuous predictors are standardized only when ``normalize=True``.
    Binary predictors are kept on their original 0/1 scale. The target
    variable is also standardized when ``normalize=True``.
    """

    use_cols = feature_cols + [target_col]
    data = df[use_cols].copy().dropna()

    if data.empty:
        raise ValueError("No data are available after dropping missing values.")

    X = data[feature_cols].copy().astype(float)
    y = data[target_col].copy()

    binary_feature_cols = []
    continuous_feature_cols = []

    for col in feature_cols:
        unique_vals = set(X[col].dropna().unique())

        if unique_vals.issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
            binary_feature_cols.append(col)
        else:
            continuous_feature_cols.append(col)

    x_scaler = None
    y_scaler = None

    if normalize:
        if continuous_feature_cols:
            x_scaler = StandardScaler()
            X.loc[:, continuous_feature_cols] = x_scaler.fit_transform(
                X[continuous_feature_cols]
            )

        y_scaler = StandardScaler()
        y_for_model = y_scaler.fit_transform(
            y.astype(float).to_numpy().reshape(-1, 1)
        ).ravel()
    else:
        y_for_model = y.astype(float).to_numpy()

    model = LinearRegression()
    model.fit(X, y_for_model)

    y_pred = model.predict(X)
    residuals = y_for_model - y_pred
    r2 = r2_score(y_for_model, y_pred)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": model.coef_,
    })

    result = {
        "model": model,
        "used_index": data.index,
        "y_true": pd.Series(y_for_model, index=data.index, name=target_col),
        "y_pred": pd.Series(y_pred, index=data.index, name=f"{target_col}_pred"),
        "residuals": pd.Series(
            residuals,
            index=data.index,
            name=f"{target_col}_residual",
        ),
        "coef_df": coef_df,
        "intercept": float(model.intercept_),
        "r2": float(r2),
        "binary_feature_cols": binary_feature_cols,
        "continuous_feature_cols": continuous_feature_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }

    return result


# ============================================================
# Auto regression: linear or logistic
# ============================================================

def run_auto_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    normalize: bool = False,
    logistic_max_iter: int = 1000,
):
    """Fit either logistic or linear regression depending on the target type.

    If the target is binary, logistic regression is used and residuals are
    defined as ``y - predicted_probability``. Otherwise, linear regression is
    used and residuals are defined as ``y - predicted_value``.
    """

    use_cols = feature_cols + [target_col]
    data = df[use_cols].copy().dropna()

    if data.empty:
        raise ValueError(f"{target_col}: No data are available after dropping missing values.")

    X = data[feature_cols].copy().astype(float)
    y = data[target_col].copy()

    target_is_binary = is_binary_series(y)

    binary_feature_cols = detect_binary_columns(X, feature_cols)
    continuous_feature_cols = [
        col for col in feature_cols if col not in binary_feature_cols
    ]

    x_scaler = None
    y_scaler = None

    if normalize:
        if continuous_feature_cols:
            x_scaler = StandardScaler()
            X.loc[:, continuous_feature_cols] = x_scaler.fit_transform(
                X[continuous_feature_cols]
            )

    if target_is_binary:
        y_for_model = y.astype(int).to_numpy()

        if len(np.unique(y_for_model)) < 2:
            raise ValueError(
                f"{target_col}: the target is binary, but only one class remains "
                "after dropping missing values."
            )

        model = LogisticRegression(
            max_iter=logistic_max_iter,
            solver="lbfgs",
        )
        model.fit(X, y_for_model)

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        residuals = y_for_model - y_prob

        try:
            auc = roc_auc_score(y_for_model, y_prob)
        except ValueError:
            auc = np.nan

        coef = model.coef_[0]
        intercept = model.intercept_[0]

        metrics = {
            "accuracy": float(accuracy_score(y_for_model, y_pred)),
            "auc": float(auc) if not np.isnan(auc) else np.nan,
        }

        prediction_series = pd.Series(
            y_prob,
            index=data.index,
            name=f"{target_col}_pred_prob",
        )

        residual_series = pd.Series(
            residuals,
            index=data.index,
            name=f"{target_col}_residual",
        )

        model_type = "logistic"

    else:
        y_float = y.astype(float)

        if normalize:
            y_scaler = StandardScaler()
            y_for_model = y_scaler.fit_transform(
                y_float.to_numpy().reshape(-1, 1)
            ).ravel()
        else:
            y_for_model = y_float.to_numpy()

        model = LinearRegression()
        model.fit(X, y_for_model)

        y_pred = model.predict(X)
        residuals = y_for_model - y_pred

        coef = model.coef_
        intercept = model.intercept_

        metrics = {
            "r2": float(r2_score(y_for_model, y_pred)),
        }

        prediction_series = pd.Series(
            y_pred,
            index=data.index,
            name=f"{target_col}_pred",
        )

        residual_series = pd.Series(
            residuals,
            index=data.index,
            name=f"{target_col}_residual",
        )

        model_type = "linear"

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": coef,
    })

    if model_type == "logistic":
        coef_df["odds_ratio"] = np.exp(coef_df["coef"])

    result = {
        "model": model,
        "model_type": model_type,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "used_index": data.index,
        "y_true": pd.Series(y.to_numpy(), index=data.index, name=target_col),
        "y_pred": prediction_series,
        "residuals": residual_series,
        "coef_df": coef_df,
        "intercept": float(intercept),
        "metrics": metrics,
        "target_is_binary": target_is_binary,
        "binary_feature_cols": binary_feature_cols,
        "continuous_feature_cols": continuous_feature_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "normalize": normalize,
    }

    return result


# ============================================================
# DirectLiNGAM for L2 residuals
# ============================================================

def run_direct_lingam_with_bootstrap(
    residual_df: pd.DataFrame,
    n_boot: int = 100,
    edge_threshold: float = 0.70,
    min_causal_effect: float = 0.01,
    random_state: int | None = 0,
):
    """Run DirectLiNGAM on residual data and select stable directions.

    A causal direction is selected when its bootstrap frequency is at least
    ``edge_threshold``.
    """

    if not isinstance(residual_df, pd.DataFrame):
        raise TypeError("residual_df must be a pandas DataFrame.")

    if residual_df.shape[1] < 2:
        raise ValueError("At least two variables are required to run LiNGAM.")

    data = residual_df.copy().dropna()

    if data.empty:
        raise ValueError("No data are available after dropping missing values.")

    try:
        X = data.astype(float).to_numpy()
    except ValueError as e:
        raise ValueError("residual_df must contain numeric variables only.") from e

    labels = list(data.columns)

    model = lingam.DirectLiNGAM(random_state=random_state)
    model.fit(X)

    adjacency_matrix = model.adjacency_matrix_
    causal_order = model.causal_order_

    adjacency_matrix_df = pd.DataFrame(
        adjacency_matrix,
        index=labels,
        columns=labels,
    )

    bootstrap_model = lingam.DirectLiNGAM(random_state=random_state)
    bootstrap_result = bootstrap_model.bootstrap(
        X,
        n_sampling=n_boot,
    )

    causal_direction_counts = bootstrap_result.get_causal_direction_counts(
        min_causal_effect=min_causal_effect,
        split_by_causal_effect_sign=False,
    )

    edge_rows = []

    bootstrap_adjacency_matrices = bootstrap_result.adjacency_matrices_

    for from_idx, to_idx, count in zip(
        causal_direction_counts["from"],
        causal_direction_counts["to"],
        causal_direction_counts["count"],
    ):
        probability = count / n_boot

        effects = bootstrap_adjacency_matrices[:, to_idx, from_idx]

        nonzero_effects = effects[
            np.abs(effects) >= min_causal_effect
        ]

        if len(nonzero_effects) > 0:
            mean_direct_effect = float(np.mean(nonzero_effects))
            median_direct_effect = float(np.median(nonzero_effects))
        else:
            mean_direct_effect = np.nan
            median_direct_effect = np.nan

        edge_rows.append({
            "from": labels[from_idx],
            "to": labels[to_idx],
            "count": int(count),
            "probability": float(probability),
            "mean_direct_effect": mean_direct_effect,
            "median_direct_effect": median_direct_effect,
        })

    all_bootstrap_edges = pd.DataFrame(edge_rows)

    if not all_bootstrap_edges.empty:
        all_bootstrap_edges = all_bootstrap_edges.sort_values(
            by="probability",
            ascending=False,
        ).reset_index(drop=True)

    if all_bootstrap_edges.empty:
        selected_edges = all_bootstrap_edges.copy()
    else:
        selected_edges = all_bootstrap_edges[
            all_bootstrap_edges["probability"] >= edge_threshold
        ].copy().reset_index(drop=True)

    result = {
        "model": model,
        "causal_order": [labels[i] for i in causal_order],
        "adjacency_matrix": adjacency_matrix,
        "adjacency_matrix_df": adjacency_matrix_df,
        "bootstrap_result": bootstrap_result,
        "all_bootstrap_edges": all_bootstrap_edges,
        "selected_edges": selected_edges,
    }

    return result


# ============================================================
# L2 residuals: L2 ~ L1
# ============================================================

def make_l2_residuals(
    df: pd.DataFrame,
    L1: list[str],
    L2: list[str],
    normalize: bool = False,
):
    """Regress each L2 variable on L1 variables and return L2 residuals."""

    required_cols = L1 + L2
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Some columns are missing from df: {missing_cols}")

    if len(L1) == 0:
        raise ValueError("L1 must not be empty.")

    if len(L2) == 0:
        raise ValueError("L2 must not be empty.")

    analysis_df = df[required_cols].copy().dropna()

    if analysis_df.empty:
        raise ValueError("No data are available after dropping missing values.")

    used_index = analysis_df.index

    regression_results = {}
    residual_series_list = []
    l1_to_l2_edge_rows = []

    for target_col in L2:
        reg_result = run_linear_regression(
            df=analysis_df,
            target_col=target_col,
            feature_cols=L1,
            normalize=normalize,
        )

        regression_results[target_col] = reg_result

        residual_series = reg_result["residuals"].copy()
        residual_series.name = target_col
        residual_series_list.append(residual_series)

        coef_df = reg_result["coef_df"].copy()

        for _, row in coef_df.iterrows():
            l1_to_l2_edge_rows.append({
                "from": row["feature"],
                "to": target_col,
                "edge_type": "L1_to_L2_regression",
                "coef": float(row["coef"]),
                "target_r2": float(reg_result["r2"]),
                "normalize": normalize,
                "from_layer": "L1",
                "to_layer": "L2",
            })

    residual_df = pd.concat(residual_series_list, axis=1)
    residual_df = residual_df.loc[used_index]

    l1_to_l2_edges = pd.DataFrame(l1_to_l2_edge_rows)

    result = {
        "layers": {
            "L1": L1,
            "L2": L2,
        },
        "analysis_df": analysis_df,
        "used_index": used_index,
        "residual_df": residual_df,
        "regression_results": regression_results,
        "l1_to_l2_edges": l1_to_l2_edges,
    }

    return result


# ============================================================
# L1 -> L2 residualization + L2 residual LiNGAM
# ============================================================

def run_layered_residual_lingam(
    df: pd.DataFrame,
    L1: list[str],
    L2: list[str],
    normalize: bool = False,
    n_boot: int = 100,
    edge_threshold: float = 0.70,
    min_causal_effect: float = 0.01,
    random_state: int | None = 0,
):
    """Run the two-layer residual-adjusted DirectLiNGAM analysis.

    The function first regresses each L2 variable on L1 variables, then applies
    DirectLiNGAM with bootstrap selection to the resulting L2 residuals.
    """

    residual_result = make_l2_residuals(
        df=df,
        L1=L1,
        L2=L2,
        normalize=normalize,
    )

    residual_df = residual_result["residual_df"]

    lingam_result = run_direct_lingam_with_bootstrap(
        residual_df=residual_df,
        n_boot=n_boot,
        edge_threshold=edge_threshold,
        min_causal_effect=min_causal_effect,
        random_state=random_state,
    )

    selected_l2_edges = lingam_result["selected_edges"].copy()

    if not selected_l2_edges.empty:
        selected_l2_edges["edge_type"] = "L2_to_L2_lingam"
        selected_l2_edges["from_layer"] = "L2"
        selected_l2_edges["to_layer"] = "L2"

    all_l2_edges = lingam_result["all_bootstrap_edges"].copy()

    if not all_l2_edges.empty:
        all_l2_edges["edge_type"] = "L2_to_L2_lingam"
        all_l2_edges["from_layer"] = "L2"
        all_l2_edges["to_layer"] = "L2"

    l1_to_l2_edges = residual_result["l1_to_l2_edges"].copy()

    node_rows = []

    for var in L1:
        node_rows.append({
            "node": var,
            "layer": "L1",
            "node_type": "observed_variable",
        })

    for var in L2:
        node_rows.append({
            "node": var,
            "layer": "L2",
            "node_type": "residualized_variable_in_lingam",
        })

    nodes_df = pd.DataFrame(node_rows)

    result = {
        "layers": {
            "L1": L1,
            "L2": L2,
        },
        "nodes": nodes_df,
        "edges": {
            "L1_to_L2": l1_to_l2_edges,
            "L2_to_L2_selected": selected_l2_edges,
            "L2_to_L2_all_bootstrap": all_l2_edges,
        },
        "residuals": {
            "residual_df": residual_result["residual_df"],
            "analysis_df": residual_result["analysis_df"],
            "used_index": residual_result["used_index"],
        },
        "regressions": residual_result["regression_results"],
        "lingam": {
            "model": lingam_result["model"],
            "causal_order": lingam_result["causal_order"],
            "adjacency_matrix": lingam_result["adjacency_matrix"],
            "adjacency_matrix_df": lingam_result["adjacency_matrix_df"],
            "bootstrap_result": lingam_result["bootstrap_result"],
            "all_bootstrap_edges": lingam_result["all_bootstrap_edges"],
            "selected_edges": lingam_result["selected_edges"],
        },
        "metadata": {
            "normalize": normalize,
            "n_boot": n_boot,
            "edge_threshold": edge_threshold,
            "min_causal_effect": min_causal_effect,
            "random_state": random_state,
            "n_samples_used": len(residual_result["used_index"]),
        },
    }

    return result


# ============================================================
# LiNGAM-adjusted L2 residuals
# ============================================================

def make_lingam_adjusted_l2_residuals(
    l2_residual_df: pd.DataFrame,
    selected_l2_edges: pd.DataFrame,
) -> dict:
    """Remove within-L2 parent effects from L2 residuals.

    ``l2_residual_df`` should contain L2 residuals after adjustment by L1.
    The selected DirectLiNGAM edges define the parent set of each L2 variable.
    Each L2 residual is then regressed on its selected L2 parent residuals, and
    the resulting residual is used as the LiNGAM-adjusted L2 component.
    """

    if not isinstance(l2_residual_df, pd.DataFrame):
        raise TypeError("l2_residual_df must be a pandas DataFrame.")

    if l2_residual_df.empty:
        raise ValueError("l2_residual_df must not be empty.")

    data = l2_residual_df.copy().dropna()

    if data.empty:
        raise ValueError("No data are available after dropping missing values.")

    labels = list(data.columns)

    parent_map: dict[str, list[str]] = {col: [] for col in labels}

    if selected_l2_edges is not None and not selected_l2_edges.empty:
        for _, row in selected_l2_edges.iterrows():
            parent = row["from"]
            child = row["to"]

            if parent in labels and child in labels and parent != child:
                parent_map[child].append(parent)

    adjusted_residuals = pd.DataFrame(index=data.index)
    adjustment_results = {}
    parent_rows = []

    for target_col in labels:
        parent_cols = parent_map.get(target_col, [])

        if len(parent_cols) == 0:
            adjusted_residuals[target_col] = data[target_col]
            adjustment_results[target_col] = {
                "model": None,
                "parents": [],
                "intercept": np.nan,
                "coef_df": pd.DataFrame(columns=["feature", "coef"]),
                "r2": np.nan,
            }
            parent_rows.append({
                "target": target_col,
                "parents": [],
                "n_parents": 0,
                "r2": np.nan,
            })
            continue

        X = data[parent_cols].astype(float)
        y = data[target_col].astype(float)

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        residuals = y.to_numpy() - y_pred

        adjusted_residuals[target_col] = residuals

        r2 = float(r2_score(y, y_pred))
        coef_df = pd.DataFrame({
            "feature": parent_cols,
            "coef": model.coef_,
        })

        adjustment_results[target_col] = {
            "model": model,
            "parents": parent_cols,
            "intercept": float(model.intercept_),
            "coef_df": coef_df,
            "r2": r2,
        }
        parent_rows.append({
            "target": target_col,
            "parents": parent_cols,
            "n_parents": len(parent_cols),
            "r2": r2,
        })

    parent_table = pd.DataFrame(parent_rows)

    return {
        "adjusted_residual_df": adjusted_residuals,
        "parent_map": parent_map,
        "parent_table": parent_table,
        "adjustment_results": adjustment_results,
    }


# ============================================================
# Forward layered regressions
# ============================================================

def run_forward_layered_regressions(
    df: pd.DataFrame,
    layers: dict[str, list[str]],
    start_layer: int = 3,
    end_layer: int = 9,
    normalize: bool = False,
    logistic_max_iter: int = 1000,
    n_boot: int = 100,
    edge_threshold: float = 0.70,
    min_causal_effect: float = 0.01,
    random_state: int | None = 0,
):
    """Run forward regressions using LiNGAM-adjusted L2 residuals.

    First, each L2 variable is adjusted by L1 variables. DirectLiNGAM is then
    applied to the L2 residuals. Each L2 residual is further adjusted by its
    selected within-L2 LiNGAM parents, and these LiNGAM-adjusted L2 residuals
    are used as predictors for L3 and later layers.
    """

    # --------------------------------------------------------
    # 1. Check layer definitions
    # --------------------------------------------------------
    if start_layer < 3:
        raise ValueError("start_layer must be at least 3 for this workflow.")

    if end_layer < start_layer:
        raise ValueError("end_layer must be greater than or equal to start_layer.")

    required_layer_names = [f"L{i}" for i in range(1, end_layer + 1)]

    missing_layers = [
        layer_name for layer_name in required_layer_names
        if layer_name not in layers
    ]

    if missing_layers:
        raise ValueError(f"Some required layers are missing from layers: {missing_layers}")

    if len(layers["L1"]) == 0:
        raise ValueError("L1 must not be empty.")

    if len(layers["L2"]) == 0:
        raise ValueError("L2 must not be empty.")

    # --------------------------------------------------------
    # 2. Check variable definitions
    # --------------------------------------------------------
    all_vars = []

    for layer_name in required_layer_names:
        all_vars.extend(layers[layer_name])

    duplicated_vars = [
        var for var in set(all_vars)
        if all_vars.count(var) > 1
    ]

    if duplicated_vars:
        raise ValueError(
            f"Some variables appear in multiple layers: {duplicated_vars}"
        )

    missing_cols = [col for col in all_vars if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Some columns are missing from df: {missing_cols}")

    # --------------------------------------------------------
    # 3. Use a common complete-case dataset
    # --------------------------------------------------------
    analysis_df = df[all_vars].copy().dropna()

    if analysis_df.empty:
        raise ValueError("No data are available after dropping missing values.")

    used_index = analysis_df.index
    working_df = analysis_df.copy()

    # --------------------------------------------------------
    # 4. Build the node table
    # --------------------------------------------------------
    node_rows = []

    for layer_name in required_layer_names:
        for col in layers[layer_name]:
            if layer_name == "L1":
                node_type = "raw_baseline_variable"
            elif layer_name == "L2":
                node_type = "lingam_adjusted_residual_variable"
            else:
                node_type = "observed_variable_residualized_for_forward_models"

            node_rows.append({
                "node": col,
                "layer": layer_name,
                "node_type": node_type,
            })

    nodes_df = pd.DataFrame(node_rows)

    # --------------------------------------------------------
    # 5. Create LiNGAM-adjusted L2 residual features
    # --------------------------------------------------------
    regression_results = {}
    residual_results = {}
    residual_feature_rows = []
    edge_rows = []

    baseline_feature_cols = list(layers["L1"])
    residual_feature_cols_by_layer: dict[str, list[str]] = {}
    residual_feature_info: dict[str, dict] = {}

    l2_residual_result = make_l2_residuals(
        df=analysis_df,
        L1=layers["L1"],
        L2=layers["L2"],
        normalize=normalize,
    )

    raw_l2_residual_df = l2_residual_result["residual_df"]

    l2_lingam_result = run_direct_lingam_with_bootstrap(
        residual_df=raw_l2_residual_df,
        n_boot=n_boot,
        edge_threshold=edge_threshold,
        min_causal_effect=min_causal_effect,
        random_state=random_state,
    )

    selected_l2_edges = l2_lingam_result["selected_edges"].copy()

    l2_adjustment_result = make_lingam_adjusted_l2_residuals(
        l2_residual_df=raw_l2_residual_df,
        selected_l2_edges=selected_l2_edges,
    )

    adjusted_l2_residual_df = l2_adjustment_result["adjusted_residual_df"]

    regression_results["L2"] = l2_residual_result["regression_results"]
    residual_results["L2"] = {}
    residual_feature_cols_by_layer["L2"] = []

    for target_col in layers["L2"]:
        residual_col = make_residual_col_name("L2", target_col)
        working_df.loc[adjusted_l2_residual_df.index, residual_col] = adjusted_l2_residual_df[target_col]

        residual_series = adjusted_l2_residual_df[target_col].copy()
        residual_series.name = f"{target_col}_lingam_adjusted_residual"
        residual_results["L2"][target_col] = residual_series

        parents = l2_adjustment_result["parent_map"].get(target_col, [])
        residual_feature_cols_by_layer["L2"].append(residual_col)

        residual_feature_info[residual_col] = {
            "residual_feature": residual_col,
            "original_node": target_col,
            "original_layer": "L2",
            "created_by_model_type": "linear_lingam_adjustment",
            "created_from_features": baseline_feature_cols,
            "lingam_adjusted": True,
            "lingam_parents": parents,
            "normalize": normalize,
        }

        residual_feature_rows.append({
            "residual_feature": residual_col,
            "original_node": target_col,
            "original_layer": "L2",
            "created_by_model_type": "linear_lingam_adjustment",
            "created_from_features": baseline_feature_cols,
            "lingam_adjusted": True,
            "lingam_parents": parents,
            "normalize": normalize,
        })

    # --------------------------------------------------------
    # 6. Create residual features for L3 and later layers
    # --------------------------------------------------------
    for n in range(3, end_layer + 1):
        target_layer = f"L{n}"
        target_cols = layers[target_layer]

        previous_residual_feature_cols = []

        for m in range(2, n):
            previous_layer = f"L{m}"
            previous_residual_feature_cols.extend(
                residual_feature_cols_by_layer.get(previous_layer, [])
            )

        feature_cols_for_this_layer = (
            baseline_feature_cols + previous_residual_feature_cols
        )

        if len(feature_cols_for_this_layer) == 0:
            raise ValueError(
                f"{target_layer}: no predictors are available. Please check L1."
            )

        regression_results[target_layer] = {}
        residual_results[target_layer] = {}
        residual_feature_cols_by_layer[target_layer] = []

        for target_col in target_cols:
            reg_result = run_auto_regression(
                df=working_df,
                target_col=target_col,
                feature_cols=feature_cols_for_this_layer,
                normalize=normalize,
                logistic_max_iter=logistic_max_iter,
            )

            regression_results[target_layer][target_col] = reg_result
            residual_results[target_layer][target_col] = reg_result["residuals"]

            residual_col = make_residual_col_name(target_layer, target_col)
            working_df.loc[reg_result["used_index"], residual_col] = reg_result[
                "residuals"
            ]

            residual_feature_cols_by_layer[target_layer].append(residual_col)

            residual_feature_info[residual_col] = {
                "residual_feature": residual_col,
                "original_node": target_col,
                "original_layer": target_layer,
                "created_by_model_type": reg_result["model_type"],
                "created_from_features": feature_cols_for_this_layer,
                "lingam_adjusted": False,
                "lingam_parents": [],
                "normalize": normalize,
            }

            residual_feature_rows.append({
                "residual_feature": residual_col,
                "original_node": target_col,
                "original_layer": target_layer,
                "created_by_model_type": reg_result["model_type"],
                "created_from_features": feature_cols_for_this_layer,
                "lingam_adjusted": False,
                "lingam_parents": [],
                "normalize": normalize,
            })

            if n >= start_layer:
                coef_df = reg_result["coef_df"].copy()

                for _, row in coef_df.iterrows():
                    feature_used = row["feature"]

                    if feature_used in baseline_feature_cols:
                        from_var = feature_used
                        from_layer = "L1"
                        feature_role = "raw_L1"
                        original_feature_layer = "L1"
                        lingam_adjusted_feature = False
                    else:
                        info = residual_feature_info[feature_used]
                        from_var = info["original_node"]
                        from_layer = info["original_layer"]
                        original_feature_layer = info["original_layer"]
                        lingam_adjusted_feature = bool(info.get("lingam_adjusted", False))

                        if lingam_adjusted_feature:
                            feature_role = "lingam_adjusted_l2_residual"
                        else:
                            feature_role = "residualized_previous_layer"

                    edge_data = {
                        "from": from_var,
                        "to": target_col,
                        "from_layer": from_layer,
                        "to_layer": target_layer,
                        "edge_type": f"{from_layer}_to_{target_layer}_residualized_regression",
                        "model_type": reg_result["model_type"],
                        "coef": float(row["coef"]),
                        "abs_coef": float(abs(row["coef"])),
                        "target": target_col,
                        "feature_used": feature_used,
                        "feature_role": feature_role,
                        "original_feature_layer": original_feature_layer,
                        "lingam_adjusted_feature": lingam_adjusted_feature,
                        "normalize": normalize,
                        "n_samples_used": len(reg_result["used_index"]),
                    }

                    if reg_result["model_type"] == "logistic":
                        edge_data["odds_ratio"] = float(row["odds_ratio"])
                        edge_data["accuracy"] = reg_result["metrics"].get(
                            "accuracy",
                            np.nan,
                        )
                        edge_data["auc"] = reg_result["metrics"].get(
                            "auc",
                            np.nan,
                        )
                        edge_data["r2"] = np.nan
                    else:
                        edge_data["odds_ratio"] = np.nan
                        edge_data["accuracy"] = np.nan
                        edge_data["auc"] = np.nan
                        edge_data["r2"] = reg_result["metrics"].get("r2", np.nan)

                    edge_rows.append(edge_data)

    all_edges_df = pd.DataFrame(edge_rows)

    if not all_edges_df.empty:
        all_edges_df = all_edges_df.reset_index(drop=True)

    residual_features_df = pd.DataFrame(residual_feature_rows)

    # --------------------------------------------------------
    # 7. Edge tables by target layer
    # --------------------------------------------------------
    edges_by_target_layer = {}

    for n in range(start_layer, end_layer + 1):
        target_layer = f"L{n}"

        if all_edges_df.empty:
            edges_by_target_layer[target_layer] = pd.DataFrame()
        else:
            edges_by_target_layer[target_layer] = all_edges_df[
                all_edges_df["to_layer"] == target_layer
            ].copy().reset_index(drop=True)

    # --------------------------------------------------------
    # 8. Residual data by layer
    # --------------------------------------------------------
    residual_df_by_layer = {}

    for layer_name, residual_cols in residual_feature_cols_by_layer.items():
        if residual_cols:
            tmp = working_df[residual_cols].copy()
            rename_map = {}

            for residual_col in residual_cols:
                original_node = residual_feature_info[residual_col]["original_node"]
                rename_map[residual_col] = original_node

            residual_df_by_layer[layer_name] = tmp.rename(columns=rename_map)
        else:
            residual_df_by_layer[layer_name] = pd.DataFrame(index=used_index)

    # --------------------------------------------------------
    # 9. Return results
    # --------------------------------------------------------
    result = {
        "layers": layers,
        "nodes": nodes_df,
        "analysis_df": analysis_df,
        "working_df": working_df,
        "used_index": used_index,
        "residual_features": residual_features_df,
        "edges": {
            "all_forward_regression_edges": all_edges_df,
            "by_target_layer": edges_by_target_layer,
        },
        "regressions": regression_results,
        "residuals": {
            "by_layer": residual_df_by_layer,
            "feature_cols_by_layer": residual_feature_cols_by_layer,
            "feature_info": residual_feature_info,
            "raw_l2_residual_df": raw_l2_residual_df,
            "lingam_adjusted_l2_residual_df": adjusted_l2_residual_df,
        },
        "l2_lingam_adjustment": {
            "lingam_result": l2_lingam_result,
            "selected_edges": selected_l2_edges,
            "parent_map": l2_adjustment_result["parent_map"],
            "parent_table": l2_adjustment_result["parent_table"],
            "adjustment_results": l2_adjustment_result["adjustment_results"],
        },
        "metadata": {
            "start_layer": start_layer,
            "end_layer": end_layer,
            "normalize": normalize,
            "logistic_max_iter": logistic_max_iter,
            "n_boot": n_boot,
            "edge_threshold": edge_threshold,
            "min_causal_effect": min_causal_effect,
            "random_state": random_state,
            "n_samples_used": len(used_index),
            "forward_design": (
                "L1 raw variables are always included; L2 variables are included "
                "through LiNGAM-adjusted residuals; L3 and later previous layers "
                "are included through residuals after adjustment by earlier layers."
            ),
        },
    }

    return result
