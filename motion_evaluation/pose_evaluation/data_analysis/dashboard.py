from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_analysis.config import OUTPUT_DIR

try:
    st.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    pass

if hasattr(st, "cache_data"):
    cache_outputs = st.cache_data(show_spinner=False)
else:
    cache_outputs = st.cache(allow_output_mutation=True, suppress_st_warning=True)


def safe_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]) or pd.api.types.is_bool_dtype(table[column]):
            continue
        table[column] = table[column].fillna("").astype(str)
    return table


def render_html_table(frame: pd.DataFrame) -> None:
    table = safe_table(frame)
    html = table.to_html(index=False, escape=True, border=0, classes="analysis-table")
    st.markdown(
        """
        <style>
        .analysis-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.88rem;
        }
        .analysis-table th, .analysis-table td {
          border: 1px solid #dbe2ea;
          padding: 0.3rem 0.45rem;
          text-align: left;
          vertical-align: top;
        }
        .analysis-table th {
          background: #f7fafc;
          font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(html, unsafe_allow_html=True)


@cache_outputs
def load_outputs() -> dict[str, pd.DataFrame | str]:
    result: dict[str, pd.DataFrame] = {}
    for csv_name in (
        "metrics_long.csv",
        "summary_by_trial.csv",
        "summary_by_module.csv",
        "summary_by_domain.csv",
        "normalization_summary.csv",
        "warnings.csv",
        "joint_availability.csv",
        "plot_manifest.csv",
        "bland_altman.csv",
    ):
        path = OUTPUT_DIR / csv_name
        result[csv_name] = pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()
    return result


def filter_plot_rows(plot_df: pd.DataFrame, pair_key: str, wrist_variant: str, alignment_mode: str) -> pd.DataFrame:
    if plot_df.empty:
        return plot_df
    mask = (
        plot_df["plot_path"].astype(str).str.contains(pair_key, regex=False)
        & plot_df["plot_path"].astype(str).str.contains(f"__{wrist_variant}__", regex=False)
        & plot_df["plot_path"].astype(str).str.contains(f"__{alignment_mode}__", regex=False)
    )
    return plot_df.loc[mask].copy()


st.set_page_config(page_title="Motion Pattern Evaluation Dashboard", layout="wide")
st.title("Motion Pattern Evaluation Dashboard")

if not OUTPUT_DIR.exists():
    st.error(f"No evaluation outputs found under {OUTPUT_DIR}. Run `python data_analysis/run_analysis.py` first.")
    st.stop()

outputs = load_outputs()
metrics_df = outputs["metrics_long.csv"]
trial_summary_df = outputs["summary_by_trial.csv"]
module_summary_df = outputs["summary_by_module.csv"]
domain_summary_df = outputs["summary_by_domain.csv"]
normalization_df = outputs["normalization_summary.csv"]
warnings_df = outputs["warnings.csv"]
availability_df = outputs["joint_availability.csv"]
plot_df = outputs["plot_manifest.csv"]
bland_altman_df = outputs["bland_altman.csv"]

if isinstance(trial_summary_df, pd.DataFrame) and trial_summary_df.empty:
    st.error("The evaluation output directory exists, but the summary files are empty. Rerun `python data_analysis/run_analysis.py`.")
    st.stop()

st.caption(
    "Read-only inspection of the body-centered 2D evaluation outputs. "
    "The metrics compare motion-pattern fidelity after temporal alignment rather than synchronized ground-truth tracking error."
)

comparison_options = sorted(trial_summary_df["comparison_name"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    selected_comparison = st.selectbox("Comparison", comparison_options)
    comparison_trial_slice = trial_summary_df[trial_summary_df["comparison_name"] == selected_comparison]
    selected_trial = st.selectbox("Trial", sorted(comparison_trial_slice["trial_id"].dropna().unique().tolist()))
    trial_wrist_slice = comparison_trial_slice[comparison_trial_slice["trial_id"] == selected_trial]
    selected_wrist_variant = st.selectbox("Wrist variant", sorted(trial_wrist_slice["wrist_variant"].dropna().unique().tolist()))
    trial_alignment_slice = trial_wrist_slice[trial_wrist_slice["wrist_variant"] == selected_wrist_variant]
    selected_alignment_mode = st.selectbox("Alignment mode", sorted(trial_alignment_slice["alignment_mode"].dropna().unique().tolist()))

pair_slice = trial_summary_df[
    (trial_summary_df["comparison_name"] == selected_comparison)
    & (trial_summary_df["trial_id"] == selected_trial)
    & (trial_summary_df["wrist_variant"] == selected_wrist_variant)
]
pair_keys = sorted(pair_slice["pair_key"].dropna().unique().tolist())
selected_pair_key = pair_keys[0]

summary_selection = pair_slice[pair_slice["alignment_mode"] == selected_alignment_mode].copy()
metrics_selection = metrics_df[
    (metrics_df["comparison_name"] == selected_comparison)
    & (metrics_df["pair_key"] == selected_pair_key)
    & (metrics_df["wrist_variant"] == selected_wrist_variant)
    & (metrics_df["alignment_mode"] == selected_alignment_mode)
].copy()
normalization_selection = normalization_df[
    (normalization_df["comparison_name"] == selected_comparison)
    & (normalization_df["pair_key"] == selected_pair_key)
    & (normalization_df["wrist_variant"] == selected_wrist_variant)
]
if "alignment_mode" in normalization_selection.columns:
    normalization_selection = normalization_selection[normalization_selection["alignment_mode"].isin(["body_only", selected_alignment_mode])]
warning_selection = warnings_df[
    (warnings_df["comparison_name"] == selected_comparison)
    & (warnings_df["pair_key"] == selected_pair_key)
    & (warnings_df["wrist_variant"] == selected_wrist_variant)
    & (warnings_df["alignment_mode"] == selected_alignment_mode)
]
availability_selection = availability_df[
    (availability_df["comparison_name"] == selected_comparison)
    & (availability_df["pair_key"] == selected_pair_key)
    & (availability_df["wrist_variant"] == selected_wrist_variant)
]
plot_selection = filter_plot_rows(plot_df, selected_pair_key, selected_wrist_variant, selected_alignment_mode)

st.subheader("Trial Summary")
render_html_table(summary_selection.sort_values(["metric_family"]))

st.subheader("Normalization Summary")
render_html_table(
    normalization_selection[
        [
            "system_name",
            "system_label",
            "alignment_mode",
            "origin_source",
            "scale_source",
            "scale_value",
            "rotation_source",
            "rotation_angle_deg",
            "orientation_x_sign",
            "orientation_y_sign",
            "orientation_source",
            "similarity_scale",
            "similarity_rotation_deg",
            "similarity_translation_x",
            "similarity_translation_y",
        ]
    ]
)

st.subheader("Metric Details")
metric_family_options = ["all"] + sorted(metrics_selection["metric_family"].dropna().unique().tolist())
selected_metric_family = st.selectbox("Metric family", metric_family_options)
joint_options = ["all"] + sorted(metrics_selection["joint"].dropna().unique().tolist())
selected_joint = st.selectbox("Joint / signal", joint_options)
side_options = ["all"] + sorted(metrics_selection["side"].dropna().unique().tolist())
selected_side = st.selectbox("Side", side_options)

detailed_metrics = metrics_selection.copy()
if selected_metric_family != "all":
    detailed_metrics = detailed_metrics[detailed_metrics["metric_family"] == selected_metric_family]
if selected_joint != "all":
    detailed_metrics = detailed_metrics[detailed_metrics["joint"] == selected_joint]
if selected_side != "all":
    detailed_metrics = detailed_metrics[detailed_metrics["side"] == selected_side]

render_html_table(detailed_metrics.sort_values(["metric_family", "joint", "side", "metric_name"]))

st.subheader("Aggregate Module Summary")
module_selection = module_summary_df[
    (module_summary_df["comparison_name"] == selected_comparison)
    & (module_summary_df["wrist_variant"] == selected_wrist_variant)
    & (module_summary_df["alignment_mode"] == selected_alignment_mode)
]
render_html_table(module_selection.sort_values(["metric_family", "metric_name"]))

st.subheader("Hand/Arm and Knee/Pedal Summary")
if domain_summary_df.empty:
    st.write("No domain summary available for the selected slice.")
else:
    domain_selection = domain_summary_df[
        (domain_summary_df["comparison_name"] == selected_comparison)
        & (domain_summary_df["pair_key"] == selected_pair_key)
        & (domain_summary_df["wrist_variant"] == selected_wrist_variant)
        & (domain_summary_df["alignment_mode"] == selected_alignment_mode)
    ]
    render_html_table(domain_selection.sort_values(["metric_domain", "side", "metric_name"]))

st.subheader("Bland-Altman Agreement")
if bland_altman_df.empty:
    st.write("No Bland-Altman table available for the selected slice.")
else:
    bland_altman_selection = bland_altman_df[
        (bland_altman_df["comparison_name"] == selected_comparison)
        & (bland_altman_df["pair_key"] == selected_pair_key)
        & (bland_altman_df["wrist_variant"] == selected_wrist_variant)
        & (bland_altman_df["alignment_mode"] == selected_alignment_mode)
    ]
    if bland_altman_selection.empty:
        st.write("No Bland-Altman rows available for the selected slice.")
    else:
        render_html_table(bland_altman_selection.sort_values(["variable_family", "side", "variable_name"]))

st.subheader("Joint Availability")
render_html_table(availability_selection.sort_values(["system_name", "joint_id"]))

st.subheader("Warnings")
if warning_selection.empty:
    st.write("No warnings for the selected slice.")
else:
    render_html_table(warning_selection)

st.subheader("Generated Plots")
if plot_selection.empty:
    st.write("No plots available for the selected slice.")
else:
    for _, row in plot_selection.iterrows():
        plot_path = Path(row["plot_path"])
        st.markdown(f"**{row['plot_type']}**")
        if plot_path.exists():
            st.image(str(plot_path))
        else:
            st.write(f"Missing plot: `{plot_path}`")
