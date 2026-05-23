from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.charts import (
    GROUP_TREND_HIGHLIGHT,
    bump_chart,
    distribution_chart,
    entity_color_map,
    grouped_bar_chart,
    heatmap_chart,
    lollipop_chart,
    line_metric_chart,
    normalize_group_legend_label,
    order_group_legend_values,
    scatter_chart,
    stacked_bar_chart,
)
from src.filters import (
    apply_common_filters,
    build_breadcrumb_text,
    build_region_comparison_selection,
    build_scope_selection,
    get_selected_adjacent_gusigun_names,
    render_sidebar_filters,
    resolve_region_level,
)
from src.election_scope import ElectionScope, build_local_vote_scope_options, filter_by_election_scope, preferred_scope_key
from src.loaders import get_cache_file_signatures, load_fact_turnout_enriched, load_fact_votes_enriched, load_global_filter_options
from src.maps import BASEMAP_STYLE_OPTIONS, build_diverging_colorscale, build_region_choropleth, build_single_hue_colorscale, get_adjacent_gusigun_keys, load_geometry_context
from src.metrics import (
    build_competitiveness_sentence,
    build_vote_rowtype_sentence,
    calc_map_metric_by_region,
    calc_distribution_by_region,
    calc_entity_share,
    calc_entity_share_by_region,
    calc_entity_top2_gap,
    calc_entity_trend,
    calc_polling_station_metrics,
    calc_rowtype_entity_breakdown,
    calc_top2_gap_by_region,
    calc_turnout_vote_scatter,
    calc_votes_summary,
    competitiveness_help_text,
    dataframe_to_csv_bytes,
    format_int,
    format_percent,
)

VOTES_PAGE_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "정당KEY",
    "선거구명",
    "시도명",
    "구시군명",
    "구시군KEY",
    "일반구명",
    "읍면동명",
    "읍면동KEY",
    "구분",
    "정당구분",
    "구분2",
    "성향",
    "RowType",
    "투표소KEY",
    "정당명",
    "후보명",
    "후보라벨",
    "후보슬롯",
    "유효투표수",
    "득표수",
)
ENTITY_TYPE_OPTIONS = ["후보", "정당", "구분", "구분2", "성향"]
ENTITY_TYPE_DISPLAY_LABELS = {
    "후보": "후보",
    "정당": "정당",
    "구분": "구분1",
    "구분2": "구분2",
    "성향": "성향",
}
ENTITY_LABEL_COLUMNS = {
    "후보": "후보명",
    "정당": "정당명",
    "구분": "정당구분",
    "구분2": "구분2",
    "성향": "성향",
}
TREND_ENTITY_TYPE_OPTIONS = ["구분", "구분2", "성향"]
TREND_GROUP_ENTITY_TYPES = {"구분", "구분2"}
TREND_GROUP_DISPLAY_LABELS = {"민주당": "민주당계", "국민의힘": "국민의힘계"}
TREND_PARTY_HOVER_COL = "포함 정당"
TREND_PARTY_SUMMARY_ORDER = ["민주당계", "국민의힘계", "제3지대"]
TREND_PIVOT_DISPLAY_RENAME = {"제3지대": "제3지대계"}
TREND_PIVOT_VALUE_COLUMNS = ["민주당계", "국민의힘계", "제3지대계", "무소속"]
TREND_PIVOT_DEFAULT_COLUMNS = ["선거축라벨", "1위 정당/후보", *TREND_PIVOT_VALUE_COLUMNS, "비고"]
TREND_PIVOT_NOTE_LABELS = {"민주당계": "민주당계", "국민의힘계": "국힘계", "제3지대": "제3지대계", "제3지대계": "제3지대계"}
TREND_TOP_PARTY_ELECTION_KEYWORDS = ("광역의회", "기초의회", "광역의원", "기초의원")
VOTE_GIS_METRIC_OPTIONS = {
    "후보별 득표수": ("후보 득표수", "후보"),
    "후보별 득표율": ("후보 득표율", "후보"),
    "정당별 득표수": ("정당 득표수", "정당"),
    "정당별 득표율": ("정당 득표율", "정당"),
    "후보 간 득표수 격차": ("후보 간 득표수 격차", "후보"),
    "후보 간 득표율 격차": ("후보 간 득표율 격차", "후보"),
    "정당 간 득표수 격차": ("정당 간 득표수 격차", "정당"),
    "정당 간 득표율 격차": ("정당 간 득표율 격차", "정당"),
}
VOTE_GIS_PAIR_GAP_METRICS = {
    "후보 간 득표수 격차": "후보",
    "후보 간 득표율 격차": "후보",
    "정당 간 득표수 격차": "정당",
    "정당 간 득표율 격차": "정당",
}
SWING_DEM_LABEL = "민주당계"
SWING_CONSERVATIVE_LABEL = "국민의힘계"
SWING_CATEGORY_ORDER = ["국민의힘계 우세", "국민의힘계 경합 우세", "경합지역", "민주당계 경합 우세", "민주당계 우세"]
SWING_CATEGORY_COLORS = {
    "국민의힘계 우세": "#D94C3D",
    "국민의힘계 경합 우세": "#F2A6B3",
    "경합지역": "#F8FAFC",
    "민주당계 경합 우세": "#9EC5E8",
    "민주당계 우세": "#1F5AA6",
}
SWING_CATEGORY_SCORES = {label: index + 1 for index, label in enumerate(SWING_CATEGORY_ORDER)}
SWING_COLOR_SCALE = [
    [0.0, SWING_CATEGORY_COLORS["국민의힘계 우세"]],
    [0.1249, SWING_CATEGORY_COLORS["국민의힘계 우세"]],
    [0.125, SWING_CATEGORY_COLORS["국민의힘계 경합 우세"]],
    [0.3749, SWING_CATEGORY_COLORS["국민의힘계 경합 우세"]],
    [0.375, SWING_CATEGORY_COLORS["경합지역"]],
    [0.6249, SWING_CATEGORY_COLORS["경합지역"]],
    [0.625, SWING_CATEGORY_COLORS["민주당계 경합 우세"]],
    [0.8749, SWING_CATEGORY_COLORS["민주당계 경합 우세"]],
    [0.875, SWING_CATEGORY_COLORS["민주당계 우세"]],
    [1.0, SWING_CATEGORY_COLORS["민주당계 우세"]],
]
SWING_OTHER_LABEL = "무소속/기타"
SWING_PARTY_ORDER = [SWING_DEM_LABEL, SWING_CONSERVATIVE_LABEL, "제3지대", SWING_OTHER_LABEL]
SWING_PARTY_COLORS = {
    SWING_DEM_LABEL: "#1F5AA6",
    SWING_CONSERVATIVE_LABEL: "#D94C3D",
    "제3지대": "#F28C28",
    SWING_OTHER_LABEL: "#7B8B97",
}
SWING_LOCAL_EARLY_VOTE_KEY = "__local_early_vote_subtotal__"
SWING_LOCAL_EARLY_VOTE_LABEL = "관내사전투표 소계"
SWING_ELECTION_DAY_VOTE_KEY = "__election_day_vote_subtotal__"
SWING_ELECTION_DAY_VOTE_LABEL = "선거일투표 소계"
SWING_OUTSIDE_VOTE_KEY = "__outside_vote_subtotal__"
SWING_OUTSIDE_VOTE_LABEL = "관외투표 소계"
SWING_OUTSIDE_EXCLUDED_ROWTYPES = {"읍면동", "관내사전투표", "투표소", "선거일투표", "합계"}
SWING_SPECIAL_VOTE_ROW_SPECS = (
    {"key": SWING_LOCAL_EARLY_VOTE_KEY, "label": SWING_LOCAL_EARLY_VOTE_LABEL, "rowtypes": ("관내사전투표",)},
    {"key": SWING_ELECTION_DAY_VOTE_KEY, "label": SWING_ELECTION_DAY_VOTE_LABEL, "rowtypes": ("선거일투표", "투표소")},
    {"key": SWING_OUTSIDE_VOTE_KEY, "label": SWING_OUTSIDE_VOTE_LABEL, "excluded_rowtypes": SWING_OUTSIDE_EXCLUDED_ROWTYPES},
)
SWING_OUTSIDE_VOTE_ROW_SPECS = (SWING_SPECIAL_VOTE_ROW_SPECS[2],)
SWING_ROWTYPE_PIVOT_SPECS = (SWING_SPECIAL_VOTE_ROW_SPECS[0], SWING_SPECIAL_VOTE_ROW_SPECS[1])
SWING_SPECIAL_VOTE_ROW_KEYS = {str(spec["key"]) for spec in SWING_SPECIAL_VOTE_ROW_SPECS}
SWING_SPECIAL_VOTE_ROW_LABELS = {str(spec["key"]): str(spec["label"]) for spec in SWING_SPECIAL_VOTE_ROW_SPECS}
SWING_SPECIAL_VOTE_ROW_ORDER = {str(spec["key"]): index for index, spec in enumerate(SWING_SPECIAL_VOTE_ROW_SPECS)}
ADMIN_GUSIGUN_RENAME_ALIASES = [
    {"시도명": "경기도", "from": "여주군", "to": "여주시"},
]
ADMIN_EMD_RENAME_ALIASES = [
    {"시도명": "경기도", "구시군명": "여주시", "from": "능서면", "to": "세종대왕면"},
    {"시도명": "부산광역시", "구시군명": "금정구", "from": "금사동", "to": "금사회동동"},
    {"시도명": "경기도", "구시군명": "의정부시", "from": "가능1동", "to": "가능동"},
    {"시도명": "전라남도", "구시군명": "화순군", "from": "북면", "to": "백아면"},
    {"시도명": "전라남도", "구시군명": "화순군", "from": "남면", "to": "사평면"},
    {"시도명": "경기도", "구시군명": "김포시", "from": "김포1동", "to": "김포본동"},
    {"시도명": "경기도", "구시군명": "김포시", "from": "김포2동", "to": "장기본동"},
    {"시도명": "충청북도", "구시군명": "제천시", "from": "의암동", "to": "의림지동"},
    {"시도명": "충청북도", "구시군명": "제천시", "from": "인성동", "to": "중앙동"},
]

TURNOUT_SCATTER_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "시도명",
    "구시군명",
    "구시군KEY",
    "일반구명",
    "읍면동명",
    "읍면동KEY",
    "RowType",
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
)


def _load_bundle() -> dict[str, object]:
    cache_signature = get_cache_file_signatures()
    return {
        "votes": load_fact_votes_enriched(cache_signature, VOTES_PAGE_COLUMNS),
        "turnout": load_fact_turnout_enriched(cache_signature, TURNOUT_SCATTER_COLUMNS),
        "filter_options": load_global_filter_options(cache_signature),
    }


def _extract_selection_value(state: Any, keys: tuple[str, ...] = ("y", "x")) -> str | None:
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    point = points[0]
    for key in keys:
        value = point.get(key) if isinstance(point, dict) else None
        if value not in (None, ""):
            return str(value)
    return None


def _label_column(entity_type: str) -> str:
    return ENTITY_LABEL_COLUMNS[entity_type]


def _display_entity_type(entity_type: str) -> str:
    return ENTITY_TYPE_DISPLAY_LABELS.get(entity_type, entity_type)


def _first_text_value(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns:
        return None
    values = df[column].dropna().astype("string")
    return None if values.empty else str(values.iloc[0])


def _filter_selected_entities(df: pd.DataFrame, label_col: str, entities: list[str]) -> pd.DataFrame:
    if df.empty or label_col not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df[label_col].astype("string").isin([str(entity) for entity in entities])].copy()


def _build_context_share_frame(df: pd.DataFrame, entity_type: str, label_col: str, entities: list[str], context_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["비교축", label_col, "득표수", "유효투표수", "득표율_계산", "정당KEY", "정당명", "후보라벨", "후보슬롯"])
    share = calc_entity_share(df, entity_type=entity_type)
    share = _filter_selected_entities(share, label_col, entities)
    if share.empty:
        return pd.DataFrame(columns=["비교축", label_col, "득표수", "유효투표수", "득표율_계산", "정당KEY", "정당명", "후보라벨", "후보슬롯"])
    share["비교축"] = context_label
    output_columns = ["비교축", label_col, "득표수", "유효투표수", "득표율_계산"]
    for extra_col in ["정당KEY", "정당명", "후보라벨", "후보슬롯"]:
        if extra_col in share.columns and extra_col not in output_columns:
            output_columns.append(extra_col)
    return share.loc[:, output_columns]


def _entity_share_value(df: pd.DataFrame, label_col: str, entity_name: str | None) -> float | None:
    if not entity_name or df.empty or label_col not in df.columns or "득표율_계산" not in df.columns:
        return None
    matched = df.loc[df[label_col].astype("string") == str(entity_name), "득표율_계산"].dropna()
    return None if matched.empty else float(matched.iloc[0])


def _format_pp_visual_comparison(target_value: float | None, baseline_value: float | None) -> tuple[str, str] | None:
    if target_value is None or baseline_value is None:
        return None

    diff_pp = (target_value - baseline_value) * 100
    if abs(diff_pp) < 0.005:
        return ("<span style='color:#64748b; font-weight:700;'>↑0.00%p</span>", "")

    direction = "↑" if diff_pp > 0 else "↓"
    color = "#15803d" if diff_pp > 0 else "#dc2626"
    return (f"<span style='color:{color}; font-weight:700;'>{direction}{abs(diff_pp):.2f}%p</span>", "")


def _focus_axis_category_order(axis_labels: dict[str, str]) -> list[str]:
    return [
        axis_labels["target_axis"],
        axis_labels["national_axis"],
        axis_labels["sido_axis"],
        axis_labels["adjacent_axis"],
    ]


def _build_focus_axis_labels(focus_target_name: str | None, focus_sido_name: str | None) -> dict[str, str]:
    target_base_label = str(focus_target_name) if focus_target_name else "선택한 구시군"
    sido_base_label = f"{focus_sido_name} 평균" if focus_sido_name else "광역 시도 평균"
    return {
        "target_axis": f"{target_base_label} 득표율",
        "national_axis": "전국 득표율",
        "sido_axis": f"{sido_base_label} 득표율",
        "adjacent_axis": "인접 구시군권 득표율",
        "target_base": target_base_label,
        "national_base": "전국",
        "sido_base": sido_base_label,
        "adjacent_base": "인접 구시군권",
    }


def _normalize_entity_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _build_highlight_entities(entity_type: str, top_gap: pd.Series | None, primary_entity: str | None) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    if entity_type == "후보" and top_gap is not None:
        first = _normalize_entity_name(top_gap.get("1위"))
        second = _normalize_entity_name(top_gap.get("2위"))
        if first:
            entities.append(("1위 후보", first))
        if second and second != first:
            entities.append(("2위 후보", second))
    elif primary_entity:
        entities.append((f"{_display_entity_type(entity_type)} 대표", str(primary_entity)))
    return entities


def _resolve_adjacent_compare_entity(current_gusigun_votes: pd.DataFrame, entity_type: str) -> str | None:
    label_col = _label_column(entity_type)
    entity_share = calc_entity_share(current_gusigun_votes, entity_type=entity_type)
    if entity_share.empty or label_col not in entity_share.columns:
        return None

    options = entity_share[label_col].dropna().astype("string").tolist()
    if not options:
        return None

    state_key = f"votes_adjacent_compare_entity_{entity_type}"
    default_value = str(options[0])
    current_value = st.session_state.get(state_key)
    if current_value not in options:
        st.session_state[state_key] = default_value

    selected_value = st.selectbox(
        "인접 구시군 비교 대상",
        options=options,
        key=state_key,
        help=f"선택 구시군의 {_display_entity_type(entity_type)} 가운데 원하는 대상을 골라 인접 구시군과 비교합니다.",
    )
    return str(selected_value)


def _entity_legend_order(df: pd.DataFrame, entity_type: str, label_col: str, selected_entities: list[str]) -> list[str]:
    selected_values = [str(entity) for entity in selected_entities if entity not in (None, "")]
    if not selected_values or label_col not in df.columns:
        return selected_values

    order_frame = df.loc[df[label_col].notna(), [column for column in [label_col, "후보슬롯", "정당KEY"] if column in df.columns]].copy()
    if order_frame.empty:
        return selected_values

    order_frame[label_col] = order_frame[label_col].astype("string")
    if entity_type in {"구분", "구분2", "성향"}:
        ordered_values = order_group_legend_values(order_frame[label_col], entity_type=entity_type)
    else:
        order_frame["__slot_order__"] = pd.to_numeric(order_frame["후보슬롯"], errors="coerce") if "후보슬롯" in order_frame.columns else pd.NA
        if "정당KEY" in order_frame.columns:
            order_frame["__party_order__"] = pd.to_numeric(
                order_frame["정당KEY"].astype("string").str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )
        else:
            order_frame["__party_order__"] = pd.NA

        order_map = (
            order_frame.groupby(label_col, observed=True)
            .agg(__slot_order__=("__slot_order__", "min"), __party_order__=("__party_order__", "min"))
            .reset_index()
            .sort_values(by=["__slot_order__", "__party_order__", label_col], ascending=[True, True, True], kind="stable", na_position="last")
        )
        ordered_values = order_map[label_col].astype("string").tolist()
    ordered_selected = [value for value in ordered_values if value in selected_values]
    remaining = [value for value in selected_values if value not in ordered_selected]
    return [*ordered_selected, *remaining]


def _render_focus_metric_row(
    context_df: pd.DataFrame,
    label_col: str,
    entity_name: str,
    title: str,
    axis_labels: dict[str, str],
) -> None:
    national_axis = axis_labels["national_axis"]
    sido_axis = axis_labels["sido_axis"]
    target_axis = axis_labels["target_axis"]
    adjacent_axis = axis_labels["adjacent_axis"]

    national_value = _entity_share_value(context_df.loc[context_df["비교축"].astype("string") == national_axis], label_col, entity_name)
    sido_value = _entity_share_value(context_df.loc[context_df["비교축"].astype("string") == sido_axis], label_col, entity_name)
    target_value = _entity_share_value(context_df.loc[context_df["비교축"].astype("string") == target_axis], label_col, entity_name)
    adjacent_value = _entity_share_value(
        context_df.loc[context_df["비교축"].astype("string") == adjacent_axis],
        label_col,
        entity_name,
    )
    st.caption(title)
    metric1, metric2, metric3, metric4 = st.columns(4)
    comparison_layout = [
        (metric1, target_axis, target_value, None),
        (metric2, national_axis, national_value, "전국대비"),
        (metric3, sido_axis, sido_value, "광역시도대비"),
        (metric4, adjacent_axis, adjacent_value, "인접구시군권대비"),
    ]
    for column, label, value, compare_label in comparison_layout:
        column.metric(label, "-" if value is None else format_percent(value))
        if compare_label is not None:
            delta_text = _format_pp_visual_comparison(target_value, value)
            if delta_text:
                delta_html, _ = delta_text
                column.markdown(f'{axis_labels["target_base"]} {delta_html}', unsafe_allow_html=True)


def _scatter_point_description(
    region_level: str,
    scatter_entity_name: str,
    gusigun_focus_mode: bool,
    focus_target_name: str | None,
    selected_adjacent_names: list[str],
) -> str:
    if region_level == "시도":
        return f"각 점은 하나의 시도를 뜻하며, x축은 투표율, y축은 {scatter_entity_name} 득표율입니다."
    if region_level == "읍면동":
        return "각 점은 하나의 읍면동을 뜻하며, 라벨은 읍면동명만 표시됩니다."
    if gusigun_focus_mode and focus_target_name:
        if selected_adjacent_names:
            adjacent_text = ", ".join(str(name) for name in selected_adjacent_names if name)
            return (
                f"각 점은 하나의 구시군을 뜻하며, 라벨은 선택 구시군 `{focus_target_name}`과 "
                f"선택한 인접 구시군 `{adjacent_text}`의 구시군명만 표시됩니다."
            )
        return (
            f"각 점은 하나의 구시군을 뜻하며, 라벨은 선택 구시군 `{focus_target_name}`의 구시군명만 표시됩니다. "
            "인접 구시군을 고르면 해당 구시군에도 라벨이 표시됩니다."
        )
    return f"각 점은 하나의 구시군을 뜻하며, x축은 투표율, y축은 {scatter_entity_name} 득표율입니다."


def _build_scatter_point_text(region_name: str) -> str:
    return region_name


def _build_scatter_point_plain_text(region_name: str) -> str:
    return region_name


def _annotation_rect(center_x: float, center_y: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def _rect_overlap_area(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    return float((right - left) * (bottom - top))


def _add_scatter_label_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    label_col: str,
    x_col: str = "투표율",
    y_col: str = "득표율_계산",
) -> go.Figure:
    if df.empty or label_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return fig

    label_df = df.loc[df[label_col].notna(), [x_col, y_col, label_col]].copy()
    if label_df.empty:
        return fig

    label_df[label_col] = label_df[label_col].astype("string")
    label_df = (
        label_df.sort_values(by=[y_col, x_col], ascending=[False, False], kind="stable")
        .drop_duplicates(subset=[label_col], keep="first")
        .reset_index(drop=True)
    )

    x_min = float(label_df[x_col].min())
    x_max = float(label_df[x_col].max())
    y_min = float(label_df[y_col].min())
    y_max = float(label_df[y_col].max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    plot_width = 1040.0
    plot_height = 520.0

    entries: list[dict[str, object]] = []
    for _, row in label_df.iterrows():
        point_px_x = ((float(row[x_col]) - x_min) / x_span) * plot_width
        point_px_y = (1 - ((float(row[y_col]) - y_min) / y_span)) * plot_height
        plain_text = _build_scatter_point_plain_text(str(row[label_col]))
        lines = plain_text.splitlines()
        max_line_length = max((len(line) for line in lines), default=0)
        entries.append(
            {
                "x": float(row[x_col]),
                "y": float(row[y_col]),
                "label": str(row[label_col]),
                "point_px_x": point_px_x,
                "point_px_y": point_px_y,
                "label_width": max(86.0, max_line_length * 8.3),
                "label_height": max(38.0, len(lines) * 20.0 + 8.0),
            }
        )

    point_rects = [
        _annotation_rect(float(entry["point_px_x"]), float(entry["point_px_y"]), 28.0, 28.0)
        for entry in entries
    ]
    candidate_offsets = [
        (0, -26),
        (18, -26),
        (-18, -26),
        (0, -42),
        (28, -38),
        (-28, -38),
        (0, 22),
        (18, 22),
        (-18, 22),
        (36, -16),
        (-36, -16),
        (44, -30),
        (-44, -30),
        (0, -58),
        (56, -10),
        (-56, -10),
        (34, 30),
        (-34, 30),
        (68, -34),
        (-68, -34),
    ]
    placed_label_rects: list[tuple[float, float, float, float]] = []

    for entry_index, entry in enumerate(entries):
        label_width = float(entry["label_width"])
        label_height = float(entry["label_height"])
        point_px_x = float(entry["point_px_x"])
        point_px_y = float(entry["point_px_y"])
        best_offset = candidate_offsets[0]
        best_candidate_index = 0
        best_score: float | None = None

        for candidate_index, (ax, ay) in enumerate(candidate_offsets):
            center_x = point_px_x + ax
            center_y = point_px_y + ay
            rect = _annotation_rect(center_x, center_y, label_width, label_height)

            overlap_score = 0.0
            for placed_rect in placed_label_rects:
                overlap_score += _rect_overlap_area(rect, placed_rect) * 100.0

            point_penalty = 0.0
            for other_index, point_rect in enumerate(point_rects):
                if other_index == entry_index:
                    continue
                point_penalty += _rect_overlap_area(rect, point_rect) * 400.0

            boundary_penalty = 0.0
            if rect[0] < 0:
                boundary_penalty += abs(rect[0]) * 16.0
            if rect[2] > plot_width:
                boundary_penalty += abs(rect[2] - plot_width) * 16.0
            if rect[1] < 0:
                boundary_penalty += abs(rect[1]) * 16.0
            if rect[3] > plot_height:
                boundary_penalty += abs(rect[3] - plot_height) * 16.0

            distance_penalty = (abs(ax) + abs(ay)) * 1.2 + candidate_index * 0.8
            score = overlap_score + point_penalty + boundary_penalty + distance_penalty

            if best_score is None or score < best_score:
                best_score = score
                best_offset = (ax, ay)
                best_candidate_index = candidate_index
                if overlap_score == 0 and point_penalty == 0 and boundary_penalty == 0 and candidate_index <= 2:
                    break

        ax, ay = best_offset
        placed_label_rects.append(_annotation_rect(point_px_x + ax, point_px_y + ay, label_width, label_height))
        text_align = "center" if ax == 0 else ("left" if ax > 0 else "right")
        use_arrow = best_candidate_index >= 3 or abs(ax) >= 28 or abs(ay) >= 34
        fig.add_annotation(
            x=float(entry["x"]),
            y=float(entry["y"]),
            xref="x",
            yref="y",
            text=_build_scatter_point_text(str(entry["label"])),
            showarrow=use_arrow,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="rgba(100,116,139,0.7)",
            ax=ax if use_arrow else 0,
            ay=ay if use_arrow else 0,
            xshift=0 if use_arrow else ax,
            yshift=0 if use_arrow else ay,
            align=text_align,
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(255,255,255,0)",
            font=dict(size=17, color="#334155"),
        )
    return fig


def _resolve_current_election_key(filtered_votes: pd.DataFrame, filter_options: dict[str, object]) -> str:
    election_candidates = (
        filtered_votes.loc[:, ["선거KEY", "선거시점"]]
        .drop_duplicates()
        .sort_values(by=["선거시점", "선거KEY"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    if election_candidates.empty:
        raise ValueError("현재 필터 조건에 맞는 선거를 찾을 수 없습니다.")

    option_keys = election_candidates["선거KEY"].astype("string").tolist()
    if len(option_keys) == 1:
        return option_keys[0]

    key_to_label = filter_options.get("election_key_to_label", {})
    current_value = str(st.session_state.get("votes_current_election_key", option_keys[0]))
    if current_value not in option_keys:
        current_value = option_keys[0]
        st.session_state["votes_current_election_key"] = current_value

    selected_key = st.selectbox(
        "현재 선거",
        options=option_keys,
        index=option_keys.index(current_value),
        key="votes_current_election_key",
        format_func=lambda value: str(key_to_label.get(str(value), value)),
    )
    st.caption("여러 선거가 필터에 포함되어 있어 현재 분석 선거를 직접 선택하도록 표시했습니다.")
    return str(selected_key)


def _resolve_local_vote_scope(current_votes: pd.DataFrame, selected: dict[str, list[str]]) -> ElectionScope | None:
    scope_options = build_local_vote_scope_options(current_votes)
    if len(scope_options) <= 1:
        return scope_options[0] if scope_options else None

    scope_by_key = {scope.key: scope for scope in scope_options}
    option_keys = list(scope_by_key)
    suggested_key = preferred_scope_key(scope_options, selected) or option_keys[0]
    has_region_filter = bool(selected.get("시도명") or selected.get("구시군명") or selected.get("읍면동명"))
    current_value = str(st.session_state.get("votes_local_vote_scope", suggested_key))
    if has_region_filter:
        current_value = suggested_key
    if current_value not in scope_by_key:
        current_value = suggested_key
    st.session_state["votes_local_vote_scope"] = current_value

    selected_key = st.selectbox(
        "세부 선거구/집계 범위",
        options=option_keys,
        index=option_keys.index(current_value),
        key="votes_local_vote_scope",
        format_func=lambda key: scope_by_key[str(key)].label,
    )
    scope = scope_by_key[str(selected_key)]
    st.caption(
        f"이 선거는 한 선거KEY 안에 여러 지역별 선거가 들어 있어 `{scope.label}` 기준으로 득표수와 득표율을 계산합니다."
    )
    return scope


def _default_region_level(selected: dict[str, list[str]], local_vote_scope: ElectionScope | None) -> str:
    if selected.get("읍면동명"):
        return "읍면동"
    if local_vote_scope is not None:
        if local_vote_scope.level == "시도":
            return "구시군"
        if local_vote_scope.level in {"구시군", "선거구"}:
            return "읍면동"
    return resolve_region_level(selected, "구시군")


def _trend_group_display_label(value: object, entity_type: str) -> object:
    if pd.isna(value) or entity_type not in TREND_GROUP_ENTITY_TYPES:
        return value
    normalized = normalize_group_legend_label(value, entity_type=entity_type)
    return TREND_GROUP_DISPLAY_LABELS.get(normalized, normalized)


def _apply_trend_group_display_labels(df: pd.DataFrame, entity_type: str, label_col: str) -> pd.DataFrame:
    if df.empty or entity_type not in TREND_GROUP_ENTITY_TYPES or label_col not in df.columns:
        return df
    result = df.copy()
    result[label_col] = result[label_col].map(lambda value: _trend_group_display_label(value, entity_type))
    return result


def _join_party_names(values: pd.Series) -> str:
    parties: list[str] = []
    seen: set[str] = set()
    for value in values.dropna().astype("string").tolist():
        party = str(value).strip()
        if not party or party in seen:
            continue
        seen.add(party)
        parties.append(party)
    return " + ".join(parties) if parties else "-"


def _build_trend_party_membership(votes_df: pd.DataFrame, entity_type: str, label_col: str) -> pd.DataFrame:
    required = {"선거KEY", label_col, "정당명"}
    if votes_df.empty or entity_type not in TREND_GROUP_ENTITY_TYPES or not required.issubset(votes_df.columns):
        return pd.DataFrame(columns=["선거KEY", label_col, TREND_PARTY_HOVER_COL])

    columns = ["선거KEY", label_col, "정당명"] + (["득표수"] if "득표수" in votes_df.columns else [])
    source = votes_df.loc[:, columns].copy()
    if "득표수" in source.columns:
        source = source.loc[pd.to_numeric(source["득표수"], errors="coerce").fillna(0).gt(0)].copy()
    source[label_col] = source[label_col].map(lambda value: _trend_group_display_label(value, entity_type))
    source["정당명"] = source["정당명"].astype("string").str.strip()
    source = source.dropna(subset=["선거KEY", label_col, "정당명"])
    source = source.loc[source["정당명"].ne("")].copy()
    if source.empty:
        return pd.DataFrame(columns=["선거KEY", label_col, TREND_PARTY_HOVER_COL])

    result = (
        source.groupby(["선거KEY", label_col], as_index=False, observed=True)["정당명"]
        .agg(_join_party_names)
        .rename(columns={"정당명": TREND_PARTY_HOVER_COL})
    )
    return result


def _attach_trend_party_membership(trend_df: pd.DataFrame, votes_df: pd.DataFrame, entity_type: str, label_col: str) -> pd.DataFrame:
    if trend_df.empty or entity_type not in TREND_GROUP_ENTITY_TYPES:
        return trend_df
    membership = _build_trend_party_membership(votes_df, entity_type, label_col)
    if membership.empty:
        result = trend_df.copy()
        result[TREND_PARTY_HOVER_COL] = "-"
        return result
    result = trend_df.merge(membership, on=["선거KEY", label_col], how="left", copy=False)
    result[TREND_PARTY_HOVER_COL] = result[TREND_PARTY_HOVER_COL].fillna("-")
    return result


def _build_trend_party_summary(votes_df: pd.DataFrame, entity_type: str, label_col: str, visible_labels: list[str]) -> list[tuple[str, str]]:
    required = {label_col, "정당명"}
    if votes_df.empty or entity_type not in TREND_GROUP_ENTITY_TYPES or not required.issubset(votes_df.columns):
        return []
    source = votes_df.loc[:, [label_col, "정당명"] + (["득표수"] if "득표수" in votes_df.columns else [])].copy()
    if "득표수" in source.columns:
        source = source.loc[pd.to_numeric(source["득표수"], errors="coerce").fillna(0).gt(0)].copy()
    source[label_col] = source[label_col].map(lambda value: _trend_group_display_label(value, entity_type))
    source["정당명"] = source["정당명"].astype("string").str.strip()
    source = source.dropna(subset=[label_col, "정당명"])
    source = source.loc[source["정당명"].ne("")].copy()
    if visible_labels:
        source = source.loc[source[label_col].astype("string").isin([str(label) for label in visible_labels])].copy()
    if source.empty:
        return []

    summary = source.groupby(label_col, observed=True)["정당명"].agg(_join_party_names).to_dict()
    ordered_labels = [label for label in TREND_PARTY_SUMMARY_ORDER if label in summary]
    ordered_labels.extend([label for label in visible_labels if label in summary and label not in ordered_labels])
    return [(label, str(summary[label])) for label in ordered_labels]


def _build_entity_trend_pivot_frame(
    trend_df: pd.DataFrame,
    label_col: str,
    visible_labels: list[str] | None = None,
) -> pd.DataFrame:
    if trend_df.empty or label_col not in trend_df.columns or "득표율_계산" not in trend_df.columns:
        return pd.DataFrame()

    meta_cols = [
        column
        for column in ["선거시점", "선거KEY", "선거명", "선거종류", "선거라벨", "선거축라벨"]
        if column in trend_df.columns
    ]
    if not meta_cols:
        meta_cols = ["선거KEY"]

    pivot = (
        trend_df.pivot_table(
            index=meta_cols,
            columns=label_col,
            values="득표율_계산",
            aggfunc="first",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivot.columns.name = None

    ordered_value_cols = [str(label) for label in visible_labels or [] if str(label) in pivot.columns]
    remaining_value_cols = [
        column
        for column in pivot.columns
        if column not in meta_cols and column not in ordered_value_cols
    ]
    ordered_cols = [*meta_cols, *ordered_value_cols, *remaining_value_cols]
    sort_cols = [column for column in ["선거시점", "선거KEY"] if column in pivot.columns]
    if sort_cols:
        pivot = pivot.sort_values(by=sort_cols, kind="stable").reset_index(drop=True)
    return pivot.loc[:, ordered_cols]


def _build_entity_trend_pivot_note_frame(
    trend_df: pd.DataFrame,
    label_col: str,
    visible_labels: list[str] | None = None,
) -> pd.DataFrame:
    if trend_df.empty or label_col not in trend_df.columns or TREND_PARTY_HOVER_COL not in trend_df.columns:
        return pd.DataFrame()

    meta_cols = [
        column
        for column in ["선거시점", "선거KEY", "선거명", "선거종류", "선거라벨", "선거축라벨"]
        if column in trend_df.columns
    ]
    if not meta_cols:
        return pd.DataFrame()

    note_labels = [str(label) for label in visible_labels or []]
    preferred = [label for label in ["민주당계", "국민의힘계", "제3지대"] if label in note_labels]
    if not preferred:
        preferred = [label for label in ["민주당계", "국민의힘계", "제3지대"] if label in trend_df[label_col].astype("string").unique().tolist()]
    if not preferred:
        return pd.DataFrame(columns=[*meta_cols, "비고"])

    source = trend_df.loc[:, [*meta_cols, label_col, TREND_PARTY_HOVER_COL]].copy()
    source[label_col] = source[label_col].astype("string")
    source[TREND_PARTY_HOVER_COL] = source[TREND_PARTY_HOVER_COL].fillna("-").astype("string")
    source = source.loc[source[label_col].isin(preferred)].copy()
    if source.empty:
        return pd.DataFrame(columns=[*meta_cols, "비고"])

    def _note_for_group(group: pd.DataFrame) -> str:
        notes: list[str] = []
        for label in preferred:
            values = group.loc[group[label_col].astype("string") == label, TREND_PARTY_HOVER_COL].dropna().astype("string")
            text = "-" if values.empty else str(values.iloc[0])
            notes.append(f"{TREND_PIVOT_NOTE_LABELS.get(label, label)}: {text}")
        return " / ".join(notes)

    return source.groupby(meta_cols, observed=True).apply(_note_for_group).reset_index(name="비고")


def _is_party_top_winner_election(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value)
    return any(keyword in text for keyword in TREND_TOP_PARTY_ELECTION_KEYWORDS)


def _top_candidate_label(row: pd.Series) -> str:
    candidate = "" if pd.isna(row.get("후보명")) else str(row.get("후보명")).strip()
    party = "" if pd.isna(row.get("정당명")) else str(row.get("정당명")).strip()
    label = "" if pd.isna(row.get("후보라벨")) else str(row.get("후보라벨")).strip()
    if label and candidate and party and candidate != party:
        return label
    if candidate and party and candidate != party:
        return f"{party} {candidate}"
    return candidate or party or label or "-"


def _top_party_label(row: pd.Series) -> str:
    party = "" if pd.isna(row.get("정당명")) else str(row.get("정당명")).strip()
    return party or "-"


def _build_entity_trend_top_winner_frame(votes_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [column for column in ["선거시점", "선거KEY", "선거명", "선거종류", "선거라벨", "선거축라벨"] if column in votes_df.columns]
    if votes_df.empty or not meta_cols:
        return pd.DataFrame(columns=[*meta_cols, "1위 정당/후보"])

    candidate_share = calc_entity_share(votes_df, entity_type="후보")
    party_share = calc_entity_share(votes_df, entity_type="정당")
    if candidate_share.empty and party_share.empty:
        return pd.DataFrame(columns=[*meta_cols, "1위 정당/후보"])

    candidate_top = candidate_share.sort_values(by=["선거KEY", "득표수"], ascending=[True, False], kind="stable").drop_duplicates("선거KEY", keep="first")
    party_top = party_share.sort_values(by=["선거KEY", "득표수"], ascending=[True, False], kind="stable").drop_duplicates("선거KEY", keep="first")
    candidate_lookup = {str(row["선거KEY"]): row for _, row in candidate_top.iterrows()} if "선거KEY" in candidate_top.columns else {}
    party_lookup = {str(row["선거KEY"]): row for _, row in party_top.iterrows()} if "선거KEY" in party_top.columns else {}

    meta_source = pd.concat(
        [
            frame.loc[:, [column for column in meta_cols if column in frame.columns]]
            for frame in [candidate_top, party_top]
            if not frame.empty
        ],
        ignore_index=True,
    ).drop_duplicates("선거KEY", keep="first")

    rows: list[dict[str, object]] = []
    for _, meta_row in meta_source.iterrows():
        election_key = str(meta_row["선거KEY"])
        use_party = _is_party_top_winner_election(meta_row.get("선거종류")) or _is_party_top_winner_election(meta_row.get("선거축라벨"))
        source_row = party_lookup.get(election_key) if use_party else candidate_lookup.get(election_key)
        if source_row is None:
            source_row = candidate_lookup.get(election_key)
        if source_row is None:
            source_row = party_lookup.get(election_key)
        row = {column: meta_row.get(column) for column in meta_cols}
        row["1위 정당/후보"] = _top_party_label(source_row) if use_party and source_row is not None else _top_candidate_label(source_row) if source_row is not None else "-"
        rows.append(row)

    result = pd.DataFrame(rows)
    sort_cols = [column for column in ["선거시점", "선거KEY"] if column in result.columns]
    if sort_cols:
        result = result.sort_values(by=sort_cols, kind="stable").reset_index(drop=True)
    return result


def _build_entity_trend_pivot_display_frame(
    pivot_df: pd.DataFrame,
    note_df: pd.DataFrame | None = None,
    top_winner_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pivot_df.empty:
        return pivot_df.copy()

    meta_cols = {
        "선거시점",
        "선거KEY",
        "선거명",
        "선거종류",
        "선거라벨",
        "선거축라벨",
    }
    result = pivot_df.copy()
    if top_winner_df is not None and not top_winner_df.empty:
        merge_cols = [column for column in ["선거시점", "선거KEY", "선거명", "선거종류", "선거라벨", "선거축라벨"] if column in result.columns and column in top_winner_df.columns]
        if merge_cols:
            result = result.merge(top_winner_df.loc[:, [*merge_cols, "1위 정당/후보"]], on=merge_cols, how="left", copy=False)
    if note_df is not None and not note_df.empty:
        merge_cols = [column for column in ["선거시점", "선거KEY", "선거명", "선거종류", "선거라벨", "선거축라벨"] if column in result.columns and column in note_df.columns]
        if merge_cols:
            result = result.merge(note_df.loc[:, [*merge_cols, "비고"]], on=merge_cols, how="left", copy=False)
    result = result.rename(columns=TREND_PIVOT_DISPLAY_RENAME)
    for column in TREND_PIVOT_VALUE_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
    if "1위 정당/후보" not in result.columns:
        result["1위 정당/후보"] = "-"
    if "비고" not in result.columns:
        result["비고"] = "-"
    for column in result.columns:
        if column in meta_cols or column in {"1위 정당/후보", "비고"}:
            continue
        result[column] = result[column].map(format_percent)
    display_cols = [column for column in TREND_PIVOT_DEFAULT_COLUMNS if column in result.columns]
    if display_cols:
        return result.loc[:, display_cols]
    return result


def _vote_gis_label_col(df: pd.DataFrame, level: str) -> str:
    if level == "읍면동" and "읍면동명" in df.columns:
        return "읍면동명"
    if level == "구시군" and "구시군명" in df.columns:
        return "구시군명"
    return "지역"


def _vote_gis_colorscale(df: pd.DataFrame, entity_type: str | None, label_col: str | None, entity_name: object | None) -> list[list[object]] | None:
    if df.empty or not entity_type or not label_col or not entity_name or label_col not in df.columns:
        return None
    color_map = entity_color_map(df, label_col)
    return build_single_hue_colorscale(color_map.get(str(entity_name)))


def _vote_gis_entity_options(df: pd.DataFrame, entity_type: str) -> list[str]:
    label_col = _label_column(entity_type)
    entity_share = calc_entity_share(df, entity_type=entity_type)
    if entity_share.empty or label_col not in entity_share.columns:
        return []
    entity_share = entity_share.sort_values(by="득표수", ascending=False, kind="stable")
    return entity_share[label_col].dropna().astype("string").drop_duplicates().tolist()


def _vote_gis_pair_colorscale(df: pd.DataFrame, entity_type: str, base_entity_name: object | None, compare_entity_name: object | None) -> list[list[object]] | None:
    if df.empty or not base_entity_name or not compare_entity_name:
        return None
    label_col = _label_column(entity_type)
    color_source = calc_entity_share(df, entity_type=entity_type)
    color_map = entity_color_map(color_source, label_col)
    return build_diverging_colorscale(color_map.get(str(compare_entity_name)), color_map.get(str(base_entity_name)))


def _infer_vote_gis_upper_scope(filtered_votes: pd.DataFrame, selected: dict[str, list[str]], level: str) -> dict[str, list[str]]:
    if level == "구시군":
        columns = ["시도명"]
    elif level == "읍면동":
        columns = ["시도명", "구시군명", "일반구명"]
    else:
        columns = []

    inferred: dict[str, list[str]] = {}
    for column in columns:
        selected_values = [str(value) for value in selected.get(column, [])]
        if selected_values:
            inferred[column] = selected_values
            continue
        if filtered_votes.empty or column not in filtered_votes.columns:
            continue
        values = filtered_votes[column].dropna().astype("string").unique().tolist()
        if values:
            inferred[column] = values
    return inferred


def _apply_vote_gis_upper_scope(df: pd.DataFrame, scope: dict[str, list[str]]) -> pd.DataFrame:
    result = df
    for column, values in scope.items():
        if not values or column not in result.columns:
            continue
        result = result.loc[result[column].astype("string").isin(values)]
    return result.copy() if result is not df else df


def _swing_vote_selection(selected: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in selected.items()}
    result["정당명"] = []
    result["후보명"] = []
    result["RowType"] = []
    return result


def _swing_region_level(selected: dict[str, list[str]]) -> str:
    if selected.get("구시군명") or selected.get("일반구명"):
        return "읍면동"
    return "구시군"


def _filter_to_geometry_locations(df: pd.DataFrame, geometry_context: dict[str, object], key_col: str) -> pd.DataFrame:
    if df.empty or key_col not in df.columns or not geometry_context.get("available"):
        return df
    featureidkey = geometry_context.get("featureidkey")
    geojson = geometry_context.get("geojson")
    if not featureidkey or not isinstance(geojson, dict):
        return df
    property_name = str(featureidkey).split(".", maxsplit=1)[1] if "." in str(featureidkey) else str(featureidkey)
    allowed = {
        str(feature.get("properties", {}).get(property_name))
        for feature in geojson.get("features", [])
        if feature.get("properties", {}).get(property_name) is not None
    }
    if not allowed:
        return df
    return df.loc[df[key_col].astype("string").isin(allowed)].copy()


def _filter_swing_vote_pivot_to_geometry_locations(df: pd.DataFrame, geometry_context: dict[str, object], key_col: str) -> pd.DataFrame:
    if df.empty or key_col not in df.columns:
        return df
    special_mask = df[key_col].astype("string").isin(SWING_SPECIAL_VOTE_ROW_KEYS).fillna(False)
    special_rows = df.loc[special_mask].copy()
    filtered = _filter_to_geometry_locations(df.loc[~special_mask].copy(), geometry_context, key_col)
    if special_rows.empty:
        return filtered
    return pd.concat([filtered, special_rows], ignore_index=True)


def _build_region_key(row: pd.Series, level: str) -> str:
    parts: list[str] = []
    for column in ["시도명", "구시군명"]:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    if level == "읍면동":
        general = row.get("일반구명")
        if pd.notna(general) and str(general).strip():
            parts.append(str(general).strip())
        dong = row.get("읍면동명")
        if pd.notna(dong) and str(dong).strip():
            parts.append(str(dong).strip())
    return " ".join(parts)


def _apply_admin_rename_aliases(votes_df: pd.DataFrame, level: str) -> pd.DataFrame:
    if votes_df.empty:
        return votes_df

    result = votes_df.copy()
    for column in ["시도명", "구시군명", "일반구명", "읍면동명", "구시군KEY", "읍면동KEY"]:
        if column in result.columns:
            result[column] = result[column].astype("string")
    if {"시도명", "구시군명"}.issubset(result.columns):
        for alias in ADMIN_GUSIGUN_RENAME_ALIASES:
            mask = result["시도명"].astype("string").eq(str(alias["시도명"])) & result["구시군명"].astype("string").eq(str(alias["from"]))
            if not bool(mask.any()):
                continue
            result.loc[mask, "구시군명"] = str(alias["to"])
            if "구시군KEY" in result.columns:
                result.loc[mask, "구시군KEY"] = result.loc[mask].apply(lambda row: _build_region_key(row, "구시군"), axis=1)

    if level != "읍면동" or not {"시도명", "구시군명", "읍면동명"}.issubset(result.columns):
        return result

    for alias in ADMIN_EMD_RENAME_ALIASES:
        mask = (
            result["시도명"].astype("string").eq(str(alias["시도명"]))
            & result["구시군명"].astype("string").eq(str(alias["구시군명"]))
            & result["읍면동명"].astype("string").eq(str(alias["from"]))
        )
        if not bool(mask.any()):
            continue
        result.loc[mask, "읍면동명"] = str(alias["to"])
        if "읍면동KEY" in result.columns:
            result.loc[mask, "읍면동KEY"] = result.loc[mask].apply(lambda row: _build_region_key(row, "읍면동"), axis=1)
    return result


def _static_swing_scores(values: pd.Series, thresholds: tuple[float, float, float, float]) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    numeric = pd.to_numeric(values, errors="coerce")
    result.loc[numeric.le(thresholds[0])] = 1
    result.loc[numeric.gt(thresholds[0]) & numeric.le(thresholds[1])] = 2
    result.loc[numeric.gt(thresholds[1]) & numeric.lt(thresholds[2])] = 3
    result.loc[numeric.ge(thresholds[2]) & numeric.lt(thresholds[3])] = 4
    result.loc[numeric.ge(thresholds[3])] = 5
    return result


def _swing_score_labels(scores: pd.Series) -> pd.Series:
    return scores.map(lambda value: SWING_CATEGORY_ORDER[int(value) - 1] if pd.notna(value) else pd.NA)


def _collapse_swing_party_label(value: object) -> str:
    label = str(value).strip()
    if label in {SWING_DEM_LABEL, SWING_CONSERVATIVE_LABEL, "제3지대"}:
        return label
    return SWING_OTHER_LABEL


def _format_swing_percent(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric) * 100:.1f}%"


def _format_swing_pp_gap(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric) * 100:+.1f}%p"


def _format_swing_vote_gap(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"{int(round(float(numeric))):+,}"


def _swing_display_region_names(df: pd.DataFrame, level: str) -> pd.Series:
    label_col = _vote_gis_label_col(df, level)
    if label_col in df.columns:
        return df[label_col].fillna(df.get("지역", "")).astype("string")
    if "지역" in df.columns:
        return df["지역"].fillna("").astype("string")
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _swing_region_display_column(level: str) -> str:
    return "읍면동명" if level == "읍면동" else "구시군명"


def _swing_election_label_map(share_df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    election_meta_cols = [column for column in ["선거KEY", "선거시점", "선거축라벨", "선거라벨"] if column in share_df.columns]
    if "선거KEY" not in election_meta_cols:
        return {}, []

    election_meta = share_df.loc[:, election_meta_cols].drop_duplicates(subset=["선거KEY"]).copy()
    sort_cols = [column for column in ["선거시점", "선거KEY"] if column in election_meta.columns]
    if sort_cols:
        election_meta = election_meta.sort_values(by=sort_cols, kind="stable").reset_index(drop=True)

    labels: list[str] = []
    for _, row in election_meta.iterrows():
        label = row.get("선거축라벨")
        if pd.isna(label) or not str(label).strip():
            label = row.get("선거라벨")
        if pd.isna(label) or not str(label).strip():
            label = row.get("선거KEY")
        labels.append(str(label))

    duplicated = pd.Series(labels).duplicated(keep=False).tolist()
    label_map: dict[str, str] = {}
    ordered_labels: list[str] = []
    for duplicate, (_, row), label in zip(duplicated, election_meta.iterrows(), labels):
        final_label = f"{label} ({row['선거KEY']})" if duplicate else label
        label_map[str(row["선거KEY"])] = final_label
        ordered_labels.append(final_label)
    return label_map, ordered_labels


def _drop_polling_summary_rows_when_detail_exists(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return source
    summary_mask = pd.Series(False, index=source.index)
    if "투표소KEY" in source.columns:
        summary_mask = summary_mask | source["투표소KEY"].astype("string").fillna("").str.contains("합계", na=False)
    if "구분" in source.columns:
        summary_mask = summary_mask | source["구분"].astype("string").eq("합계").fillna(False)
    if not summary_mask.any():
        return source
    if "선거KEY" not in source.columns:
        return source.loc[~summary_mask].copy() if (~summary_mask).any() else source
    detail_exists = (~summary_mask).groupby(source["선거KEY"].astype("string")).transform("any")
    return source.loc[~(summary_mask & detail_exists)].copy()


def _select_swing_special_vote_source(votes_df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    rowtype = votes_df["RowType"].astype("string")
    if spec.get("key") == SWING_ELECTION_DAY_VOTE_KEY:
        election_day = votes_df.loc[rowtype.eq("선거일투표")].copy()
        polling = votes_df.loc[rowtype.eq("투표소")].copy()
        polling = _drop_polling_summary_rows_when_detail_exists(polling)
        if election_day.empty:
            return polling
        if polling.empty or "선거KEY" not in votes_df.columns:
            return election_day
        election_day_keys = set(election_day["선거KEY"].dropna().astype("string"))
        polling_fallback = polling.loc[~polling["선거KEY"].astype("string").isin(election_day_keys).fillna(False)].copy()
        if polling_fallback.empty:
            return election_day
        return pd.concat([election_day, polling_fallback], ignore_index=True)
    if "excluded_rowtypes" in spec:
        return votes_df.loc[~rowtype.isin(spec["excluded_rowtypes"])].copy()
    return votes_df.loc[rowtype.isin(spec.get("rowtypes", ()))].copy()


def _build_swing_special_vote_rows(
    votes_df: pd.DataFrame,
    level: str,
    key_col: str,
    label_map: dict[str, str],
    specs: tuple[dict[str, object], ...] | None = None,
) -> pd.DataFrame:
    label_col = _label_column("구분")
    required = {"선거KEY", "RowType", label_col, "득표수", "유효투표수"}
    if level != "읍면동" or votes_df.empty or not required.issubset(votes_df.columns):
        return pd.DataFrame()

    result_parts: list[pd.DataFrame] = []
    for spec in specs or SWING_OUTSIDE_VOTE_ROW_SPECS:
        source = _select_swing_special_vote_source(votes_df, spec)
        if source.empty:
            continue

        source[label_col] = source[label_col].map(lambda value: _trend_group_display_label(value, "구분"))
        source["정당계"] = source[label_col].map(_collapse_swing_party_label)
        source["선거축"] = source["선거KEY"].astype("string").map(label_map)
        source = source.loc[source["선거축"].notna()].copy()
        if source.empty:
            continue

        group_cols = ["선거KEY", "선거축", "정당계"]
        grouped = source.groupby(group_cols, as_index=False, observed=True)["득표수"].sum(min_count=1)
        unit_cols = [
            column
            for column in ["선거KEY", "선거구명", "시도명", "구시군명", "일반구명", "읍면동명", "투표소KEY", "구분", "RowType", "유효투표수"]
            if column in source.columns
        ]
        valid_votes = (
            source.loc[:, unit_cols]
            .drop_duplicates(subset=unit_cols)
            .groupby("선거KEY", as_index=False, observed=True)["유효투표수"]
            .sum(min_count=1)
        )
        grouped = grouped.merge(valid_votes, on="선거KEY", how="left", copy=False)
        grouped["득표율_계산"] = grouped["득표수"] / grouped["유효투표수"].replace(0, pd.NA)
        grouped[key_col] = str(spec["key"])
        grouped["지역"] = str(spec["label"])
        grouped["읍면동명"] = str(spec["label"])
        result_parts.append(grouped.loc[:, [key_col, "지역", "읍면동명", "선거KEY", "선거축", "정당계", "득표수", "득표율_계산"]])

    if not result_parts:
        return pd.DataFrame()
    return pd.concat(result_parts, ignore_index=True)


def _build_swing_rowtype_vote_pivot_table(votes_df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    if votes_df.empty or spec.get("key") not in {SWING_LOCAL_EARLY_VOTE_KEY, SWING_ELECTION_DAY_VOTE_KEY}:
        return pd.DataFrame()
    source = _apply_admin_rename_aliases(votes_df, "읍면동")
    source = _select_swing_special_vote_source(source, spec)
    if source.empty:
        return pd.DataFrame()

    source = source.copy()
    source["RowType"] = "읍면동"
    return _build_swing_vote_pivot_table(source, "읍면동")


def _build_swing_vote_pivot_table(votes_df: pd.DataFrame, level: str) -> pd.DataFrame:
    label_col = _label_column("구분")
    votes_df = _apply_admin_rename_aliases(votes_df, level)
    share_df = calc_entity_share_by_region(votes_df, entity_type="구분", level=level)
    if share_df.empty or label_col not in share_df.columns:
        return pd.DataFrame()

    share_df = _apply_trend_group_display_labels(share_df, "구분", label_col)
    key_col = "읍면동KEY" if level == "읍면동" else "구시군KEY"
    if key_col not in share_df.columns:
        key_col = "지역"
    if key_col not in share_df.columns or "선거KEY" not in share_df.columns:
        return pd.DataFrame()

    display_col = _swing_region_display_column(level)
    region_cols = [column for column in [key_col, "지역", "시도명", "구시군명", "일반구명", "읍면동명"] if column in share_df.columns]
    region_meta = share_df.loc[:, region_cols].drop_duplicates(subset=[key_col]).copy()
    if display_col not in region_meta.columns:
        region_meta[display_col] = _swing_display_region_names(region_meta, level)
    else:
        region_meta[display_col] = region_meta[display_col].fillna(_swing_display_region_names(region_meta, level)).astype("string")

    label_map, ordered_election_labels = _swing_election_label_map(share_df)
    if not label_map:
        return pd.DataFrame()
    special_vote_rows = _build_swing_special_vote_rows(votes_df, level, key_col, label_map)
    if not special_vote_rows.empty:
        special_meta_rows: list[dict[str, str]] = []
        for special_key, special_label in SWING_SPECIAL_VOTE_ROW_LABELS.items():
            if not special_vote_rows[key_col].astype("string").eq(special_key).fillna(False).any():
                continue
            special_meta = {key_col: special_key, display_col: special_label}
            if "지역" in region_meta.columns:
                special_meta["지역"] = special_label
            if "읍면동명" in region_meta.columns:
                special_meta["읍면동명"] = special_label
            special_meta_rows.append(special_meta)
        if special_meta_rows:
            region_meta = pd.concat([region_meta, pd.DataFrame(special_meta_rows)], ignore_index=True)

    collapsed = share_df.copy()
    collapsed["정당계"] = collapsed[label_col].map(_collapse_swing_party_label)
    collapsed = collapsed.loc[collapsed["정당계"].isin([SWING_DEM_LABEL, SWING_CONSERVATIVE_LABEL])].copy()
    if not special_vote_rows.empty:
        collapsed = pd.concat(
            [
                collapsed,
                special_vote_rows.loc[:, [column for column in collapsed.columns if column in special_vote_rows.columns]],
            ],
            ignore_index=True,
        )
    if collapsed.empty:
        return pd.DataFrame()

    collapsed["선거축"] = collapsed["선거KEY"].astype("string").map(label_map)
    grouped = (
        collapsed.groupby([key_col, "선거KEY", "선거축", "정당계"], as_index=False, observed=True)[["득표수", "득표율_계산"]]
        .sum(min_count=1)
    )
    availability = (
        grouped.pivot_table(index=key_col, columns="선거축", values="득표수", aggfunc="count")
        .reindex(columns=ordered_election_labels)
        .notna()
    )

    result = region_meta.loc[:, [key_col, display_col]].drop_duplicates(subset=[key_col]).copy()
    value_column_groups: list[list[str]] = []
    for party_label in [SWING_DEM_LABEL, SWING_CONSERVATIVE_LABEL]:
        party_wide = (
            grouped.loc[grouped["정당계"].eq(party_label)]
            .pivot_table(index=key_col, columns="선거축", values="득표율_계산", aggfunc="first")
            .reindex(columns=ordered_election_labels)
        )
        for label in ordered_election_labels:
            if label in party_wide.columns and label in availability.columns:
                has_data = availability[label].reindex(party_wide.index).fillna(False)
                party_wide[label] = party_wide[label].where(~has_data, party_wide[label].fillna(0.0))
        party_wide = party_wide.reset_index()
        party_cols = [f"{party_label} {label} 득표율" for label in ordered_election_labels]
        party_wide = party_wide.rename(columns=dict(zip(ordered_election_labels, party_cols)))
        party_wide[f"{party_label} 득표율 평균"] = party_wide[party_cols].mean(axis=1)
        result = result.merge(party_wide.loc[:, [key_col, *party_cols, f"{party_label} 득표율 평균"]], on=key_col, how="left", copy=False)
        value_column_groups.append([*party_cols, f"{party_label} 득표율 평균"])

    party_votes = (
        grouped.pivot_table(index=[key_col, "선거축"], columns="정당계", values="득표수", aggfunc="first")
        .reset_index()
    )
    party_rates = (
        grouped.pivot_table(index=[key_col, "선거축"], columns="정당계", values="득표율_계산", aggfunc="first")
        .reset_index()
    )
    for frame in [party_votes, party_rates]:
        for party_label in [SWING_DEM_LABEL, SWING_CONSERVATIVE_LABEL]:
            if party_label not in frame.columns:
                frame[party_label] = 0
            else:
                frame[party_label] = frame[party_label].fillna(0)

    party_votes["득표수 격차"] = party_votes[SWING_DEM_LABEL] - party_votes[SWING_CONSERVATIVE_LABEL]
    party_rates["득표율 격차"] = party_rates[SWING_DEM_LABEL] - party_rates[SWING_CONSERVATIVE_LABEL]

    vote_gap_cols = [f"{label} 득표수 격차" for label in ordered_election_labels]
    vote_gap_wide = (
        party_votes.pivot_table(index=key_col, columns="선거축", values="득표수 격차", aggfunc="first")
        .reindex(columns=ordered_election_labels)
        .reset_index()
        .rename(columns=dict(zip(ordered_election_labels, vote_gap_cols)))
    )
    rate_gap_cols = [f"{label} 득표율 격차" for label in ordered_election_labels]
    rate_gap_wide = (
        party_rates.pivot_table(index=key_col, columns="선거축", values="득표율 격차", aggfunc="first")
        .reindex(columns=ordered_election_labels)
        .reset_index()
        .rename(columns=dict(zip(ordered_election_labels, rate_gap_cols)))
    )
    result = result.merge(vote_gap_wide.loc[:, [key_col, *vote_gap_cols]], on=key_col, how="left", copy=False)
    result = result.merge(rate_gap_wide.loc[:, [key_col, *rate_gap_cols]], on=key_col, how="left", copy=False)

    ordered_columns = [key_col, display_col, *value_column_groups[0], *value_column_groups[1], *vote_gap_cols, *rate_gap_cols]
    result = result.loc[:, [column for column in ordered_columns if column in result.columns]]
    special_order = result[key_col].astype("string").map(SWING_SPECIAL_VOTE_ROW_ORDER)
    result["__special_sort"] = special_order.notna().astype(int)
    result["__special_order"] = special_order.fillna(-1).astype(int)
    result = (
        result.sort_values(by=["__special_sort", "__special_order", display_col], kind="stable")
        .drop(columns=["__special_sort", "__special_order"])
        .reset_index(drop=True)
    )
    return result


def _swing_vote_pivot_display_frame(pivot_df: pd.DataFrame, level: str) -> pd.DataFrame:
    if pivot_df.empty:
        return pivot_df.copy()
    key_col = "읍면동KEY" if level == "읍면동" else "구시군KEY"
    display = pivot_df.drop(columns=[key_col], errors="ignore").copy()
    for column in display.columns:
        if column == _swing_region_display_column(level):
            continue
        if column.endswith("득표수 격차"):
            display[column] = display[column].map(_format_swing_vote_gap)
        elif column.endswith("득표율 격차"):
            display[column] = display[column].map(_format_swing_pp_gap)
        elif "득표율" in column:
            display[column] = display[column].map(_format_swing_percent)
    return display


def _build_swing_frames(votes_df: pd.DataFrame, level: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    label_col = _label_column("구분")
    votes_df = _apply_admin_rename_aliases(votes_df, level)
    share_df = calc_entity_share_by_region(votes_df, entity_type="구분", level=level)
    if share_df.empty or label_col not in share_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    share_df = _apply_trend_group_display_labels(share_df, "구분", label_col)
    key_col = "읍면동KEY" if level == "읍면동" else "구시군KEY"
    if key_col not in share_df.columns:
        key_col = "지역"
    region_cols = [
        column
        for column in [key_col, "지역", "시도명", "구시군명", "일반구명", "읍면동명"]
        if column in share_df.columns
    ]
    group_cols = ["선거KEY", key_col]

    pivot = (
        share_df.loc[share_df[label_col].astype("string").isin([SWING_DEM_LABEL, SWING_CONSERVATIVE_LABEL])]
        .pivot_table(index=group_cols, columns=label_col, values="득표율_계산", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    if pivot.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    for column in [SWING_DEM_LABEL, SWING_CONSERVATIVE_LABEL]:
        if column not in pivot.columns:
            pivot[column] = 0.0
    pivot["민주-국민의힘 격차"] = pivot[SWING_DEM_LABEL] - pivot[SWING_CONSERVATIVE_LABEL]

    meta = share_df.loc[:, region_cols].drop_duplicates(subset=[key_col]).copy()
    election_meta_cols = [column for column in ["선거KEY", "선거시점", "선거명", "선거라벨", "선거축라벨"] if column in share_df.columns]
    election_meta = share_df.loc[:, election_meta_cols].drop_duplicates(subset=["선거KEY"]) if "선거KEY" in election_meta_cols else pd.DataFrame()
    if not election_meta.empty:
        pivot = pivot.merge(election_meta, on="선거KEY", how="left", copy=False)

    summary = (
        pivot.groupby(key_col, as_index=False, observed=True)
        .agg(
            평균격차=("민주-국민의힘 격차", "mean"),
            **{
                "민주당계 평균득표율": (SWING_DEM_LABEL, "mean"),
                "국민의힘계 평균득표율": (SWING_CONSERVATIVE_LABEL, "mean"),
            },
            최저격차=("민주-국민의힘 격차", "min"),
            최고격차=("민주-국민의힘 격차", "max"),
            선거수=("선거KEY", "nunique"),
        )
        .merge(meta, on=key_col, how="left", copy=False)
    )
    sort_cols = [column for column in ["선거시점", "선거KEY"] if column in pivot.columns]
    latest = pivot.sort_values(by=sort_cols or ["선거KEY"], kind="stable").groupby(key_col, as_index=False, observed=True).tail(1)
    latest = latest.loc[:, [key_col, "민주-국민의힘 격차"]].rename(columns={"민주-국민의힘 격차": "최근격차"})
    summary = summary.merge(latest, on=key_col, how="left", copy=False)
    summary["평균격차스윙점수"] = _static_swing_scores(summary["평균격차"], thresholds=(-0.10, -0.03, 0.03, 0.10))
    summary["평균격차스윙분류"] = _swing_score_labels(summary["평균격차스윙점수"])
    summary["지도지표_평균격차"] = summary["평균격차스윙점수"].astype("float64")

    collapsed_share = share_df.copy()
    collapsed_share["정당계"] = collapsed_share[label_col].map(_collapse_swing_party_label)
    collapsed_group_cols = list(dict.fromkeys([*group_cols, "지역", "정당계"]))
    collapsed_share = (
        collapsed_share.groupby(collapsed_group_cols, as_index=False, observed=True)[["득표수", "득표율_계산"]]
        .sum(min_count=1)
    )

    winner_source = collapsed_share.sort_values(by=["선거KEY", key_col, "득표수"], ascending=[True, True, False], kind="stable")
    winners = winner_source.drop_duplicates(subset=["선거KEY", key_col]).copy()
    winners = winners.rename(columns={"정당계": "승리정당계"})
    win_group_cols = list(dict.fromkeys([key_col, "지역", "승리정당계"]))
    win_counts = (
        winners.groupby(win_group_cols, as_index=False, observed=True)
        .size()
        .rename(columns={"size": "승리횟수"})
    )
    win_counts = win_counts.merge(meta.loc[:, [key_col, *[column for column in ["구시군명", "읍면동명"] if column in meta.columns]]], on=key_col, how="left", copy=False)
    win_counts["차트지역명"] = _swing_display_region_names(win_counts, level)
    win_pivot = (
        win_counts.pivot_table(index=key_col, columns="승리정당계", values="승리횟수", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for column in SWING_PARTY_ORDER:
        if column not in win_pivot.columns:
            win_pivot[column] = 0
    win_pivot = win_pivot.rename(
        columns={
            SWING_DEM_LABEL: "민주당계 승리횟수",
            SWING_CONSERVATIVE_LABEL: "국민의힘계 승리횟수",
            "제3지대": "제3지대 승리횟수",
            SWING_OTHER_LABEL: "무소속/기타 승리횟수",
        }
    )
    win_pivot["승리횟수격차"] = win_pivot["민주당계 승리횟수"] - win_pivot["국민의힘계 승리횟수"]
    win_pivot["양대정당계승리횟수"] = win_pivot["민주당계 승리횟수"] + win_pivot["국민의힘계 승리횟수"]
    win_pivot["승리균형지수"] = win_pivot["승리횟수격차"] / win_pivot["양대정당계승리횟수"].replace(0, pd.NA)
    win_pivot["승리횟수스윙점수"] = _static_swing_scores(win_pivot["승리균형지수"], thresholds=(-0.60, -0.20, 0.20, 0.60))
    win_pivot["승리횟수스윙분류"] = _swing_score_labels(win_pivot["승리횟수스윙점수"])
    win_pivot["지도지표_승리횟수"] = win_pivot["승리횟수스윙점수"].astype("float64")
    summary = summary.merge(
        win_pivot.loc[
            :,
            [
                key_col,
                "민주당계 승리횟수",
                "국민의힘계 승리횟수",
                "제3지대 승리횟수",
                "무소속/기타 승리횟수",
                "승리횟수격차",
                "승리균형지수",
                "승리횟수스윙점수",
                "승리횟수스윙분류",
                "지도지표_승리횟수",
            ],
        ],
        on=key_col,
        how="left",
        copy=False,
    )

    avg_group_cols = list(dict.fromkeys([key_col, "지역", "정당계"]))
    avg_share_df = (
        collapsed_share.groupby(avg_group_cols, as_index=False, observed=True)["득표율_계산"]
        .mean()
        .rename(columns={"득표율_계산": "평균득표율"})
    )
    avg_share_df = avg_share_df.merge(meta.loc[:, [key_col, *[column for column in ["구시군명", "읍면동명"] if column in meta.columns]]], on=key_col, how="left", copy=False)
    avg_share_df["차트지역명"] = _swing_display_region_names(avg_share_df, level)

    region_names = _swing_display_region_names(summary, level)
    summary["승리횟수지도라벨"] = summary.apply(
        lambda row: (
            f"{region_names.loc[row.name]}<br>"
            f"{int(row['민주당계 승리횟수']) if pd.notna(row.get('민주당계 승리횟수')) else 0} : "
            f"{int(row['국민의힘계 승리횟수']) if pd.notna(row.get('국민의힘계 승리횟수')) else 0}"
        ),
        axis=1,
    )
    summary["평균득표율지도라벨"] = summary.apply(
        lambda row: (
            f"{region_names.loc[row.name]}<br>"
            f"{_format_swing_percent(row.get('민주당계 평균득표율'))} vs {_format_swing_percent(row.get('국민의힘계 평균득표율'))}"
        ),
        axis=1,
    )
    return summary, win_counts, avg_share_df, SWING_PARTY_ORDER


def _swing_win_count_chart(win_counts: pd.DataFrame, category_order: list[str]) -> go.Figure:
    fig = go.Figure()
    if win_counts.empty or not {"지역", "승리정당계", "승리횟수"}.issubset(win_counts.columns):
        fig.add_annotation(text="표시할 스윙 지역 승리 횟수 데이터가 없습니다.", x=0.5, y=0.5, showarrow=False, font={"size": 18})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    display_col = "차트지역명" if "차트지역명" in win_counts.columns else "지역"
    totals = win_counts.groupby(display_col, as_index=False, observed=True)["승리횟수"].sum()
    region_order = totals.sort_values(by=["승리횟수", display_col], ascending=[False, True], kind="stable")[display_col].astype("string").head(40).tolist()
    chart_df = win_counts.loc[win_counts[display_col].astype("string").isin(region_order)].copy()
    color_map = {**entity_color_map(chart_df, "승리정당계"), **SWING_PARTY_COLORS}
    ordered_categories = [label for label in SWING_PARTY_ORDER if label in category_order]
    ordered_categories.extend([label for label in category_order if label not in ordered_categories])
    for category in ordered_categories:
        part = chart_df.loc[chart_df["승리정당계"].astype("string") == str(category)].copy()
        values = part.set_index(display_col)["승리횟수"].reindex(region_order, fill_value=0)
        fig.add_trace(
            go.Bar(
                x=values.to_numpy(),
                y=region_order,
                name=str(category),
                orientation="h",
                marker_color=color_map.get(str(category), "#8D99AE"),
                text=values.map(lambda value: "" if int(value) == 0 else f"{int(value)}회"),
                textposition="inside",
                hovertemplate="%{y}<br>%{fullData.name}: %{x}회<extra></extra>",
            )
        )
    fig.update_layout(
        title="지역별 정당계 승리 횟수",
        barmode="stack",
        height=max(440, 70 + 28 * len(region_order)),
        margin=dict(l=20, r=20, t=60, b=30),
        legend_title_text="",
        legend=dict(traceorder="normal"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=16),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(region_order)), title="")
    fig.update_xaxes(title="승리 횟수", dtick=1, rangemode="tozero")
    return fig


def _swing_avg_share_chart(avg_share_df: pd.DataFrame, category_order: list[str]) -> go.Figure:
    fig = go.Figure()
    if avg_share_df.empty or not {"지역", "정당계", "평균득표율"}.issubset(avg_share_df.columns):
        fig.add_annotation(text="표시할 지역별 평균 득표율 데이터가 없습니다.", x=0.5, y=0.5, showarrow=False, font={"size": 18})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    display_col = "차트지역명" if "차트지역명" in avg_share_df.columns else "지역"
    order_source = avg_share_df.loc[avg_share_df["정당계"].astype("string") == SWING_DEM_LABEL].copy()
    if order_source.empty:
        order_source = avg_share_df.groupby(display_col, as_index=False, observed=True)["평균득표율"].sum()
    region_order = (
        order_source.sort_values(by=["평균득표율", display_col], ascending=[False, True], kind="stable")[display_col]
        .astype("string")
        .head(40)
        .tolist()
    )
    chart_df = avg_share_df.loc[avg_share_df[display_col].astype("string").isin(region_order)].copy()
    color_map = {**entity_color_map(chart_df, "정당계"), **SWING_PARTY_COLORS}
    ordered_categories = [label for label in SWING_PARTY_ORDER if label in category_order]
    ordered_categories.extend([label for label in category_order if label not in ordered_categories])
    for category in ordered_categories:
        part = chart_df.loc[chart_df["정당계"].astype("string") == str(category)].copy()
        values = part.set_index(display_col)["평균득표율"].reindex(region_order, fill_value=0)
        fig.add_trace(
            go.Bar(
                x=values.to_numpy(),
                y=region_order,
                name=str(category),
                orientation="h",
                marker_color=color_map.get(str(category), "#8D99AE"),
                text=values.map(lambda value: "" if pd.isna(value) or float(value) <= 0 else f"{float(value) * 100:.1f}%"),
                textposition="inside",
                hovertemplate="%{y}<br>%{fullData.name}: %{x:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        title="지역별 정당계 평균 득표율",
        barmode="stack",
        height=max(440, 70 + 28 * len(region_order)),
        margin=dict(l=20, r=20, t=60, b=30),
        legend_title_text="",
        legend=dict(traceorder="normal"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=16),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(region_order)), title="")
    fig.update_xaxes(title="평균 득표율", tickformat=".0%", rangemode="tozero")
    return fig


def _election_title_label(election_key: str, filter_options: dict[str, object]) -> str:
    key_to_label = filter_options.get("election_key_to_label", {})
    label = str(key_to_label.get(str(election_key), election_key))
    label = label.replace(" | ", " ")
    key_prefix = f"{election_key} "
    if label.startswith(key_prefix):
        label = label[len(key_prefix):]
    return label


def _sanitize_multiselect_state(widget_key: str, options: list[str], default_values: list[str]) -> None:
    current_values = [str(value) for value in st.session_state.get(widget_key, [])]
    option_set = {str(value) for value in options}
    sanitized = [value for value in current_values if value in option_set]
    if not sanitized:
        sanitized = list(default_values)
    st.session_state[widget_key] = sanitized


def _official_vote_selection(selected: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in selected.items()}
    result["RowType"] = []
    return result


def main() -> None:
    st.title("득표")

    try:
        app_data = _load_bundle()
    except FileNotFoundError:
        st.error("먼저 python scripts/build_parquet.py 를 실행하세요.")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.caption("cache parquet 스키마가 바뀌었을 수 있습니다. python scripts/build_parquet.py 를 다시 실행하세요.")
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.caption("cache parquet를 다시 생성한 뒤 앱을 재실행해 주세요.")
        st.stop()

    filter_options = app_data["filter_options"]
    selected = render_sidebar_filters(filter_options, include_party=True, include_candidate=True)
    analysis_selected = _official_vote_selection(selected)
    filtered_votes = apply_common_filters(app_data["votes"], analysis_selected)
    filtered_turnout = apply_common_filters(app_data["turnout"], analysis_selected)

    if filtered_votes.empty:
        st.warning("현재 필터 조건에 맞는 득표 데이터가 없습니다.")
        st.stop()

    st.caption("현재 필터: " + build_breadcrumb_text(analysis_selected, filter_options))
    if selected.get("RowType"):
        st.caption("득표 시각화는 공식 합계 기준을 유지하기 위해 RowType 필터를 제외하고 계산합니다.")

    current_election_key = _resolve_current_election_key(filtered_votes, filter_options)
    current_votes = filtered_votes.loc[filtered_votes["선거KEY"].astype("string") == current_election_key].copy()
    current_turnout = filtered_turnout.loc[filtered_turnout["선거KEY"].astype("string") == current_election_key].copy()
    local_vote_scope = _resolve_local_vote_scope(current_votes, analysis_selected)
    current_votes = filter_by_election_scope(current_votes, local_vote_scope)
    current_turnout = filter_by_election_scope(current_turnout, local_vote_scope)
    scoped_all_votes = filter_by_election_scope(app_data["votes"], local_vote_scope)
    scoped_all_turnout = filter_by_election_scope(app_data["turnout"], local_vote_scope)
    scoped_filtered_votes = filter_by_election_scope(filtered_votes, local_vote_scope)
    current_election_kind = _first_text_value(current_votes, "선거종류")
    if local_vote_scope is not None and current_election_kind and "선거종류" in scoped_filtered_votes.columns:
        scoped_filtered_votes = scoped_filtered_votes.loc[
            scoped_filtered_votes["선거종류"].astype("string").eq(str(current_election_kind))
        ].copy()

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    with control_col1:
        entity_type = st.selectbox(
            "분석 대상",
            ENTITY_TYPE_OPTIONS,
            index=ENTITY_TYPE_OPTIONS.index("후보"),
            key="votes_entity_type",
            format_func=_display_entity_type,
        )
    with control_col2:
        region_level = st.selectbox("지역 레벨", ["시도", "구시군", "읍면동"], index=["시도", "구시군", "읍면동"].index(_default_region_level(analysis_selected, local_vote_scope)), key="votes_region_level")
    with control_col3:
        chart_mode = st.selectbox("비교 차트", ["grouped", "stacked", "100% stacked", "lollipop", "heatmap"], key="votes_chart_mode")
    with control_col4:
        distribution_mode = st.selectbox("분포 차트", ["box", "violin"], key="votes_distribution_mode")

    entity_type_label = _display_entity_type(entity_type)
    st.caption("구분1/구분2/성향 기준은 DimParty 정당 분류를 사용합니다.")

    label_col = _label_column(entity_type)
    current_entity_share = calc_entity_share(current_votes, entity_type=entity_type)
    entity_options = current_entity_share[label_col].dropna().astype("string").unique().tolist()
    default_entities = entity_options[: min(5, len(entity_options))]
    _sanitize_multiselect_state("votes_selected_entities", entity_options, default_entities)
    selected_entities = st.multiselect(
        f"{entity_type_label} 비교 대상",
        options=entity_options,
        default=default_entities,
        key="votes_selected_entities",
    )
    if not selected_entities:
        selected_entities = default_entities
    legend_entities = _entity_legend_order(current_votes, entity_type, label_col, selected_entities)

    summary = calc_votes_summary(current_votes)
    gap_df = calc_entity_top2_gap(current_votes, entity_type=entity_type)
    top_gap = gap_df.iloc[0] if not gap_df.empty else None
    primary_entity = selected_entities[0] if selected_entities else None
    highlight_entities = _build_highlight_entities(entity_type, top_gap, primary_entity)
    comparison_selected = build_region_comparison_selection(analysis_selected, region_level)
    comparison_votes = apply_common_filters(scoped_all_votes, comparison_selected)
    comparison_turnout = apply_common_filters(scoped_all_turnout, comparison_selected)
    current_comparison_votes = comparison_votes.loc[comparison_votes["선거KEY"].astype("string") == current_election_key].copy()
    current_comparison_turnout = comparison_turnout.loc[comparison_turnout["선거KEY"].astype("string") == current_election_key].copy()
    current_comparison_votes = filter_by_election_scope(current_comparison_votes, local_vote_scope)
    current_comparison_turnout = filter_by_election_scope(current_comparison_turnout, local_vote_scope)

    region_share_df = calc_entity_share_by_region(current_comparison_votes, entity_type=entity_type, level=region_level)
    region_share_df = region_share_df.loc[region_share_df[label_col].astype("string").isin(selected_entities)].copy()
    region_order = region_share_df.groupby("지역", observed=True)["득표수"].sum(min_count=1).sort_values(ascending=False).head(18).index.tolist()
    region_share_df = region_share_df.loc[region_share_df["지역"].isin(region_order)].copy()

    competition_df = calc_top2_gap_by_region(current_comparison_votes, level=region_level)
    rowtype_entity_all_df = calc_rowtype_entity_breakdown(current_votes, entity_type=entity_type)
    rowtype_entity_df = rowtype_entity_all_df.copy()
    rowtype_entity_df = rowtype_entity_df.loc[rowtype_entity_df[label_col].astype("string").isin(selected_entities)].copy()
    distribution_df = calc_distribution_by_region(current_comparison_votes, entity_type=entity_type, level=region_level)
    distribution_df = distribution_df.loc[distribution_df[label_col].astype("string").isin(selected_entities)].copy()
    scatter_df = calc_turnout_vote_scatter(
        current_comparison_turnout,
        current_comparison_votes,
        level=region_level,
        entity_type=entity_type,
        entity_name=selected_entities,
    )

    gusigun_focus_selected = build_scope_selection(analysis_selected, "gusigun")
    current_gusigun_votes = apply_common_filters(scoped_all_votes, gusigun_focus_selected)
    current_gusigun_votes = current_gusigun_votes.loc[current_gusigun_votes["선거KEY"].astype("string") == current_election_key].copy()
    current_gusigun_votes = filter_by_election_scope(current_gusigun_votes, local_vote_scope)
    focus_gusigun_keys = current_gusigun_votes["구시군KEY"].dropna().astype("string").unique().tolist() if "구시군KEY" in current_gusigun_votes.columns else []
    gusigun_focus_mode = region_level == "구시군" and len(focus_gusigun_keys) == 1 and "합계" not in focus_gusigun_keys[0]
    focus_target_key = focus_gusigun_keys[0] if gusigun_focus_mode else None
    focus_target_name = _first_text_value(current_gusigun_votes, "구시군명") if gusigun_focus_mode else None
    focus_sido_name = _first_text_value(current_gusigun_votes, "시도명") if gusigun_focus_mode else None
    focus_axis_labels = _build_focus_axis_labels(focus_target_name, focus_sido_name)
    adjacent_compare_entity_type = entity_type
    focus_adjacent_entity_name: str | None = None
    selected_adjacent_names: list[str] = []
    if gusigun_focus_mode:
        focus_adjacent_entity_name = _resolve_adjacent_compare_entity(current_gusigun_votes, adjacent_compare_entity_type)
    adjacent_compare_label_col = _label_column(adjacent_compare_entity_type)
    focus_context_df = pd.DataFrame(columns=["비교축", label_col, "득표수", "유효투표수", "득표율_계산"])
    focus_metric_context_df = pd.DataFrame(columns=["비교축", label_col, "득표수", "유효투표수", "득표율_계산"])
    focus_adjacent_region_df = pd.DataFrame(columns=["지역", label_col, "득표율_계산"])
    focus_download_df = region_share_df.copy()

    if gusigun_focus_mode and focus_target_key:
        national_selected = build_scope_selection(analysis_selected, "national")
        sido_selected = build_scope_selection(analysis_selected, "sido")
        current_national_votes = apply_common_filters(scoped_all_votes, national_selected)
        current_national_votes = current_national_votes.loc[current_national_votes["선거KEY"].astype("string") == current_election_key].copy()
        current_sido_votes = apply_common_filters(scoped_all_votes, sido_selected)
        current_sido_votes = current_sido_votes.loc[current_sido_votes["선거KEY"].astype("string") == current_election_key].copy()
        current_national_votes = filter_by_election_scope(current_national_votes, local_vote_scope)
        current_sido_votes = filter_by_election_scope(current_sido_votes, local_vote_scope)
        if focus_sido_name:
            current_sido_votes = current_sido_votes.loc[current_sido_votes["시도명"].astype("string") == str(focus_sido_name)].copy()
            if not analysis_selected.get("시도명"):
                region_share_df = region_share_df.loc[region_share_df["시도명"].astype("string") == str(focus_sido_name)].copy()
                distribution_df = distribution_df.loc[distribution_df["시도명"].astype("string") == str(focus_sido_name)].copy()
                competition_df = competition_df.loc[competition_df["시도명"].astype("string") == str(focus_sido_name)].copy()
                if "시도명" in scatter_df.columns:
                    scatter_df = scatter_df.loc[scatter_df["시도명"].astype("string") == str(focus_sido_name)].copy()
        selected_adjacent_names = get_selected_adjacent_gusigun_names(analysis_selected)
        if selected_adjacent_names:
            adjacent_votes = current_sido_votes.loc[
                current_sido_votes["구시군명"].astype("string").isin(selected_adjacent_names)
            ].copy()
        else:
            adjacent_keys = get_adjacent_gusigun_keys(focus_target_key)
            adjacent_votes = current_sido_votes.loc[current_sido_votes["구시군KEY"].astype("string").isin(adjacent_keys)].copy() if adjacent_keys else current_sido_votes.iloc[0:0].copy()

        target_axis = focus_axis_labels["target_axis"]
        national_axis = focus_axis_labels["national_axis"]
        sido_axis = focus_axis_labels["sido_axis"]
        adjacent_axis = focus_axis_labels["adjacent_axis"]
        focus_context_parts = [
            _build_context_share_frame(current_gusigun_votes, entity_type, label_col, selected_entities, target_axis),
            _build_context_share_frame(current_national_votes, entity_type, label_col, selected_entities, national_axis),
            _build_context_share_frame(current_sido_votes, entity_type, label_col, selected_entities, sido_axis),
        ]
        if not adjacent_votes.empty:
            focus_context_parts.append(
                _build_context_share_frame(adjacent_votes, entity_type, label_col, selected_entities, adjacent_axis)
            )
        focus_context_df = pd.concat(focus_context_parts, ignore_index=True)
        focus_axis_categories = [
            value
            for value in _focus_axis_category_order(focus_axis_labels)
            if value in focus_context_df["비교축"].astype("string").unique().tolist()
        ]
        focus_context_df["비교축"] = pd.Categorical(
            focus_context_df["비교축"],
            categories=focus_axis_categories,
            ordered=True,
        )
        focus_context_df = focus_context_df.sort_values(by=["비교축", "득표수"], ascending=[True, False], kind="stable").reset_index(drop=True)

        highlight_entity_names = [entity_name for _, entity_name in highlight_entities]
        focus_metric_parts = [
            _build_context_share_frame(current_gusigun_votes, entity_type, label_col, highlight_entity_names, target_axis),
            _build_context_share_frame(current_national_votes, entity_type, label_col, highlight_entity_names, national_axis),
            _build_context_share_frame(current_sido_votes, entity_type, label_col, highlight_entity_names, sido_axis),
        ]
        if not adjacent_votes.empty:
            focus_metric_parts.append(
                _build_context_share_frame(adjacent_votes, entity_type, label_col, highlight_entity_names, adjacent_axis)
            )
        focus_metric_context_df = pd.concat(focus_metric_parts, ignore_index=True)
        focus_metric_categories = [
            value
            for value in _focus_axis_category_order(focus_axis_labels)
            if value in focus_metric_context_df["비교축"].astype("string").unique().tolist()
        ]
        focus_metric_context_df["비교축"] = pd.Categorical(
            focus_metric_context_df["비교축"],
            categories=focus_metric_categories,
            ordered=True,
        )
        focus_metric_context_df = focus_metric_context_df.sort_values(by=["비교축", "득표수"], ascending=[True, False], kind="stable").reset_index(drop=True)

        province_region_df = calc_entity_share_by_region(current_sido_votes, entity_type=adjacent_compare_entity_type, level="구시군")
        compare_region_keys = [focus_target_key, *adjacent_votes["구시군KEY"].dropna().astype("string").unique().tolist()]
        focus_adjacent_region_df = province_region_df.loc[province_region_df["구시군KEY"].astype("string").isin(compare_region_keys)].copy()
        if not focus_adjacent_region_df.empty and focus_adjacent_entity_name:
            focus_adjacent_region_df = focus_adjacent_region_df.loc[
                focus_adjacent_region_df[adjacent_compare_label_col].astype("string") == str(focus_adjacent_entity_name)
            ].copy()
        if not focus_adjacent_region_df.empty:
            focus_adjacent_region_df["비교권역"] = focus_adjacent_region_df["구시군KEY"].astype("string").map(
                lambda value: "선택 구시군" if value == focus_target_key else "인접 구시군"
            )
            focus_adjacent_region_df["지역정렬"] = focus_adjacent_region_df["구시군KEY"].astype("string").map(
                lambda value: 0 if value == focus_target_key else 1
            )
            focus_adjacent_region_df = focus_adjacent_region_df.sort_values(
                by=["지역정렬", "득표수"],
                ascending=[True, False],
                kind="stable",
            ).reset_index(drop=True)
        focus_download_df = pd.concat(
            [
                focus_context_df.assign(비교유형="전국/시도/구시군"),
                focus_adjacent_region_df.assign(비교유형="인접 구시군"),
            ],
            ignore_index=True,
        )

    rowtype_messages: list[str] = []
    for rank_label, entity_name in highlight_entities:
        entity_rowtype_df = rowtype_entity_all_df.loc[rowtype_entity_all_df[label_col].astype("string") == str(entity_name)].copy()
        sentence = build_vote_rowtype_sentence(entity_rowtype_df, entity_name)
        if sentence:
            rowtype_messages.append(f"{rank_label}: {sentence}")
    if not rowtype_messages:
        rowtype_messages.append(
            build_vote_rowtype_sentence(
                rowtype_entity_all_df.loc[rowtype_entity_all_df[label_col].astype("string") == str(primary_entity)],
                primary_entity or entity_type_label,
            )
        )
    st.info("\n\n".join(rowtype_messages))
    if not competition_df.empty:
        st.caption(build_competitiveness_sentence(competition_df.head(1), f"{region_level} 경쟁도"))
    st.caption(competitiveness_help_text())

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("현재 선거KEY", current_election_key)
    col2.metric("총 유효투표수", format_int(summary["유효투표수"]))
    col3.metric("총 득표수", format_int(summary["득표수"]))
    col4.metric("1위", "-" if top_gap is None else str(top_gap["1위"]))
    col5.metric("2위", "-" if top_gap is None or pd.isna(top_gap["2위"]) else str(top_gap["2위"]))
    col6.metric("1-2위 격차", "-" if top_gap is None else format_percent(top_gap["득표율격차"]))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["비교 차트", "지역 분포", "경쟁도", "GIS 분석", "추이분석", "스윙 지역"])

    with tab1:
        if gusigun_focus_mode and not focus_context_df.empty:
            st.caption(
                f"구시군 현황 점검을 위해 `{focus_axis_labels['target_axis']} / {focus_axis_labels['national_axis']} / "
                f"{focus_axis_labels['sido_axis']} / {focus_axis_labels['adjacent_axis']}` 순서로 비교합니다."
            )
            for rank_label, entity_name in highlight_entities:
                _render_focus_metric_row(
                    focus_metric_context_df if not focus_metric_context_df.empty else focus_context_df,
                    label_col,
                    entity_name,
                    f"{rank_label} {entity_name}",
                    focus_axis_labels,
                )

            if chart_mode == "grouped":
                fig = grouped_bar_chart(
                    focus_context_df,
                    x_col="비교축",
                    y_col="득표율_계산",
                    color_col=label_col,
                    title=f"{entity_type_label} 득표율 기준 비교",
                    percent=True,
                    color_map=entity_color_map(focus_context_df, label_col),
                    legend_order=legend_entities,
                )
            elif chart_mode == "stacked":
                fig = stacked_bar_chart(
                    focus_context_df,
                    x_col="비교축",
                    y_col="득표수",
                    color_col=label_col,
                    title=f"{entity_type_label} 득표수 기준 비교",
                    percent=False,
                    color_map=entity_color_map(focus_context_df, label_col),
                    legend_order=legend_entities,
                )
            elif chart_mode == "100% stacked":
                fig = stacked_bar_chart(
                    focus_context_df,
                    x_col="비교축",
                    y_col="득표율_계산",
                    color_col=label_col,
                    title=f"{entity_type_label} 100% 득표율 비교",
                    percent=True,
                    color_map=entity_color_map(focus_context_df, label_col),
                    legend_order=legend_entities,
                )
            elif chart_mode == "heatmap":
                fig = heatmap_chart(
                    focus_context_df,
                    x_col="비교축",
                    y_col=label_col,
                    value_col="득표율_계산",
                    title=f"{entity_type_label} 득표율 비교 heatmap",
                    percent=True,
                )
            else:
                fig = lollipop_chart(
                    focus_context_df.loc[focus_context_df["비교축"].astype("string") == focus_axis_labels["target_axis"]],
                    category_col=label_col,
                    value_col="득표율_계산",
                    title=f"{focus_target_name} {entity_type_label} 득표율 랭킹",
                    percent=True,
                    color_col="정당명" if "정당명" in focus_context_df.columns else None,
                    color_map=entity_color_map(focus_context_df, label_col),
                )
        elif chart_mode == "grouped":
            fig = grouped_bar_chart(
                region_share_df,
                x_col="지역",
                y_col="득표율_계산",
                color_col=label_col,
                title=f"{region_level}별 {entity_type_label} 득표율 비교",
                percent=True,
                color_map=entity_color_map(region_share_df, label_col),
                legend_order=legend_entities,
            )
        elif chart_mode == "stacked":
            fig = stacked_bar_chart(
                region_share_df,
                x_col="지역",
                y_col="득표수",
                color_col=label_col,
                title=f"{region_level}별 {entity_type_label} 득표수 누적",
                percent=False,
                color_map=entity_color_map(region_share_df, label_col),
                legend_order=legend_entities,
            )
        elif chart_mode == "100% stacked":
            fig = stacked_bar_chart(
                region_share_df,
                x_col="지역",
                y_col="득표율_계산",
                color_col=label_col,
                title=f"{region_level}별 {entity_type_label} 100% 누적 비교",
                percent=True,
                color_map=entity_color_map(region_share_df, label_col),
                legend_order=legend_entities,
            )
        elif chart_mode == "heatmap":
            fig = heatmap_chart(region_share_df, x_col="지역", y_col=label_col, value_col="득표율_계산", title=f"{region_level}별 {entity_type_label} 득표율 heatmap", percent=True)
        else:
            overall_compare = current_entity_share.loc[current_entity_share[label_col].astype("string").isin(selected_entities)].copy()
            fig = lollipop_chart(
                overall_compare,
                category_col=label_col,
                value_col="득표율_계산",
                title=f"{entity_type_label} 득표율 랭킹",
                percent=True,
                color_col="정당명" if "정당명" in overall_compare.columns else None,
                color_map=entity_color_map(overall_compare, label_col),
            )
        st.plotly_chart(fig, use_container_width=True)

        if gusigun_focus_mode:
            if not focus_adjacent_region_df.empty and focus_adjacent_entity_name is not None:
                st.plotly_chart(
                    lollipop_chart(
                        focus_adjacent_region_df,
                        category_col="지역",
                        value_col="득표율_계산",
                        title=f"{_display_entity_type(adjacent_compare_entity_type)} {focus_adjacent_entity_name} 인접 구시군 득표율 비교",
                        percent=True,
                        color_col="비교권역",
                        color_map={"선택 구시군": "#0B2545", "인접 구시군": "#8D99AE"},
                    ),
                    use_container_width=True,
                )
                st.caption(
                    f"인접 비교 기준은 `선택 구시군 {_display_entity_type(adjacent_compare_entity_type)} {focus_adjacent_entity_name}`이며, "
                    "선택 구시군을 포함해 실제 경계를 맞댄 인접 구시군만 비교합니다."
                )
            else:
                st.info("인접 구시군 경계가 없거나 인접 비교 대상이 없어 전국/시도/선택 구시군 비교만 표시합니다.")

        st.caption("RowType 분해는 `전체 합계 / 관내사전투표 / 선거일투표(직접 선거일·투표소 합계 우선, 없으면 읍면동-관내사전투표) / 관외사전투표` 기준으로 표시합니다.")
        st.plotly_chart(
            grouped_bar_chart(
                rowtype_entity_df,
                x_col="RowType",
                y_col="득표율_계산",
                color_col=label_col,
                title=f"정규화된 RowType별 {entity_type_label} 득표율 분해",
                percent=True,
                color_map=entity_color_map(rowtype_entity_df, label_col),
                legend_order=legend_entities,
            ),
            use_container_width=True,
        )

    with tab2:
        if comparison_selected != analysis_selected:
            st.caption(f"{region_level} 비교는 선택된 상위 지역 범위를 유지하고 하위 지역 필터는 풀어서 비교합니다.")
        scatter_entity_name = str(primary_entity) if len(selected_entities) == 1 and primary_entity else f"선택 {entity_type_label}"
        scatter_chart_df = scatter_df.copy()
        if region_level == "읍면동":
            scatter_title = f"읍면동별 투표율과 {scatter_entity_name} 득표율 분포"
        elif region_level == "구시군" or gusigun_focus_mode:
            scatter_title = f"구시군별 투표율과 {scatter_entity_name} 득표율 분포"
        else:
            scatter_title = f"{region_level}별 투표율과 {scatter_entity_name} 득표율 분포"

        scatter_label_col = "__scatter_label__"
        scatter_chart_df[scatter_label_col] = ""
        label_source_col: str | None = None
        label_mask = pd.Series(False, index=scatter_chart_df.index)
        if gusigun_focus_mode and focus_target_name and "구시군명" in scatter_chart_df.columns:
            label_source_col = "구시군명"
            highlight_names = {str(focus_target_name), *(str(name) for name in selected_adjacent_names if name)}
            label_mask = scatter_chart_df["구시군명"].astype("string").isin(highlight_names)
        elif region_level == "읍면동":
            if "읍면동명" in scatter_chart_df.columns:
                label_source_col = "읍면동명"
            elif "지역" in scatter_chart_df.columns:
                label_source_col = "지역"
            if label_source_col:
                label_mask = scatter_chart_df[label_source_col].notna()

        if label_source_col:
            scatter_chart_df.loc[label_mask, scatter_label_col] = (
                scatter_chart_df.loc[label_mask, label_source_col].astype("string").fillna("")
            )

        scatter_fig = scatter_chart(
            scatter_chart_df,
            x_col="투표율",
            y_col="득표율_계산",
            title=scatter_title,
            size_col="투표수" if "투표수" in scatter_chart_df.columns else None,
            color_col=label_col if label_col in scatter_chart_df.columns else None,
            color_map=entity_color_map(scatter_chart_df, label_col),
            legend_order=legend_entities,
            text_col=scatter_label_col,
            text_template="%{text}",
            text_position="top center",
            percent_x=True,
            percent_y=True,
        )
        trace_names = [str(getattr(trace, "name", "")) for trace in scatter_fig.data if getattr(trace, "name", None) not in (None, "")]
        active_legend_name = str(primary_entity) if primary_entity and str(primary_entity) in trace_names else (trace_names[0] if trace_names else None)
        if active_legend_name and len(trace_names) > 1:
            for trace in scatter_fig.data:
                trace_name = str(getattr(trace, "name", ""))
                if not trace_name:
                    continue
                trace.visible = True if trace_name == active_legend_name else "legendonly"
        st.plotly_chart(scatter_fig, use_container_width=True)
        st.caption(
            _scatter_point_description(
                region_level,
                scatter_entity_name,
                gusigun_focus_mode,
                focus_target_name,
                selected_adjacent_names,
            )
        )
        st.caption("거품 크기는 투표수이며, 클수록 해당 지역의 투표수가 많습니다.")
        st.plotly_chart(
            distribution_chart(
                distribution_df,
                category_col=label_col,
                value_col="득표율_계산",
                title=f"{region_level}별 {entity_type_label} 득표율 분포",
                chart_type=distribution_mode,
                color_col=label_col,
                color_map=entity_color_map(distribution_df, label_col),
                legend_order=legend_entities,
                percent=True,
            ),
            use_container_width=True,
        )

    with tab3:
        if comparison_selected != analysis_selected:
            st.caption(f"{region_level} 경쟁도 비교는 선택된 상위 지역 범위를 유지하고 하위 지역 필터는 풀어서 비교합니다.")
        competition_chart = lollipop_chart(
            competition_df,
            category_col="지역",
            value_col="경쟁도지수",
            title=f"{region_level}별 경쟁도 랭킹",
            percent=True,
        )
        selection = st.plotly_chart(
            competition_chart,
            use_container_width=True,
            key="votes_competition_chart",
            on_select="rerun",
        )
        selected_region = _extract_selection_value(selection, ("y",))
        if selected_region:
            st.caption(f"선택된 지역 drill-down: {selected_region}")
            st.dataframe(
                region_share_df.loc[region_share_df["지역"].astype("string") == selected_region],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(competition_df.head(40), use_container_width=True, hide_index=True)

    with tab4:
        st.caption("득표 GIS는 후보/정당별 득표수·득표율과 후보/정당 간 격차를 구시군 또는 읍면동 경계로 표시합니다.")
        gis_control_col1, gis_control_col2, gis_control_col3 = st.columns(3)
        map_level_options = ["구시군", "읍면동"]
        default_map_level = region_level if region_level in map_level_options else "구시군"
        with gis_control_col1:
            map_level = st.selectbox(
                "지도 단위",
                map_level_options,
                index=map_level_options.index(default_map_level),
                key="votes_gis_level",
            )
        with gis_control_col2:
            if st.session_state.get("votes_gis_metric") not in VOTE_GIS_METRIC_OPTIONS:
                st.session_state["votes_gis_metric"] = next(iter(VOTE_GIS_METRIC_OPTIONS))
            vote_gis_metric_label = st.selectbox(
                "지도 지표",
                list(VOTE_GIS_METRIC_OPTIONS),
                key="votes_gis_metric",
            )
        with gis_control_col3:
            basemap_labels = list(BASEMAP_STYLE_OPTIONS.keys())
            basemap_option = st.selectbox(
                "베이스맵",
                basemap_labels,
                index=basemap_labels.index("배경 없음") if "배경 없음" in basemap_labels else 0,
                key="votes_gis_basemap",
            )

        vote_gis_metric_name, vote_gis_entity_type = VOTE_GIS_METRIC_OPTIONS[vote_gis_metric_label]
        vote_gis_pair_entity_type = VOTE_GIS_PAIR_GAP_METRICS.get(vote_gis_metric_name)
        vote_gis_entity_name = None
        vote_gis_compare_entity_name = None
        if vote_gis_pair_entity_type:
            vote_gis_entity_options = _vote_gis_entity_options(current_votes, vote_gis_pair_entity_type)
            if len(vote_gis_entity_options) < 2:
                st.warning("후보/정당 간 격차를 계산하려면 비교 대상이 2개 이상 필요합니다.")
                st.stop()
            preferred_base = (
                str(primary_entity)
                if vote_gis_pair_entity_type == entity_type and primary_entity and str(primary_entity) in vote_gis_entity_options
                else vote_gis_entity_options[0]
            )
            preferred_compare = next((name for name in vote_gis_entity_options if name != preferred_base), vote_gis_entity_options[1])
            pair_col1, pair_col2 = st.columns(2)
            base_key = f"votes_gis_gap_base_{vote_gis_pair_entity_type}"
            if st.session_state.get(base_key) not in vote_gis_entity_options:
                st.session_state[base_key] = preferred_base
            with pair_col1:
                vote_gis_entity_name = st.selectbox(
                    "격차 기준 대상",
                    options=vote_gis_entity_options,
                    index=vote_gis_entity_options.index(preferred_base),
                    key=base_key,
                )
            compare_options = [name for name in vote_gis_entity_options if name != vote_gis_entity_name]
            preferred_compare = preferred_compare if preferred_compare in compare_options else compare_options[0]
            compare_key = f"votes_gis_gap_compare_{vote_gis_pair_entity_type}"
            if st.session_state.get(compare_key) not in compare_options:
                st.session_state[compare_key] = preferred_compare
            with pair_col2:
                vote_gis_compare_entity_name = st.selectbox(
                    "격차 비교 대상",
                    options=compare_options,
                    index=compare_options.index(preferred_compare),
                    key=compare_key,
                )
        elif vote_gis_entity_type:
            vote_gis_label_col = _label_column(vote_gis_entity_type)
            vote_gis_entity_options = _vote_gis_entity_options(current_votes, vote_gis_entity_type)
            if not vote_gis_entity_options:
                st.warning("선택한 선거에는 지도에 표시할 득표 대상이 없습니다.")
                st.stop()
            preferred_entity = (
                str(primary_entity)
                if vote_gis_entity_type == entity_type and primary_entity and str(primary_entity) in vote_gis_entity_options
                else vote_gis_entity_options[0]
            )
            entity_key = f"votes_gis_entity_{vote_gis_entity_type}"
            if st.session_state.get(entity_key) not in vote_gis_entity_options:
                st.session_state[entity_key] = preferred_entity
            vote_gis_entity_name = st.selectbox(
                "지도 대상",
                options=vote_gis_entity_options,
                index=vote_gis_entity_options.index(preferred_entity),
                key=entity_key,
            )

        map_selected = build_region_comparison_selection(analysis_selected, map_level)
        map_votes = apply_common_filters(scoped_all_votes, map_selected)
        map_turnout = apply_common_filters(scoped_all_turnout, map_selected)
        inferred_map_scope = _infer_vote_gis_upper_scope(current_votes, analysis_selected, map_level)
        if inferred_map_scope:
            map_votes = _apply_vote_gis_upper_scope(map_votes, inferred_map_scope)
            map_turnout = _apply_vote_gis_upper_scope(map_turnout, inferred_map_scope)
        map_votes = map_votes.loc[map_votes["선거KEY"].astype("string") == current_election_key].copy()
        map_turnout = map_turnout.loc[map_turnout["선거KEY"].astype("string") == current_election_key].copy()
        map_votes = filter_by_election_scope(map_votes, local_vote_scope)
        map_turnout = filter_by_election_scope(map_turnout, local_vote_scope)

        vote_gis_df = calc_map_metric_by_region(
            map_turnout,
            map_votes,
            level=map_level,
            metric_name=vote_gis_metric_name,
            entity_type=vote_gis_entity_type,
            entity_name=vote_gis_entity_name,
            compare_entity_name=vote_gis_compare_entity_name,
        )
        geometry_context = load_geometry_context(map_level)
        st.info(geometry_context["message"])
        if map_selected != analysis_selected:
            st.caption(f"{map_level} 지도는 선택된 상위 지역 범위를 유지하고 하위 지역 필터는 풀어서 표시합니다.")
        if vote_gis_pair_entity_type:
            st.caption(
                f"격차는 `{vote_gis_entity_name} - {vote_gis_compare_entity_name}`로 계산합니다. "
                f"양수는 {vote_gis_entity_name} 우세, 음수는 {vote_gis_compare_entity_name} 우세입니다."
            )
        elif vote_gis_metric_name in {"득표수격차", "득표율격차"}:
            st.caption("격차 지도는 부호형으로 표시합니다. 민주계 우세 지역은 파란색, 국민의힘계 우세 지역은 빨간색이며, 기타 정당 우세 지역은 중립색에 가깝게 보입니다.")

        key_col = "읍면동KEY" if map_level == "읍면동" else "구시군KEY"
        vote_gis_percent = vote_gis_metric_name in {"후보 득표율", "정당 득표율", "득표율격차", "후보 간 득표율 격차", "정당 간 득표율 격차"}
        if vote_gis_pair_entity_type:
            map_target_suffix = f": {vote_gis_entity_name} - {vote_gis_compare_entity_name}"
        else:
            map_target_suffix = f" - {vote_gis_entity_name}" if vote_gis_entity_name else ""
        if vote_gis_df.empty:
            st.warning("현재 조건에 맞는 득표 지도 데이터가 없습니다.")
        else:
            vote_gis_label_col = _vote_gis_label_col(vote_gis_df, map_level)
            vote_gis_entity_label_col = _label_column(vote_gis_entity_type) if vote_gis_entity_type else None
            vote_gis_colorscale = (
                _vote_gis_pair_colorscale(current_votes, vote_gis_pair_entity_type, vote_gis_entity_name, vote_gis_compare_entity_name)
                if vote_gis_pair_entity_type
                else _vote_gis_colorscale(vote_gis_df, vote_gis_entity_type, vote_gis_entity_label_col, vote_gis_entity_name)
            )
            vote_gis_fig = build_region_choropleth(
                vote_gis_df,
                level=map_level,
                key_col=key_col,
                value_col="지도지표",
                title=f"{_election_title_label(current_election_key, filter_options)} {map_level} {vote_gis_metric_label}{map_target_suffix}",
                geometry_context=geometry_context,
                percent=vote_gis_percent,
                basemap_style=basemap_option,
                label_col=vote_gis_label_col,
                colorscale=vote_gis_colorscale,
            )
            st.plotly_chart(vote_gis_fig, use_container_width=True)
            display_cols = [
                column
                for column in [
                    "지역",
                    "정당명",
                    "후보명",
                    "득표수",
                    "득표율_계산",
                    "유효투표수",
                    "기준대상",
                    "비교대상",
                    "기준정당",
                    "비교정당",
                    "기준득표수",
                    "비교득표수",
                    "기준득표율",
                    "비교득표율",
                    "1위후보",
                    "1위정당",
                    "1위득표수",
                    "1위득표율",
                    "2위후보",
                    "2위정당",
                    "2위득표수",
                    "2위득표율",
                    "득표수격차",
                    "득표율격차",
                    "격차방향",
                    "지도지표",
                ]
                if column in vote_gis_df.columns
            ]
            st.dataframe(vote_gis_df.loc[:, display_cols].head(200), use_container_width=True, hide_index=True)
            st.download_button(
                "득표 GIS 데이터 CSV 다운로드",
                dataframe_to_csv_bytes(vote_gis_df),
                file_name="vote_gis_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab5:
        st.caption("득표 추이분석은 이 페이지 내부에서 관리합니다.")
        trend_entity_type = st.selectbox(
            "득표 추이 기준",
            TREND_ENTITY_TYPE_OPTIONS,
            index=TREND_ENTITY_TYPE_OPTIONS.index("구분"),
            key="votes_trend_entity_type",
            format_func=_display_entity_type,
        )
        trend_entity_type_label = _display_entity_type(trend_entity_type)
        trend_label_col = _label_column(trend_entity_type)
        entity_trend_df = calc_entity_trend(scoped_filtered_votes, entity_type=trend_entity_type)
        entity_trend_df = _apply_trend_group_display_labels(entity_trend_df, trend_entity_type, trend_label_col)
        entity_trend_df = _attach_trend_party_membership(entity_trend_df, scoped_filtered_votes, trend_entity_type, trend_label_col)
        trend_entity_options = order_group_legend_values(entity_trend_df[trend_label_col], entity_type=trend_entity_type)
        trend_default_entities = trend_entity_options[: min(5, len(trend_entity_options))]
        if trend_entity_type in TREND_GROUP_ENTITY_TYPES:
            preferred_entities = [
                entity
                for entity in trend_entity_options
                if normalize_group_legend_label(entity, entity_type=trend_entity_type) in GROUP_TREND_HIGHLIGHT
            ]
            if preferred_entities:
                trend_default_entities = preferred_entities
        _sanitize_multiselect_state("votes_trend_selected_entities", trend_entity_options, trend_default_entities)
        trend_selected_entities = st.multiselect(
            f"{trend_entity_type_label} 추이 대상",
            options=trend_entity_options,
            default=trend_default_entities,
            key="votes_trend_selected_entities",
        )
        entity_trend_pivot_source_df = entity_trend_df.copy()
        trend_pivot_order = [entity for entity in trend_entity_options if entity in entity_trend_pivot_source_df[trend_label_col].astype("string").unique().tolist()]
        if trend_pivot_order:
            entity_trend_pivot_source_df[trend_label_col] = pd.Categorical(entity_trend_pivot_source_df[trend_label_col], categories=trend_pivot_order, ordered=True)
            entity_trend_pivot_source_df = entity_trend_pivot_source_df.sort_values(by=[trend_label_col, "선거시점", "선거KEY"], kind="stable").reset_index(drop=True)
        if trend_selected_entities:
            entity_trend_df = entity_trend_df.loc[entity_trend_df[trend_label_col].astype("string").isin(trend_selected_entities)].copy()
        trend_legend_order = [entity for entity in trend_entity_options if entity in entity_trend_df[trend_label_col].astype("string").unique().tolist()]
        if trend_legend_order:
            entity_trend_df[trend_label_col] = pd.Categorical(entity_trend_df[trend_label_col], categories=trend_legend_order, ordered=True)
            entity_trend_df = entity_trend_df.sort_values(by=[trend_label_col, "선거시점", "선거KEY"], kind="stable").reset_index(drop=True)
        entity_trend_pivot_df = _build_entity_trend_pivot_frame(entity_trend_pivot_source_df, trend_label_col, trend_pivot_order)
        entity_trend_pivot_note_df = _build_entity_trend_pivot_note_frame(entity_trend_pivot_source_df, trend_label_col, trend_pivot_order)
        entity_trend_top_winner_df = _build_entity_trend_top_winner_frame(scoped_filtered_votes)
        trendline_entities = []
        if trend_entity_type in TREND_GROUP_ENTITY_TYPES:
            trendline_entities = [
                entity
                for entity in trend_legend_order
                if normalize_group_legend_label(entity, entity_type=trend_entity_type) in GROUP_TREND_HIGHLIGHT
            ]
        entity_trend_rank = entity_trend_df.sort_values(by=["선거KEY", "득표수"], ascending=[True, False], kind="stable").copy()
        entity_trend_rank["순위"] = entity_trend_rank.groupby("선거KEY", observed=True)["득표수"].rank(method="dense", ascending=False).astype("Int64")
        if trend_legend_order:
            entity_trend_rank[trend_label_col] = pd.Categorical(entity_trend_rank[trend_label_col], categories=trend_legend_order, ordered=True)
        st.plotly_chart(
            line_metric_chart(
                entity_trend_df,
                x_col="선거라벨" if "선거라벨" in entity_trend_df.columns else "선거KEY",
                y_col="득표율_계산",
                color_col=trend_label_col,
                title="선거별 득표율 추이",
                percent=True,
                show_labels=False,
                color_map=entity_color_map(entity_trend_df, trend_label_col),
                legend_order=trend_legend_order,
                trendline_entities=trendline_entities,
                hover_extra_cols=[TREND_PARTY_HOVER_COL] if TREND_PARTY_HOVER_COL in entity_trend_df.columns else None,
                hover_extra_labels={TREND_PARTY_HOVER_COL: "포함 정당"},
            ),
            use_container_width=True,
        )
        if trendline_entities:
            st.caption("민주당계, 국민의힘계, 제3지대의 점선은 선거시점 기준 선형 추세선입니다.")
        trend_party_summary = _build_trend_party_summary(scoped_filtered_votes, trend_entity_type, trend_label_col, trend_legend_order)
        if trend_party_summary:
            st.caption("선택 범위에서 각 계열 득표율은 아래 정당 득표를 합산한 값입니다.")
            st.markdown("\n".join(f"- **{label}**: {parties}" for label, parties in trend_party_summary))
        st.plotly_chart(
            grouped_bar_chart(
                entity_trend_df,
                x_col="선거라벨" if "선거라벨" in entity_trend_df.columns else "선거KEY",
                y_col="득표율_계산",
                color_col=trend_label_col,
                title="선거별 득표율 피벗 차트",
                percent=True,
                color_map=entity_color_map(entity_trend_df, trend_label_col),
                legend_order=trend_legend_order,
            ),
            use_container_width=True,
        )
        if not entity_trend_pivot_df.empty:
            st.subheader("득표율 피벗 테이블")
            st.caption("행은 선거, 열은 선택한 계열이며 값은 득표율입니다.")
            st.dataframe(
                _build_entity_trend_pivot_display_frame(entity_trend_pivot_df, entity_trend_pivot_note_df, entity_trend_top_winner_df),
                use_container_width=True,
                hide_index=True,
            )
        st.plotly_chart(
            bump_chart(
                entity_trend_rank,
                entity_col=trend_label_col,
                x_col="선거라벨" if "선거라벨" in entity_trend_rank.columns else "선거KEY",
                rank_col="순위",
                title="선거별 순위 변화",
                color_map=entity_color_map(entity_trend_rank, trend_label_col),
            ),
            use_container_width=True,
        )

    with tab6:
        st.caption("스윙 지역은 선택된 전체 선거 추이를 기준으로 승리 횟수와 평균 득표율 격차를 각각 정적 5구간으로 나눠 표시합니다.")
        swing_level = _swing_region_level(analysis_selected)
        swing_key_col = "읍면동KEY" if swing_level == "읍면동" else "구시군KEY"
        swing_selected = build_region_comparison_selection(_swing_vote_selection(analysis_selected), swing_level)
        swing_votes = apply_common_filters(app_data["votes"], swing_selected)
        swing_summary_df, swing_win_counts_df, swing_avg_share_df, swing_win_order = _build_swing_frames(swing_votes, swing_level)
        swing_vote_pivot_df = _build_swing_vote_pivot_table(swing_votes, swing_level)
        swing_rowtype_pivot_tables: list[tuple[str, pd.DataFrame, str]] = []
        if swing_level == "읍면동":
            for spec in SWING_ROWTYPE_PIVOT_SPECS:
                base_label = str(spec["label"]).removesuffix(" 소계")
                file_slug = "local_early" if spec.get("key") == SWING_LOCAL_EARLY_VOTE_KEY else "election_day"
                swing_rowtype_pivot_tables.append(
                    (
                        f"{base_label} 기준 읍면동별 피벗 테이블",
                        _build_swing_rowtype_vote_pivot_table(swing_votes, spec),
                        f"vote_swing_{file_slug}_emd_pivot.csv",
                    )
                )
        swing_geometry_context = load_geometry_context(swing_level)
        st.info(swing_geometry_context["message"])
        if not swing_summary_df.empty:
            swing_summary_df = _filter_to_geometry_locations(swing_summary_df, swing_geometry_context, swing_key_col)
        if not swing_win_counts_df.empty and swing_key_col in swing_win_counts_df.columns:
            swing_win_counts_df = _filter_to_geometry_locations(swing_win_counts_df, swing_geometry_context, swing_key_col)
        if not swing_avg_share_df.empty and swing_key_col in swing_avg_share_df.columns:
            swing_avg_share_df = _filter_to_geometry_locations(swing_avg_share_df, swing_geometry_context, swing_key_col)
        if not swing_vote_pivot_df.empty and swing_key_col in swing_vote_pivot_df.columns:
            swing_vote_pivot_df = _filter_swing_vote_pivot_to_geometry_locations(swing_vote_pivot_df, swing_geometry_context, swing_key_col)
        swing_rowtype_pivot_tables = [
            (
                title,
                _filter_to_geometry_locations(pivot_df, swing_geometry_context, swing_key_col)
                if not pivot_df.empty and swing_key_col in pivot_df.columns
                else pivot_df,
                file_name,
            )
            for title, pivot_df, file_name in swing_rowtype_pivot_tables
        ]

        if swing_summary_df.empty:
            st.warning("현재 조건에 맞는 스윙 지역 데이터가 없습니다. 시도 또는 구시군 필터와 선거 범위를 확인해 주세요.")
        else:
            win_swing_map_fig = build_region_choropleth(
                swing_summary_df,
                level=swing_level,
                key_col=swing_key_col,
                value_col="지도지표_승리횟수",
                title=f"{swing_level}별 승리 횟수 기준 스윙 지역 (민주당계 vs 국힘계)",
                geometry_context=swing_geometry_context,
                percent=False,
                basemap_style="배경 없음",
                label_col="승리횟수지도라벨",
                colorscale=SWING_COLOR_SCALE,
                show_value_label=False,
            )
            if win_swing_map_fig.data:
                win_swing_map_fig.data[0].zmin = 1
                win_swing_map_fig.data[0].zmax = 5
                win_swing_map_fig.data[0].colorbar.tickmode = "array"
                win_swing_map_fig.data[0].colorbar.tickvals = list(SWING_CATEGORY_SCORES.values())
                win_swing_map_fig.data[0].colorbar.ticktext = SWING_CATEGORY_ORDER
            st.plotly_chart(win_swing_map_fig, use_container_width=True)
            st.caption("승리 횟수 기준은 `(민주당계 승리횟수 - 국민의힘계 승리횟수) / 양대 정당계 승리횟수`를 -1~+1 범위의 정적 5구간으로 나눕니다.")

            avg_gap_swing_map_fig = build_region_choropleth(
                swing_summary_df,
                level=swing_level,
                key_col=swing_key_col,
                value_col="지도지표_평균격차",
                title=f"{swing_level}별 평균 득표율 격차 기준 스윙 지역 (민주당계 vs 국힘계)",
                geometry_context=swing_geometry_context,
                percent=False,
                basemap_style="배경 없음",
                label_col="평균득표율지도라벨",
                colorscale=SWING_COLOR_SCALE,
                show_value_label=False,
            )
            if avg_gap_swing_map_fig.data:
                avg_gap_swing_map_fig.data[0].zmin = 1
                avg_gap_swing_map_fig.data[0].zmax = 5
                avg_gap_swing_map_fig.data[0].colorbar.tickmode = "array"
                avg_gap_swing_map_fig.data[0].colorbar.tickvals = list(SWING_CATEGORY_SCORES.values())
                avg_gap_swing_map_fig.data[0].colorbar.ticktext = SWING_CATEGORY_ORDER
            st.plotly_chart(avg_gap_swing_map_fig, use_container_width=True)
            st.caption(
                "평균 득표율 격차 기준은 `민주당계 - 국민의힘계` 평균 격차를 정적 구간으로 나눕니다. "
                "-10%p 이하는 국민의힘계 우세, -3~+3%p는 경합지역, +10%p 이상은 민주당계 우세입니다."
            )

            st.plotly_chart(_swing_win_count_chart(swing_win_counts_df, swing_win_order), use_container_width=True)
            st.plotly_chart(_swing_avg_share_chart(swing_avg_share_df, swing_win_order), use_container_width=True)
            if not swing_vote_pivot_df.empty:
                st.subheader("선거별 득표율·격차 피벗 테이블")
                st.caption("득표수·득표율 격차는 `민주당계 - 국민의힘계` 기준입니다.")
                st.dataframe(
                    _swing_vote_pivot_display_frame(swing_vote_pivot_df, swing_level),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "스윙 득표율 피벗 CSV 다운로드",
                    dataframe_to_csv_bytes(_swing_vote_pivot_display_frame(swing_vote_pivot_df, swing_level)),
                    file_name="vote_swing_share_pivot.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            for title, rowtype_pivot_df, file_name in swing_rowtype_pivot_tables:
                if rowtype_pivot_df.empty:
                    continue
                st.subheader(title)
                st.caption("득표수·득표율 격차는 `민주당계 - 국민의힘계` 기준입니다.")
                st.dataframe(
                    _swing_vote_pivot_display_frame(rowtype_pivot_df, "읍면동"),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    f"{title} CSV 다운로드",
                    dataframe_to_csv_bytes(_swing_vote_pivot_display_frame(rowtype_pivot_df, "읍면동")),
                    file_name=file_name,
                    mime="text/csv",
                    use_container_width=True,
                )
            display_cols = [
                column
                for column in [
                    "지역",
                    "승리횟수스윙분류",
                    "평균격차스윙분류",
                    "민주당계 승리횟수",
                    "국민의힘계 승리횟수",
                    "제3지대 승리횟수",
                    "무소속/기타 승리횟수",
                    "민주당계 평균득표율",
                    "국민의힘계 평균득표율",
                    "승리횟수격차",
                    "승리균형지수",
                    "평균격차",
                    "최근격차",
                    "최저격차",
                    "최고격차",
                    "선거수",
                    "시도명",
                    "구시군명",
                    "일반구명",
                    "읍면동명",
                ]
                if column in swing_summary_df.columns
            ]
            swing_sort_cols = [
                column
                for column in ["승리횟수스윙점수", "평균격차스윙점수", "평균격차", "지역"]
                if column in swing_summary_df.columns
            ]
            swing_sort_ascending = [False, False, False, True][: len(swing_sort_cols)]
            st.dataframe(
                swing_summary_df.sort_values(by=swing_sort_cols, ascending=swing_sort_ascending, kind="stable").loc[:, display_cols],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "스윙 지역 CSV 다운로드",
                dataframe_to_csv_bytes(swing_summary_df),
                file_name="vote_swing_regions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        st.download_button(
            "지역 비교 CSV 다운로드",
            dataframe_to_csv_bytes(focus_download_df if gusigun_focus_mode else region_share_df),
            file_name="vote_region_compare.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "추이 CSV 다운로드",
            dataframe_to_csv_bytes(entity_trend_pivot_df),
            file_name="vote_trend_pivot.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col3:
        st.download_button(
            "경쟁도 CSV 다운로드",
            dataframe_to_csv_bytes(competition_df),
            file_name="vote_competitiveness.csv",
            mime="text/csv",
            use_container_width=True,
        )


main()
