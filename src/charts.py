from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PARTY_COLOR_OVERRIDES = {
    "더불어민주당": "#1F5AA6",
    "민주통합당": "#1F5AA6",
    "민주당": "#1F5AA6",
    "민주": "#1F5AA6",
    "민주계": "#1F5AA6",
    "민주당계": "#1F5AA6",
    "이재명": "#1F5AA6",
    "새정치민주연합": "#2B7BC1",
    "국민의당": "#2A9D8F",
    "국민의힘": "#D94C3D",
    "국힘": "#D94C3D",
    "국민의힘계": "#D94C3D",
    "김문수": "#D94C3D",
    "새누리당": "#D94C3D",
    "자유한국당": "#C63E32",
    "미래통합당": "#C63E32",
    "정의당": "#F2B90C",
    "조국혁신당": "#12736B",
    "개혁신당": "#F28C28",
    "진보당": "#C5203E",
    "무소속": "#7B8B97",
    "좌파": "#C5203E",
    "극좌": "#8E244D",
    "중도좌파": "#2B7BC1",
    "중도우파": "#F28C28",
    "우파": "#D94C3D",
    "극우": "#7A1E2C",
    "제3지대": "#F28C28",
    "진보": "#7CC7F2",
    "보수": "#F2A6B3",
    "기타": "#AAB2BF",
    "기타 진보": "#7CC7F2",
    "기타 보수": "#F2A6B3",
    "중도/기타": "#9AA3AF",
    "무소속/기타": "#7B8B97",
}
ROWTYPE_COLOR_MAP = {
    "합계": "#495057",
    "관내사전투표": "#2A9D8F",
    "관외사전투표": "#0B6E69",
    "선거일투표": "#E76F51",
    "투표소": "#8D99AE",
    "읍면동": "#457B9D",
    "오류투표": "#B56576",
}
FALLBACK_PALETTE = ["#1F5AA6", "#D94C3D", "#2A9D8F", "#F4A261", "#6D597A", "#457B9D", "#8D99AE", "#B56576"]
HARMONIOUS_PALETTE = ["#4E79A7", "#76B7B2", "#E15759", "#F28E2B", "#59A14F", "#EDC948", "#B07AA1", "#9C755F", "#BAB0AC"]
HEATMAP_SCALE = [[0, "#F7FBFF"], [0.35, "#A6CEE3"], [0.6, "#4C78A8"], [0.85, "#1D4E89"], [1, "#0B2545"]]
ELECTION_AXIS_COLUMNS = {"선거KEY", "선거라벨", "선거축라벨"}
GROUP_LEGEND_ORDER = ["민주당", "국민의힘", "제3지대", "진보", "보수", "중도/기타", "무소속"]
GROUP_LEGEND_ORDER_MAP = {
    "구분": ["민주당", "국민의힘", "제3지대", "무소속", "기타"],
    "구분2": ["민주당", "국민의힘", "제3지대", "기타 진보", "기타 보수", "중도/기타", "무소속"],
}
GROUP_LEGEND_ALIASES = {
    "민주당": ("민주당", "더불어민주", "민주통합", "새정치민주"),
    "국민의힘": ("국민의힘", "국민의미래", "새누리당", "자유한국당", "미래통합당"),
    "제3지대": ("제3지대", "국민의당", "조국혁신당", "개혁신당"),
    "진보": ("진보", "진보당", "정의당"),
    "보수": ("보수",),
    "중도/기타": ("중도/기타",),
    "무소속": ("무소속",),
}
GROUP_LABEL_SPECIAL_VALUES = {"기타", "기타 진보", "기타 보수"}
GROUP1_COLLAPSE_LABELS = {"진보", "보수", "중도/기타"}
GROUP2_RENAME_MAP = {"진보": "기타 진보", "보수": "기타 보수"}
GROUP_TREND_HIGHLIGHT = ["민주당", "국민의힘", "제3지대"]
BASE_CHART_FONT_SIZE = 17
BASE_CHART_TITLE_SIZE = 21
BASE_CHART_TEXT_SIZE = 17
VERTICAL_BAR_LABEL_TEXT_SIZE = BASE_CHART_TEXT_SIZE + 5
BASE_EMPTY_ANNOTATION_SIZE = 21


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": BASE_EMPTY_ANNOTATION_SIZE})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _apply_common_layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(
        title=title,
        title_font=dict(size=BASE_CHART_TITLE_SIZE),
        font=dict(size=BASE_CHART_FONT_SIZE),
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
        legend=dict(font=dict(size=BASE_CHART_FONT_SIZE)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(size=BASE_CHART_FONT_SIZE), tickfont=dict(size=BASE_CHART_FONT_SIZE))
    fig.update_yaxes(
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
        title_font=dict(size=BASE_CHART_FONT_SIZE),
        tickfont=dict(size=BASE_CHART_FONT_SIZE),
    )
    return fig


def _build_color_map(values: Sequence[str] | pd.Series | None, overrides: dict[str, str] | None = None) -> dict[str, str]:
    if values is None:
        return {}
    ordered_values = [str(value) for value in pd.Series(values).dropna().astype("string").unique().tolist()]
    color_map = dict(overrides or {})
    palette_index = 0
    for value in ordered_values:
        if value in color_map:
            continue
        color_map[value] = FALLBACK_PALETTE[palette_index % len(FALLBACK_PALETTE)]
        palette_index += 1
    return color_map


def _build_palette_map(values: Sequence[str] | pd.Series | None, palette: Sequence[str]) -> dict[str, str]:
    if values is None:
        return {}
    ordered_values = [str(value) for value in pd.Series(values).dropna().astype("string").unique().tolist()]
    return {value: palette[idx % len(palette)] for idx, value in enumerate(ordered_values)}


def _chart_single_color(title: str) -> str:
    if not title:
        return HARMONIOUS_PALETTE[0]
    return HARMONIOUS_PALETTE[sum(ord(char) for char in title) % len(HARMONIOUS_PALETTE)]


def _bar_label_text_size(orientation: str = "v") -> int:
    return VERTICAL_BAR_LABEL_TEXT_SIZE if orientation == "v" else BASE_CHART_TEXT_SIZE


def party_color_map(values: Sequence[str] | pd.Series | None) -> dict[str, str]:
    return _build_color_map(values, PARTY_COLOR_OVERRIDES)


def rowtype_color_map(values: Sequence[str] | pd.Series | None) -> dict[str, str]:
    return _build_color_map(values, ROWTYPE_COLOR_MAP)


def normalize_group_legend_label(value: object, entity_type: str | None = None) -> str:
    if pd.isna(value):
        return ""
    label = re.sub(r"^\s*\d+\.\s*", "", str(value)).strip()
    if label in GROUP_LABEL_SPECIAL_VALUES:
        return label

    normalized_label = label
    for canonical, aliases in GROUP_LEGEND_ALIASES.items():
        if any(alias in label for alias in aliases):
            normalized_label = canonical
            break

    if entity_type == "구분" and normalized_label in GROUP1_COLLAPSE_LABELS:
        return "기타"
    if entity_type == "구분2":
        return GROUP2_RENAME_MAP.get(normalized_label, normalized_label)
    return normalized_label


def order_group_legend_values(values: Sequence[str] | pd.Series | None, entity_type: str | None = None) -> list[str]:
    if values is None:
        return []

    unique_values = [str(value) for value in pd.Series(values).dropna().astype("string").unique().tolist()]
    original_order = {value: idx for idx, value in enumerate(unique_values)}
    legend_order = GROUP_LEGEND_ORDER_MAP.get(entity_type, GROUP_LEGEND_ORDER)

    def _sort_key(value: str) -> tuple[int, int, str]:
        normalized = normalize_group_legend_label(value, entity_type=entity_type)
        try:
            rank = legend_order.index(normalized)
        except ValueError:
            rank = len(legend_order)
        return (rank, original_order.get(value, len(unique_values)), value)

    return sorted(unique_values, key=_sort_key)


def format_data_label(value: object, percent: bool = False) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1%}" if percent else f"{int(round(float(value))):,}"


def build_data_labels(values: Sequence[object] | pd.Series, percent: bool = False) -> pd.Series:
    return pd.Series(values).map(lambda value: format_data_label(value, percent=percent))


def _resolve_trace_color(trace: go.BaseTraceType, fallback: str = "#334155") -> str:
    line = getattr(trace, "line", None)
    if line is not None:
        color = getattr(line, "color", None)
        if color:
            return str(color)
    marker = getattr(trace, "marker", None)
    if marker is not None:
        color = getattr(marker, "color", None)
        if isinstance(color, str) and color:
            return color
    return fallback


def _is_party_series(x_col: str, color_col: str) -> bool:
    party_col = "\uC815\uB2F9\uBA85"
    return color_col == party_col or x_col == party_col


def _has_redundant_bar_legend(df: pd.DataFrame, x_col: str, color_col: str) -> bool:
    if x_col not in df.columns or color_col not in df.columns:
        return False
    x_values = df[x_col].dropna().astype("string").unique().tolist()
    color_values = df[color_col].dropna().astype("string").unique().tolist()
    return len(x_values) == len(color_values) and set(x_values) == set(color_values)


def entity_color_map(
    df: pd.DataFrame,
    entity_col: str,
    party_col: str = "정당명",
) -> dict[str, str]:
    if entity_col not in df.columns:
        return {}
    if entity_col == party_col or party_col not in df.columns:
        return party_color_map(df[entity_col]) if entity_col == party_col else _build_color_map(df[entity_col], PARTY_COLOR_OVERRIDES)

    entity_party = (
        df.loc[:, [entity_col, party_col]]
        .dropna(subset=[entity_col])
        .drop_duplicates(subset=[entity_col], keep="first")
    )
    party_colors = party_color_map(entity_party[party_col])
    color_map: dict[str, str] = {}
    fallback_index = 0

    for entity_value, party_value in entity_party.itertuples(index=False, name=None):
        entity_name = str(entity_value)
        if pd.notna(party_value) and str(party_value) in party_colors:
            color_map[entity_name] = party_colors[str(party_value)]
        else:
            color_map[entity_name] = FALLBACK_PALETTE[fallback_index % len(FALLBACK_PALETTE)]
            fallback_index += 1

    for entity_name in pd.Series(df[entity_col]).dropna().astype("string").unique().tolist():
        entity_name = str(entity_name)
        if entity_name in color_map:
            continue
        color_map[entity_name] = FALLBACK_PALETTE[fallback_index % len(FALLBACK_PALETTE)]
        fallback_index += 1

    return color_map


def _format_axes(fig: go.Figure, percent_y: bool = False, percent_x: bool = False) -> go.Figure:
    if percent_y:
        fig.update_yaxes(tickformat=".0%")
    else:
        fig.update_yaxes(tickformat=",")
    if percent_x:
        fig.update_xaxes(tickformat=".0%")
    else:
        fig.update_xaxes(tickformat=",")
    for axis_name in ("xaxis", "yaxis"):
        axis = getattr(fig.layout, axis_name, None)
        title = getattr(axis, "title", None)
        if title is None or not getattr(title, "text", None):
            continue
        title_text = str(title.text)
        if title_text == "득표율_계산":
            title.text = "득표율"
        elif "득표율_계산" in title_text:
            title.text = title_text.replace("득표율_계산", "득표율")
    return fig


def _is_election_axis(df: pd.DataFrame, x_col: str) -> bool:
    return x_col in ELECTION_AXIS_COLUMNS and x_col in df.columns


def _prepare_axis_frame(df: pd.DataFrame, x_col: str) -> tuple[pd.DataFrame, str, list[str] | None, bool]:
    chart_df = df.copy()
    display_col = x_col
    ordered_x: list[str] | None = None
    rotate_labels = False

    if x_col in chart_df.columns and isinstance(chart_df[x_col].dtype, pd.CategoricalDtype):
        chart_df[x_col] = chart_df[x_col].astype("string")
        categories = [str(value) for value in df[x_col].cat.categories.tolist()]
        available_values = set(chart_df[x_col].dropna().astype("string").tolist())
        ordered_x = [value for value in categories if value in available_values]
        return chart_df, display_col, ordered_x, rotate_labels

    if not _is_election_axis(chart_df, x_col):
        return chart_df, display_col, ordered_x, rotate_labels

    if x_col != "선거축라벨" and "선거축라벨" in chart_df.columns:
        display_col = "선거축라벨"

    chart_df[display_col] = chart_df[display_col].astype("string")
    sort_columns = [column for column in ["선거시점", "선거KEY"] if column in chart_df.columns]
    if sort_columns:
        order_frame = chart_df.loc[:, [*sort_columns, display_col]].drop_duplicates()
        order_frame = order_frame.sort_values(by=sort_columns, ascending=[True] * len(sort_columns), kind="stable")
        ordered_x = order_frame[display_col].astype("string").tolist()
    rotate_labels = True
    return chart_df, display_col, ordered_x, rotate_labels


def _apply_x_axis_format(fig: go.Figure, ordered_x: list[str] | None = None, rotate_labels: bool = False, bottom_margin: int = 20) -> go.Figure:
    if ordered_x:
        fig.update_xaxes(categoryorder="array", categoryarray=ordered_x)
    if rotate_labels:
        fig.update_xaxes(tickangle=-90, automargin=True)
        current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
        fig.update_layout(margin=dict(l=current_margin.get("l", 20), r=current_margin.get("r", 20), t=current_margin.get("t", 60), b=max(current_margin.get("b", 20), bottom_margin)))
    return fig


def line_metric_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str = "",
    percent: bool = False,
    area: bool = False,
    marker: bool = True,
    color_map: dict[str, str] | None = None,
    show_labels: bool = False,
    legend_order: list[str] | None = None,
    line_dash_map: dict[str, str] | None = None,
    trendline_entities: Sequence[str] | None = None,
    end_label_entities: Sequence[str] | None = None,
    end_label_text_map: dict[str, str] | None = None,
    hover_extra_cols: Sequence[str] | None = None,
    hover_extra_labels: dict[str, str] | None = None,
) -> go.Figure:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return _empty_figure("표시할 추이 데이터가 없습니다.")

    chart_df, display_x_col, ordered_x, rotate_labels = _prepare_axis_frame(df, x_col)
    category_orders: dict[str, list[str]] | None = None
    if color_col and color_col in chart_df.columns:
        legend_values = legend_order or chart_df[color_col].dropna().astype("string").unique().tolist()
        if legend_values:
            category_orders = {color_col: legend_values}
            chart_df[color_col] = pd.Categorical(chart_df[color_col], categories=legend_values, ordered=True)
            if ordered_x:
                x_order_map = {value: idx for idx, value in enumerate(ordered_x)}
                chart_df["__x_order__"] = chart_df[display_x_col].astype("string").map(x_order_map).fillna(len(ordered_x))
                chart_df = chart_df.sort_values(by=[color_col, "__x_order__"], kind="stable")
            else:
                chart_df = chart_df.sort_values(by=[color_col], kind="stable")
    if show_labels:
        chart_df["__label__"] = build_data_labels(chart_df[y_col], percent=percent)

    hover_columns: list[str] = []
    if display_x_col != x_col and "선거라벨" in chart_df.columns:
        hover_columns.append("선거라벨")
    for column in hover_extra_cols or []:
        if column in chart_df.columns and column not in hover_columns:
            hover_columns.append(column)
    custom_data = hover_columns or None
    fig = (
        px.area(
            chart_df,
            x=display_x_col,
            y=y_col,
            color=color_col,
            color_discrete_map=color_map,
            custom_data=custom_data,
            category_orders=category_orders,
        )
        if area
        else px.line(
            chart_df,
            x=display_x_col,
            y=y_col,
            color=color_col,
            markers=marker,
            color_discrete_map=color_map,
            text="__label__" if show_labels else None,
            custom_data=custom_data,
            category_orders=category_orders,
        )
    )
    fig.update_traces(
        mode="lines+markers" if marker and not area else None,
        textposition="top center" if show_labels else None,
        textfont=dict(size=BASE_CHART_TEXT_SIZE),
    )
    hover_title = "%{customdata[0]}" if hover_columns and hover_columns[0] == "선거라벨" else "%{x}"
    hover_value = "%{y:.2%}" if percent else "%{y:,.0f}"
    hover_lines = [hover_title]
    if color_col:
        hover_lines.append("대상: %{fullData.name}")
    hover_lines.append(hover_value)
    extra_start = 1 if hover_columns and hover_columns[0] == "선거라벨" else 0
    for index, column in enumerate(hover_columns[extra_start:], start=extra_start):
        label = (hover_extra_labels or {}).get(column, column)
        hover_lines.append(f"{label}: %{{customdata[{index}]}}")
    hover_template = "<br>".join(hover_lines) + "<extra></extra>"
    fig.update_traces(hovertemplate=hover_template)

    if line_dash_map:
        for trace in fig.data:
            dash = line_dash_map.get(str(trace.name))
            if dash:
                trace.line["dash"] = dash

    if trendline_entities and color_col and color_col in chart_df.columns:
        x_values = ordered_x or chart_df[display_x_col].dropna().astype("string").unique().tolist()
        x_order_map = {value: idx for idx, value in enumerate(x_values)}
        for entity in trendline_entities:
            entity_df = chart_df.loc[chart_df[color_col].astype("string") == str(entity)].copy()
            if len(entity_df) < 2:
                continue
            entity_df["__x_order__"] = entity_df[display_x_col].astype("string").map(x_order_map)
            entity_df = entity_df.dropna(subset=["__x_order__", y_col]).sort_values(by="__x_order__", kind="stable")
            if len(entity_df) < 2:
                continue
            trend_x_values = entity_df["__x_order__"].astype(float)
            if "선거시점" in entity_df.columns:
                period_values = pd.to_numeric(entity_df["선거시점"].astype("string").str.replace(r"\D", "", regex=True).str.slice(0, 6), errors="coerce")
                if period_values.nunique(dropna=True) >= 2:
                    trend_x_values = period_values.astype(float)
            if trend_x_values.nunique(dropna=True) < 2:
                continue
            try:
                slope, intercept = np.polyfit(trend_x_values.astype(float), entity_df[y_col].astype(float), 1)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            trend_x = entity_df[display_x_col].astype("string").tolist()
            trend_y = slope * trend_x_values.astype(float).to_numpy() + intercept
            trend_color = (color_map or {}).get(str(entity), "#6c757d")
            fig.add_trace(
                go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode="lines",
                    name=f"{entity} 선형 추세",
                    showlegend=False,
                    line=dict(color=trend_color, width=2, dash="dot"),
                    hovertemplate="%{x}<br>선거시점 기준 선형 추세: %{y:.2%}<extra></extra>" if percent else "%{x}<br>선거시점 기준 선형 추세: %{y:,.0f}<extra></extra>",
                )
            )

    if end_label_entities and color_col and color_col in chart_df.columns:
        x_values = ordered_x or chart_df[display_x_col].dropna().astype("string").unique().tolist()
        x_order_map = {value: idx for idx, value in enumerate(x_values)}
        color_lookup = {
            str(getattr(trace, "name", "")): _resolve_trace_color(
                trace,
                fallback=(color_map or {}).get(str(getattr(trace, "name", "")), "#334155"),
            )
            for trace in fig.data
        }
        label_entries: list[dict[str, object]] = []
        for entity in [str(value) for value in end_label_entities if value not in (None, "")]:
            entity_df = chart_df.loc[chart_df[color_col].astype("string") == entity].copy()
            if entity_df.empty:
                continue
            entity_df["__x_order__"] = entity_df[display_x_col].astype("string").map(x_order_map).fillna(len(x_values))
            entity_df = entity_df.dropna(subset=[y_col]).sort_values(by="__x_order__", kind="stable")
            if entity_df.empty:
                continue
            last_row = entity_df.iloc[-1]
            label_entries.append(
                {
                    "entity": entity,
                    "x": str(last_row[display_x_col]),
                    "y": float(last_row[y_col]),
                    "label": (end_label_text_map or {}).get(entity, entity),
                }
            )
        has_end_labels = bool(label_entries)
        if label_entries:
            y_values = [float(entry["y"]) for entry in label_entries]
            y_range = max(y_values) - min(y_values) if len(y_values) > 1 else 0.0
            threshold = max(y_range * 0.08, 0.01 if percent else 1.0)
            position_pattern = [
                "middle right",
                "top right",
                "bottom right",
                "top left",
                "bottom left",
                "middle left",
            ]
            previous_y: float | None = None
            cluster_index = 0
            for entry in sorted(label_entries, key=lambda item: float(item["y"]), reverse=True):
                current_y = float(entry["y"])
                if previous_y is None or abs(previous_y - current_y) > threshold:
                    cluster_index = 0
                else:
                    cluster_index += 1
                entry["textposition"] = position_pattern[min(cluster_index, len(position_pattern) - 1)]
                previous_y = current_y
        for entry in label_entries:
            entity = str(entry["entity"])
            fig.add_trace(
                go.Scatter(
                    x=[str(entry["x"])],
                    y=[float(entry["y"])],
                    mode="text",
                    name=entity,
                    legendgroup=entity,
                    text=[f"{entry['label']}<br>{format_data_label(entry['y'], percent=percent)}"],
                    textposition=str(entry.get("textposition", "middle right")),
                    textfont=dict(color=color_lookup.get(entity, (color_map or {}).get(entity, "#334155")), size=BASE_CHART_TEXT_SIZE),
                    showlegend=False,
                    hoverinfo="skip",
                    cliponaxis=False,
                )
            )
        if has_end_labels:
            current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
            fig.update_layout(
                margin=dict(
                    l=current_margin.get("l", 20),
                    r=max(current_margin.get("r", 20), 180),
                    t=current_margin.get("t", 60),
                    b=current_margin.get("b", 20),
                )
            )

    fig = _apply_common_layout(fig, title)
    if end_label_entities and color_col and color_col in chart_df.columns:
        current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
        fig.update_layout(
            margin=dict(
                l=current_margin.get("l", 20),
                r=max(current_margin.get("r", 20), 180),
                t=current_margin.get("t", 60),
                b=current_margin.get("b", 20),
            )
        )
    fig = _apply_x_axis_format(fig, ordered_x=ordered_x, rotate_labels=rotate_labels, bottom_margin=140)
    return _format_axes(fig, percent_y=percent, percent_x=False)


def grouped_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    percent: bool = False,
    orientation: str = "v",
    color_map: dict[str, str] | None = None,
    single_color: str | None = None,
    legend_order: list[str] | None = None,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [x_col, y_col, color_col]):
        return _empty_figure("표시할 비교 데이터가 없습니다.")

    chart_df, display_x_col, ordered_x, rotate_labels = _prepare_axis_frame(df, x_col)
    party_series = _is_party_series(x_col, color_col)
    redundant_legend = _has_redundant_bar_legend(chart_df, x_col, color_col)
    if legend_order and color_col in chart_df.columns:
        ordered_legend = [str(value) for value in legend_order if str(value) in chart_df[color_col].astype("string").unique().tolist()]
        if ordered_legend:
            chart_df[color_col] = pd.Categorical(chart_df[color_col], categories=ordered_legend, ordered=True)
            chart_df = chart_df.sort_values(by=[color_col], kind="stable")
    chart_df["__label__"] = build_data_labels(chart_df[y_col], percent=percent)
    if single_color is not None and redundant_legend and not party_series:
        bar_color = single_color or _chart_single_color(title)
        fig = px.bar(
            chart_df,
            x=display_x_col if orientation == "v" else y_col,
            y=y_col if orientation == "v" else display_x_col,
            orientation=orientation,
            text="__label__",
        )
        hover_template = (
            "%{x}<br>%{y:.2%}<extra></extra>"
            if percent and orientation == "v"
            else "%{y}<br>%{x:.2%}<extra></extra>"
            if percent
            else "%{x}<br>%{y:,.0f}<extra></extra>"
            if orientation == "v"
            else "%{y}<br>%{x:,.0f}<extra></extra>"
        )
        fig.update_traces(
            marker_color=bar_color,
            hovertemplate=hover_template,
            textposition="outside" if orientation == "v" else "auto",
            textfont=dict(size=_bar_label_text_size(orientation)),
            cliponaxis=False,
            showlegend=False,
        )
        fig.update_layout(bargap=0.18, bargroupgap=0.0, showlegend=False)
    else:
        if color_map:
            effective_color_map = color_map
        elif party_series:
            effective_color_map = party_color_map(chart_df[color_col])
        else:
            effective_color_map = _build_palette_map(chart_df[color_col], HARMONIOUS_PALETTE)
        fig = px.bar(
            chart_df,
            x=display_x_col if orientation == "v" else y_col,
            y=y_col if orientation == "v" else display_x_col,
            color=color_col,
            barmode="group",
            orientation=orientation,
            color_discrete_map=effective_color_map,
            text="__label__",
        )
        hover_template = (
            "%{x}<br>%{fullData.name}: %{y:.2%}<extra></extra>"
            if percent and orientation == "v"
            else "%{y}<br>%{fullData.name}: %{x:.2%}<extra></extra>"
            if percent
            else "%{x}<br>%{fullData.name}: %{y:,.0f}<extra></extra>"
            if orientation == "v"
            else "%{y}<br>%{fullData.name}: %{x:,.0f}<extra></extra>"
        )
        fig.update_traces(
            hovertemplate=hover_template,
            textposition="outside" if orientation == "v" else "auto",
            textfont=dict(size=_bar_label_text_size(orientation)),
            cliponaxis=False,
        )
        if party_series and color_col != x_col:
            fig.update_layout(barmode="overlay", bargap=0.16, bargroupgap=0.0)
            fig.update_traces(opacity=0.82, marker_line=dict(color="rgba(255,255,255,0.9)", width=0.8))
        else:
            fig.update_layout(bargap=0.18, bargroupgap=0.06)
        if redundant_legend and not party_series:
            fig.update_layout(showlegend=False)
    fig = _apply_common_layout(fig, title, height=480)
    if orientation == "v":
        fig = _apply_x_axis_format(fig, ordered_x=ordered_x, rotate_labels=rotate_labels, bottom_margin=140)
    return _format_axes(fig, percent_y=percent and orientation == "v", percent_x=percent and orientation == "h")


def stacked_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    percent: bool = False,
    color_map: dict[str, str] | None = None,
    legend_order: list[str] | None = None,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [x_col, y_col, color_col]):
        return _empty_figure("표시할 누적 데이터가 없습니다.")

    chart_df, display_x_col, ordered_x, rotate_labels = _prepare_axis_frame(df, x_col)
    party_series = _is_party_series(x_col, color_col)
    redundant_legend = _has_redundant_bar_legend(chart_df, display_x_col, color_col)
    if legend_order and color_col in chart_df.columns:
        ordered_legend = [str(value) for value in legend_order if str(value) in chart_df[color_col].astype("string").unique().tolist()]
        if ordered_legend:
            chart_df[color_col] = pd.Categorical(chart_df[color_col], categories=ordered_legend, ordered=True)
            chart_df = chart_df.sort_values(by=[color_col], kind="stable")
    if color_map:
        effective_color_map = color_map
    elif party_series:
        effective_color_map = party_color_map(chart_df[color_col])
    else:
        effective_color_map = _build_palette_map(chart_df[color_col], HARMONIOUS_PALETTE)
    chart_df["__label__"] = build_data_labels(chart_df[y_col], percent=percent)
    fig = px.bar(
        chart_df,
        x=display_x_col,
        y=y_col,
        color=color_col,
        barmode="stack",
        color_discrete_map=effective_color_map,
        text="__label__",
    )
    if percent:
        fig.update_layout(barnorm="percent")
        fig.update_traces(
            hovertemplate="%{x}<br>%{fullData.name}: %{y:.2%}<extra></extra>",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=_bar_label_text_size("v")),
        )
    else:
        fig.update_traces(
            hovertemplate="%{x}<br>%{fullData.name}: %{y:,.0f}<extra></extra>",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=_bar_label_text_size("v")),
        )
    fig.update_layout(bargap=0.14, bargroupgap=0.0)
    if redundant_legend and not party_series:
        fig.update_layout(showlegend=False)
    fig = _apply_common_layout(fig, title, height=480)
    fig = _apply_x_axis_format(fig, ordered_x=ordered_x, rotate_labels=rotate_labels, bottom_margin=140)
    fig = _format_axes(fig, percent_y=percent)
    if percent:
        fig.update_yaxes(tickformat=",", ticksuffix="%")
    return fig


def lollipop_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    percent: bool = False,
    color_col: str | None = None,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [category_col, value_col]):
        return _empty_figure("표시할 랭킹 데이터가 없습니다.")

    chart_df = df.copy().sort_values(by=value_col, ascending=True, kind="stable").reset_index(drop=True)
    marker_color = chart_df[color_col].map(color_map).tolist() if color_col and color_col in chart_df.columns and color_map else "#1F5AA6"
    labels = build_data_labels(chart_df[value_col], percent=percent)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df[value_col],
            y=chart_df[category_col],
            mode="markers+text",
            marker=dict(size=11, color=marker_color),
            customdata=chart_df[[category_col]].to_numpy(),
            text=labels,
            textposition="middle right",
            textfont=dict(size=BASE_CHART_TEXT_SIZE),
            hovertemplate="%{y}<br>%{x:.2%}<extra></extra>" if percent else "%{y}<br>%{x:,.0f}<extra></extra>",
        )
    )
    for _, row in chart_df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            x1=row[value_col],
            y0=row[category_col],
            y1=row[category_col],
            line=dict(color="rgba(31, 90, 166, 0.25)", width=2),
        )
    fig = _apply_common_layout(fig, title, height=560)
    return _format_axes(fig, percent_x=percent)


def heatmap_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str,
    percent: bool = False,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [x_col, y_col, value_col]):
        return _empty_figure("표시할 heatmap 데이터가 없습니다.")

    chart_df, display_x_col, ordered_x, rotate_labels = _prepare_axis_frame(df, x_col)
    pivot = chart_df.pivot_table(index=y_col, columns=display_x_col, values=value_col, aggfunc="first")
    if pivot.empty:
        return _empty_figure("표시할 heatmap 데이터가 없습니다.")

    if ordered_x:
        available_order = [value for value in ordered_x if value in pivot.columns]
        pivot = pivot.reindex(columns=available_order)

    formatter = (
        (lambda value: f"{value:.1%}" if pd.notna(value) else "")
        if percent
        else (lambda value: f"{value:,.0f}" if pd.notna(value) else "")
    )
    text = pivot.apply(lambda column: column.map(formatter))
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=HEATMAP_SCALE,
            text=text.values,
            texttemplate="%{text}",
            textfont=dict(size=BASE_CHART_TEXT_SIZE),
            hovertemplate="%{y}<br>%{x}<br>%{z:.2%}<extra></extra>" if percent else "%{y}<br>%{x}<br>%{z:,.0f}<extra></extra>",
        )
    )
    fig = _apply_common_layout(fig, title, height=max(420, 80 + 28 * len(pivot.index)))
    fig = _apply_x_axis_format(fig, ordered_x=ordered_x, rotate_labels=rotate_labels, bottom_margin=140)
    return fig


def scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: str | None = None,
    size_col: str | None = None,
    color_map: dict[str, str] | None = None,
    legend_order: list[str] | None = None,
    text_col: str | None = None,
    text_template: str | None = None,
    text_position: str = "top center",
    percent_x: bool = False,
    percent_y: bool = False,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [x_col, y_col]):
        return _empty_figure("표시할 산점도 데이터가 없습니다.")

    chart_df = df.copy()
    category_orders: dict[str, list[str]] = {}
    if legend_order and color_col and color_col in chart_df.columns:
        ordered_legend = [str(value) for value in legend_order if str(value) in chart_df[color_col].astype("string").unique().tolist()]
        if ordered_legend:
            chart_df[color_col] = pd.Categorical(chart_df[color_col], categories=ordered_legend, ordered=True)
            chart_df = chart_df.sort_values(by=[color_col], kind="stable")
            category_orders[color_col] = ordered_legend
    if text_col and text_col in chart_df.columns:
        chart_df[text_col] = chart_df[text_col].fillna("").astype("string")

    fig = px.scatter(
        chart_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in chart_df.columns else None,
        size=size_col if size_col and size_col in chart_df.columns else None,
        hover_name="지역" if "지역" in chart_df.columns else None,
        color_discrete_map=color_map,
        category_orders=category_orders or None,
        text=text_col if text_col and text_col in chart_df.columns else None,
    )
    if text_col and text_col in chart_df.columns and text_template:
        fig.update_traces(texttemplate=text_template, textposition=text_position)
        for trace in fig.data:
            if getattr(trace, "type", None) == "scatter":
                trace.update(cliponaxis=False)
    fig = _apply_common_layout(fig, title, height=520)
    return _format_axes(fig, percent_y=percent_y, percent_x=percent_x)


def distribution_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    chart_type: str = "box",
    color_col: str | None = None,
    color_map: dict[str, str] | None = None,
    legend_order: list[str] | None = None,
    percent: bool = False,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [category_col, value_col]):
        return _empty_figure("표시할 분포 데이터가 없습니다.")

    chart_df = df.copy()
    category_orders: dict[str, list[str]] = {}
    if legend_order and color_col and color_col in chart_df.columns:
        ordered_legend = [str(value) for value in legend_order if str(value) in chart_df[color_col].astype("string").unique().tolist()]
        if ordered_legend:
            chart_df[color_col] = pd.Categorical(chart_df[color_col], categories=ordered_legend, ordered=True)
            chart_df = chart_df.sort_values(by=[color_col], kind="stable")
            category_orders[color_col] = ordered_legend

    if chart_type == "violin":
        fig = px.violin(
            chart_df,
            x=category_col,
            y=value_col,
            color=color_col if color_col in chart_df.columns else None,
            box=True,
            color_discrete_map=color_map,
            category_orders=category_orders or None,
        )
    else:
        fig = px.box(
            chart_df,
            x=category_col,
            y=value_col,
            color=color_col if color_col in chart_df.columns else None,
            color_discrete_map=color_map,
            category_orders=category_orders or None,
        )
    fig = _apply_common_layout(fig, title, height=500)
    return _format_axes(fig, percent_y=percent)


def slope_chart(
    df: pd.DataFrame,
    entity_col: str,
    x_col: str,
    y_col: str,
    title: str,
    percent: bool = False,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [entity_col, x_col, y_col]):
        return _empty_figure("표시할 slope 데이터가 없습니다.")

    chart_df, display_x_col, ordered_x, rotate_labels = _prepare_axis_frame(df, x_col)
    if not ordered_x:
        ordered_x = chart_df[display_x_col].dropna().astype("string").unique().tolist()
    if len(ordered_x) < 2:
        return _empty_figure("slope chart는 최소 2개 선거가 필요합니다.")

    first_x, last_x = ordered_x[0], ordered_x[-1]
    subset = chart_df.loc[chart_df[display_x_col].astype("string").isin([first_x, last_x])].copy()
    if subset.empty:
        return _empty_figure("표시할 slope 데이터가 없습니다.")

    x_order_map = {value: index for index, value in enumerate(ordered_x)}

    fig = go.Figure()
    for entity, entity_df in subset.groupby(entity_col, observed=True):
        entity_df = entity_df.assign(__x_order__=entity_df[display_x_col].astype("string").map(x_order_map))
        entity_df = entity_df.sort_values(by="__x_order__", kind="stable")
        if len(entity_df) < 2:
            continue
        color = color_map.get(str(entity), "#1F5AA6") if color_map else "#1F5AA6"
        fig.add_trace(
            go.Scatter(
                x=entity_df[display_x_col],
                y=entity_df[y_col],
                mode="lines+markers+text",
                name=str(entity),
                line=dict(color=color, width=3),
                marker=dict(size=9, color=color),
                text=["", str(entity)],
                textposition="top right",
                textfont=dict(size=BASE_CHART_TEXT_SIZE),
                hovertemplate="%{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>" if percent else "%{x}<br>%{y:,.0f}<extra>%{fullData.name}</extra>",
            )
        )
    fig = _apply_common_layout(fig, title, height=440)
    fig = _apply_x_axis_format(fig, ordered_x=ordered_x, rotate_labels=rotate_labels, bottom_margin=140)
    return _format_axes(fig, percent_y=percent)


def bump_chart(
    df: pd.DataFrame,
    entity_col: str,
    x_col: str,
    rank_col: str,
    title: str,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [entity_col, x_col, rank_col]):
        return _empty_figure("표시할 bump chart 데이터가 없습니다.")

    chart_df, display_x_col, ordered_x, rotate_labels = _prepare_axis_frame(df, x_col)
    x_order_map = {value: index for index, value in enumerate(ordered_x or chart_df[display_x_col].dropna().astype("string").unique().tolist())}
    fig = go.Figure()
    for entity, entity_df in chart_df.groupby(entity_col, observed=True):
        entity_df = entity_df.assign(__x_order__=entity_df[display_x_col].astype("string").map(x_order_map))
        entity_df = entity_df.sort_values(by="__x_order__", kind="stable")
        color = color_map.get(str(entity), "#1F5AA6") if color_map else "#1F5AA6"
        fig.add_trace(
            go.Scatter(
                x=entity_df[display_x_col],
                y=entity_df[rank_col],
                mode="lines+markers",
                name=str(entity),
                line=dict(color=color, width=3),
                marker=dict(size=9, color=color),
                hovertemplate="%{x}<br>순위 %{y}<extra>%{fullData.name}</extra>",
            )
        )
    fig = _apply_common_layout(fig, title, height=440)
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig = _apply_x_axis_format(fig, ordered_x=ordered_x, rotate_labels=rotate_labels, bottom_margin=140)
    return fig


def bar_turnout_by_region(df: pd.DataFrame, value_col: str = "투표율", title: str = "지역별 투표율") -> go.Figure:
    if df.empty or value_col not in df.columns or "지역" not in df.columns:
        return _empty_figure("표시할 지역별 투표율 데이터가 없습니다.")
    return lollipop_chart(df, "지역", value_col, title=title, percent="율" in value_col)


def bar_turnout_by_rowtype(df: pd.DataFrame, value_col: str = "투표수", title: str = "RowType별 투표수") -> go.Figure:
    if df.empty or value_col not in df.columns:
        return _empty_figure("표시할 RowType 데이터가 없습니다.")
    return grouped_bar_chart(df, "RowType", value_col, "RowType", title=title, percent="율" in value_col, color_map=rowtype_color_map(df["RowType"]))


def line_turnout_by_election(df: pd.DataFrame, value_col: str = "투표율", title: str = "선거별 투표율 비교") -> go.Figure:
    if df.empty or value_col not in df.columns:
        return _empty_figure("표시할 선거별 비교 데이터가 없습니다.")
    x_col = "선거라벨" if "선거라벨" in df.columns else "선거KEY"
    return line_metric_chart(df, x_col, value_col, title=title, percent="율" in value_col)


def bar_candidate_votes(
    df: pd.DataFrame,
    x_col: str = "후보명",
    y_col: str = "득표수",
    color_col: str | None = "정당명",
    title: str = "후보별 득표수",
) -> go.Figure:
    if df.empty or any(column not in df.columns for column in [x_col, y_col]):
        return _empty_figure("표시할 후보 득표 데이터가 없습니다.")
    color_map = party_color_map(df[color_col]) if color_col and color_col in df.columns else None
    return grouped_bar_chart(df, x_col, y_col, color_col or x_col, title=title, orientation="h", color_map=color_map)


def bar_party_share(df: pd.DataFrame, x_col: str = "정당명", y_col: str = "득표율_계산", title: str = "정당별 득표율") -> go.Figure:
    if df.empty or any(column not in df.columns for column in [x_col, y_col]):
        return _empty_figure("표시할 정당 득표율 데이터가 없습니다.")
    return grouped_bar_chart(df, x_col, y_col, x_col, title=title, percent=True, color_map=party_color_map(df[x_col]))


def bar_candidate_share_by_region(df: pd.DataFrame, value_col: str = "득표율_계산", title: str = "지역별 후보 득표율") -> go.Figure:
    if df.empty or value_col not in df.columns or "지역" not in df.columns:
        return _empty_figure("표시할 지역별 후보 득표율 데이터가 없습니다.")
    return lollipop_chart(df, "지역", value_col, title=title, percent=True, color_col="정당명", color_map=party_color_map(df.get("정당명")))
