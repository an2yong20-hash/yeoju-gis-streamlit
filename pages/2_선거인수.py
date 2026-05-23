from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import grouped_bar_chart, heatmap_chart, line_metric_chart, lollipop_chart
from src.filters import (
    apply_common_filters,
    build_breadcrumb_text,
    build_region_comparison_selection,
    get_selected_adjacent_gusigun_names,
    render_sidebar_filters,
    resolve_region_level,
)
from src.loaders import (
    get_cache_file_signatures,
    load_fact_confirmed_electorate_enriched,
    load_fact_resident_composition_enriched,
    load_global_filter_options,
)
from src.maps import build_region_choropleth, get_adjacent_gusigun_keys, load_geometry_context
from src.metrics import (
    build_region_label,
    calc_confirmed_electorate_summary_by_election,
    calc_confirmed_electorate_trend_by_level,
    dataframe_to_csv_bytes,
    format_delta_pp,
    format_int,
    format_percent,
    get_latest_and_previous,
    safe_divide,
)

CONFIRMED_COLUMNS = (
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
    "투표구명",
    "투표소KEY",
    "RowType",
    "확정선거인수",
)
EXACT_RESIDENT_AGE_COLUMNS = [f"{age}세" for age in range(0, 100)] + ["100세이상"]
DETAIL_RESIDENT_AGE_COLUMNS = ["70대인구", "80대인구", "90대인구", "100세이상인구"]
RESIDENT_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "선거연령기준",
    "기준월",
    "기준월라벨",
    "행정기관코드",
    "시도명",
    "구시군명",
    "구시군KEY",
    "일반구명",
    "읍면동명",
    "읍면동KEY",
    "RowType",
    "총인구수",
    "세대수",
    "남자인구수",
    "여자인구수",
    "평균연령",
    "전월인구수",
    "인구증감",
    "아동인구",
    "청소년인구",
    "청년인구",
    "고령인구",
    "1인가구수",
    "청년1인가구수_추정",
    "노년1인가구수_추정",
    "30대인구",
    "40대인구",
    "50대인구",
    "60대인구",
    *DETAIL_RESIDENT_AGE_COLUMNS,
    "70세이상인구",
    *EXACT_RESIDENT_AGE_COLUMNS,
    "선거연령청년인구_추정",
    "남성구성비",
    "여성구성비",
    "선거연령청년구성비_추정",
    "30대구성비",
    "40대구성비",
    "50대구성비",
    "60대구성비",
    "70세이상구성비",
    "아동구성비",
    "청소년구성비",
    "청년구성비",
    "고령구성비",
    "1인가구비율",
    "청년1인가구비율_추정",
    "노년1인가구비율_추정",
    "1인가구연령통계가용여부",
)
PERIOD_EVENT_NAME_MAP = {
    "P": "대통령선거",
    "N": "국회의원선거",
    "L": "지방선거",
    "POP": "주민등록인구통계",
}
REGION_KEY_COLUMNS = ["시도명", "구시군KEY", "구시군명", "읍면동KEY", "일반구명", "읍면동명", "투표소KEY", "지역"]
OVERVIEW_SNAPSHOT_COLORS = {
    "age": "#5C7C96",
    "life": "#6FA49C",
    "gender": "#B06F86",
    "household": "#C69458",
}
RESIDENT_SUM_COLUMNS = [
    "총인구수",
    "세대수",
    "남자인구수",
    "여자인구수",
    "전월인구수",
    "인구증감",
    "아동인구",
    "청소년인구",
    "청년인구",
    "고령인구",
    "1인가구수",
    "청년1인가구수_추정",
    "노년1인가구수_추정",
    "30대인구",
    "40대인구",
    "50대인구",
    "60대인구",
    *DETAIL_RESIDENT_AGE_COLUMNS,
    "70세이상인구",
    *EXACT_RESIDENT_AGE_COLUMNS,
    "선거연령청년인구_추정",
]
RESIDENT_LEVEL_GROUP_COLUMNS = {
    "시도": ["시도명"],
    "구시군": ["구시군KEY", "시도명", "구시군명"],
    "읍면동": ["읍면동KEY", "구시군KEY", "시도명", "구시군명", "일반구명", "읍면동명"],
}
RESIDENT_LEVEL_LABEL_COLUMNS = {
    "시도": ["시도명"],
    "구시군": ["시도명", "구시군명"],
    "읍면동": ["시도명", "구시군명", "일반구명", "읍면동명"],
}
RESIDENT_ADMIN_EMD_RENAME_ALIASES = [
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
RESIDENT_METRIC_CONFIG: dict[str, dict[str, object]] = {
    "선거인수비중": {"label": "인구 대비 선거인 비중", "kind": "percent"},
    "인구대비선거인비중_주민등록기준": {"label": "인구 대비 선거인 비중", "kind": "percent"},
    "선거연령청년구성비_추정": {"label": "18/19~29세 구성비", "kind": "percent"},
    "평균연령": {"label": "평균연령", "kind": "float", "suffix": "세"},
    "선거인평균연령_주민등록기준": {"label": "선거인 평균연령", "kind": "float", "suffix": "세"},
    "1인가구비율": {"label": "1인가구 비율", "kind": "percent"},
    "50대이하선거인비중_주민등록기준": {"label": "50대 이하 선거인수 비중", "kind": "percent"},
    "60세이상선거인비중_주민등록기준": {"label": "60세 이상 선거인수 비중", "kind": "percent"},
    "고령비고령선거인격차_주민등록기준": {"label": "고령-비고령 선거인 격차", "kind": "signed_count"},
    "남성구성비": {"label": "남성 구성비", "kind": "percent"},
    "여성구성비": {"label": "여성 구성비", "kind": "percent"},
    "30대구성비": {"label": "30대 구성비", "kind": "percent"},
    "40대구성비": {"label": "40대 구성비", "kind": "percent"},
    "50대구성비": {"label": "50대 구성비", "kind": "percent"},
    "60대구성비": {"label": "60대 구성비", "kind": "percent"},
    "70세이상구성비": {"label": "70세 이상 구성비", "kind": "percent"},
    "아동구성비": {"label": "아동 구성비", "kind": "percent"},
    "청소년구성비": {"label": "청소년 구성비", "kind": "percent"},
    "청년구성비": {"label": "청년 구성비", "kind": "percent"},
    "고령구성비": {"label": "고령 구성비", "kind": "percent"},
    "인구증감률": {"label": "인구증감률", "kind": "percent"},
    "선거인수_주민등록기준": {"label": "선거인 수(주민등록인구 기준)", "kind": "count"},
    "50대이하선거인수_주민등록기준": {"label": "50대 이하 선거인수", "kind": "count"},
    "60세이상선거인수_주민등록기준": {"label": "60세 이상 선거인수", "kind": "count"},
    "총인구수": {"label": "총인구수", "kind": "count"},
    "아동인구": {"label": "아동 인구수", "kind": "count"},
    "청소년인구": {"label": "청소년 인구수", "kind": "count"},
    "청년인구": {"label": "청년 인구수", "kind": "count"},
    "고령인구": {"label": "고령 인구수", "kind": "count"},
    "30대인구": {"label": "30대 인구수", "kind": "count"},
    "40대인구": {"label": "40대 인구수", "kind": "count"},
    "50대인구": {"label": "50대 인구수", "kind": "count"},
    "60대인구": {"label": "60대 인구수", "kind": "count"},
    "70세이상인구": {"label": "70세 이상 인구수", "kind": "count"},
    "남자인구수": {"label": "남성 수", "kind": "count"},
    "여자인구수": {"label": "여성 수", "kind": "count"},
    "세대수": {"label": "세대수", "kind": "count"},
    "1인가구수": {"label": "1인가구수", "kind": "count"},
    "선거연령청년인구_추정": {"label": "18/19~29세 인구", "kind": "count"},
    "인구증감": {"label": "인구증감", "kind": "signed_count"},
    "청년1인가구수_추정": {"label": "청년 1인가구수", "kind": "count", "requires_age_stats": True},
    "노년1인가구수_추정": {"label": "노년 1인가구수", "kind": "count", "requires_age_stats": True},
    "청년1인가구비율_추정": {"label": "청년 1인가구 비율", "kind": "percent", "requires_age_stats": True},
    "노년1인가구비율_추정": {"label": "노년 1인가구 비율", "kind": "percent", "requires_age_stats": True},
}
RESIDENT_DISPLAY_COLUMN_ALIASES = {
    "선거인수_주민등록기준": "선거인 수(주민등록인구 기준)",
    "선거인수비중": "인구 대비 선거인 비중",
    "인구대비선거인비중_주민등록기준": "인구 대비 선거인 비중",
    "선거인평균연령_주민등록기준": "선거인 평균연령",
    "50대이하선거인수_주민등록기준": "50대 이하 선거인수",
    "50대이하선거인비중_주민등록기준": "50대 이하 선거인수 비중",
    "60세이상선거인수_주민등록기준": "60세 이상 선거인수",
    "60세이상선거인비중_주민등록기준": "60세 이상 선거인수 비중",
    "고령비고령선거인격차_주민등록기준": "고령-비고령 선거인 격차",
    "선거연령청년인구_추정": "18/19~29세 인구",
    "선거연령청년구성비_추정": "18/19~29세 구성비",
    "청년1인가구수_추정": "청년1인가구수",
    "노년1인가구수_추정": "노년1인가구수",
    "청년1인가구비율_추정": "청년1인가구비율",
    "노년1인가구비율_추정": "노년1인가구비율",
    "남자인구수": "남성수",
    "여자인구수": "여성수",
}
RESIDENT_METRIC_OPTIONS = [
    "선거인수비중",
    "50대이하선거인비중_주민등록기준",
    "60세이상선거인비중_주민등록기준",
    "선거연령청년구성비_추정",
    "평균연령",
    "선거인평균연령_주민등록기준",
    "1인가구비율",
    "남성구성비",
    "여성구성비",
    "30대구성비",
    "40대구성비",
    "50대구성비",
    "60대구성비",
    "70세이상구성비",
    "아동구성비",
    "청소년구성비",
    "청년구성비",
    "고령구성비",
    "인구증감률",
    "선거인수_주민등록기준",
    "50대이하선거인수_주민등록기준",
    "60세이상선거인수_주민등록기준",
    "총인구수",
    "선거연령청년인구_추정",
    "30대인구",
    "40대인구",
    "50대인구",
    "60대인구",
    "70세이상인구",
    "아동인구",
    "청소년인구",
    "청년인구",
    "고령인구",
    "남자인구수",
    "여자인구수",
    "세대수",
    "1인가구수",
    "인구증감",
    "청년1인가구수_추정",
    "노년1인가구수_추정",
    "청년1인가구비율_추정",
    "노년1인가구비율_추정",
]
RESIDENT_METRIC_GROUPS: dict[str, list[str]] = {
    "개요": [],
    "선거연령/연령구성": [
        "선거인수비중",
        "50대이하선거인비중_주민등록기준",
        "60세이상선거인비중_주민등록기준",
        "선거연령청년구성비_추정",
        "30대구성비",
        "40대구성비",
        "50대구성비",
        "60대구성비",
        "70세이상구성비",
        "아동구성비",
        "청소년구성비",
        "청년구성비",
        "고령구성비",
        "평균연령",
        "선거인평균연령_주민등록기준",
    ],
    "성별/가구구성": [
        "남성구성비",
        "여성구성비",
        "1인가구비율",
        "청년1인가구비율_추정",
        "노년1인가구비율_추정",
    ],
    "인구수": [
        "선거인수_주민등록기준",
        "50대이하선거인수_주민등록기준",
        "60세이상선거인수_주민등록기준",
        "총인구수",
        "선거연령청년인구_추정",
        "30대인구",
        "40대인구",
        "50대인구",
        "60대인구",
        "70세이상인구",
        "아동인구",
        "청소년인구",
        "청년인구",
        "고령인구",
        "남자인구수",
        "여자인구수",
    ],
    "세대수": [
        "세대수",
        "1인가구수",
        "청년1인가구수_추정",
        "노년1인가구수_추정",
    ],
    "증감": [
        "인구증감",
        "인구증감률",
    ],
}


def _load_bundle() -> dict[str, object]:
    cache_signature = get_cache_file_signatures()
    return {
        "confirmed": load_fact_confirmed_electorate_enriched(cache_signature, CONFIRMED_COLUMNS),
        "resident": load_fact_resident_composition_enriched(cache_signature, RESIDENT_COLUMNS),
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


def _format_period_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if len(text) == 6 and text.isdigit():
        return f"{text[:4]}-{text[4:]}"
    return text


def _event_name_from_keys(keys: pd.Series) -> str:
    values = keys.dropna().astype("string")
    if values.str.contains("-POP", regex=False).any():
        return PERIOD_EVENT_NAME_MAP["POP"]
    if values.str.contains("-P", regex=False).any():
        return PERIOD_EVENT_NAME_MAP["P"]
    if values.str.contains("-N", regex=False).any():
        return PERIOD_EVENT_NAME_MAP["N"]
    if values.str.contains("-L", regex=False).any():
        return PERIOD_EVENT_NAME_MAP["L"]
    return "선거"


def _build_period_label(period: object, event_name: str) -> str:
    period_text = _format_period_text(period)
    return f"{period_text} {event_name}".strip()


def _build_count_delta_sentence(current: object, delta: object) -> str:
    if pd.isna(current):
        return "확정선거인수 기준이 부족합니다."
    if pd.isna(delta):
        return f"선택 범위의 최신 확정선거인수는 {format_int(current)}명입니다."
    delta_value = float(delta)
    if abs(delta_value) < 1e-12:
        return f"선택 범위의 최신 확정선거인수는 직전 선거와 같은 {format_int(current)}명입니다."
    direction = "증가" if delta_value > 0 else "감소"
    return f"선택 범위의 최신 확정선거인수는 {format_int(current)}명으로 직전 선거 대비 {format_int(abs(delta_value))}명 {direction}했습니다."


@st.cache_data(show_spinner=False)
def _prepare_resident_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    for column in ["선거시점", "선거명", "선거종류", "선거라벨"]:
        if column in result.columns:
            result[column] = result[column].astype("string")
    if "선거시점" in result.columns and "기준월" in result.columns:
        missing_period_mask = result["선거시점"].isna() & result["기준월"].notna()
        result.loc[missing_period_mask, "선거시점"] = result.loc[missing_period_mask, "기준월"].astype("string")
    if "선거명" in result.columns:
        missing_name_mask = result["선거명"].isna()
        result.loc[missing_name_mask, "선거명"] = "주민등록인구통계"
    if "선거종류" in result.columns:
        missing_type_mask = result["선거종류"].isna()
        result.loc[missing_type_mask, "선거종류"] = "주민등록인구"
    if "선거라벨" in result.columns and "기준월라벨" in result.columns:
        missing_label_mask = result["선거라벨"].isna() & result["기준월라벨"].notna()
        result.loc[missing_label_mask, "선거라벨"] = (
            result.loc[missing_label_mask, "기준월라벨"].astype("string") + " 주민등록인구통계"
        )
    return result


def _aggregate_to_period_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    passthrough_cols = [column for column in REGION_KEY_COLUMNS if column in df.columns and column not in group_cols]
    if "선거KEY" not in group_cols:
        passthrough_cols.append("선거KEY")
    rows: list[dict[str, object]] = []

    for key_values, group in df.groupby(group_cols, observed=True, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {column: value for column, value in zip(group_cols, key_values)}
        representative = group.sort_values(by=["선거KEY"], kind="stable").iloc[0]
        for column in passthrough_cols:
            row[column] = representative[column]
        row["확정선거인수"] = group["확정선거인수"].max()
        event_name = _event_name_from_keys(group["선거KEY"])
        row["선거이벤트"] = event_name
        row["선거축라벨"] = _build_period_label(row.get("선거시점"), event_name)
        row["선거라벨"] = row["선거축라벨"]
        rows.append(row)

    result = pd.DataFrame(rows)
    sort_columns = [column for column in [*group_cols, "선거KEY"] if column in result.columns]
    if sort_columns:
        result = result.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)
    return result


@st.cache_data(show_spinner=False)
def _build_period_trend_frame(confirmed_df: pd.DataFrame) -> pd.DataFrame:
    by_election = calc_confirmed_electorate_summary_by_election(confirmed_df)
    if by_election.empty:
        return by_election
    result = _aggregate_to_period_frame(by_election, ["선거시점"])
    return result.sort_values(by=["선거시점"], ascending=True, kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _build_region_period_trend_frame(confirmed_df: pd.DataFrame, level: str) -> pd.DataFrame:
    by_election = calc_confirmed_electorate_trend_by_level(confirmed_df, level=level)
    if by_election.empty:
        return by_election
    region_cols = [column for column in REGION_KEY_COLUMNS if column in by_election.columns]
    result = _aggregate_to_period_frame(by_election, ["선거시점", *region_cols])
    return result.sort_values(by=["선거시점", "지역"], ascending=[True, True], kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _build_region_current_frame(region_trend_df: pd.DataFrame, latest_period: object) -> pd.DataFrame:
    if region_trend_df.empty:
        return region_trend_df
    result = region_trend_df.loc[region_trend_df["선거시점"] == latest_period].copy()
    return result.reset_index(drop=True)


def _resolve_resident_overview_row(resident_trend_df: pd.DataFrame, election_period: object) -> pd.Series | None:
    if resident_trend_df.empty:
        return None
    if election_period is not None and "선거시점" in resident_trend_df.columns:
        matched = resident_trend_df.loc[resident_trend_df["선거시점"].astype("string") == str(election_period)].copy()
        if not matched.empty:
            sort_cols = [column for column in ["선거시점", "선거KEY"] if column in matched.columns]
            return matched.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), kind="stable").iloc[0]
    sort_cols = [column for column in ["선거시점", "선거KEY"] if column in resident_trend_df.columns]
    return resident_trend_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), kind="stable").iloc[0]


def _resident_display_value_selection(selected: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in selected.items()}
    for column in ["선거KEY", "선거시점", "선거명", "선거종류", "RowType", "정당명", "후보명"]:
        result[column] = []
    return result


def _resolve_latest_period(frame: pd.DataFrame, fallback: object) -> object:
    if frame.empty or "선거시점" not in frame.columns:
        return fallback
    period_series = frame["선거시점"].dropna()
    if period_series.empty:
        return fallback
    return period_series.max()


def _infer_sido_scope_for_gis(filtered_frame: pd.DataFrame, selected: dict[str, list[str]]) -> list[str]:
    if selected.get("시도명") or filtered_frame.empty or "시도명" not in filtered_frame.columns:
        return []
    return filtered_frame["시도명"].dropna().astype("string").unique().tolist()


def _resolve_focus_gusigun_key(filtered_frame: pd.DataFrame) -> str | None:
    if filtered_frame.empty or "구시군KEY" not in filtered_frame.columns:
        return None
    keys = [
        str(value)
        for value in filtered_frame["구시군KEY"].dropna().astype("string").unique().tolist()
        if "합계" not in str(value)
    ]
    return keys[0] if len(keys) == 1 else None


def _resolve_single_selected_region_label(selected: dict[str, list[str]], frame: pd.DataFrame) -> str | None:
    """
    if frame.empty or "吏?? not in frame.columns:
        return None
    candidate = frame.copy()
    has_region_filter = False
    for column in ["?쒕룄紐?, "援ъ떆援곕챸", "?쇰컲援щ챸", "?띾㈃?숇챸"]:
        values = [str(value) for value in selected.get(column, []) if value not in (None, "")]
        if not values:
            continue
        if len(values) != 1:
            return None
        has_region_filter = True
        if column in candidate.columns:
            candidate = candidate.loc[candidate[column].astype("string") == values[0]]
    if not has_region_filter:
        return None
    labels = candidate["吏??].dropna().astype("string").unique().tolist()
    return labels[0] if len(labels) == 1 else None


    """
    return _resolve_chart_end_label(selected, frame)


def _resolve_chart_end_labels(
    selected: dict[str, list[str]],
    frame: pd.DataFrame,
    preferred_level: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    region_col = "\uC9C0\uC5ED"
    if frame.empty or region_col not in frame.columns:
        return [], {}
    candidate = frame.copy()
    entities = candidate[region_col].dropna().astype("string").unique().tolist()
    if not entities:
        return [], {}
    if preferred_level == "읍면동" and "\uC74D\uBA74\uB3D9\uBA85" in candidate.columns:
        label_col = "\uC74D\uBA74\uB3D9\uBA85"
    elif preferred_level == "구시군" and "\uAD6C\uC2DC\uAD70\uBA85" in candidate.columns:
        label_col = "\uAD6C\uC2DC\uAD70\uBA85"
    elif preferred_level == "시도" and "\uC2DC\uB3C4\uBA85" in candidate.columns:
        label_col = "\uC2DC\uB3C4\uBA85"
    elif selected.get("\uC74D\uBA74\uB3D9\uBA85") and "\uC74D\uBA74\uB3D9\uBA85" in candidate.columns:
        label_col = "\uC74D\uBA74\uB3D9\uBA85"
    elif selected.get("\uAD6C\uC2DC\uAD70\uBA85") and "\uAD6C\uC2DC\uAD70\uBA85" in candidate.columns:
        label_col = "\uAD6C\uC2DC\uAD70\uBA85"
    elif selected.get("\uC2DC\uB3C4\uBA85") and "\uC2DC\uB3C4\uBA85" in candidate.columns:
        label_col = "\uC2DC\uB3C4\uBA85"
    else:
        label_col = region_col
    label_map: dict[str, str] = {}
    for entity in entities:
        entity_rows = candidate.loc[candidate[region_col].astype("string") == entity]
        if label_col in entity_rows.columns:
            label_values = entity_rows[label_col].dropna().astype("string").unique().tolist()
            label_map[entity] = label_values[0] if label_values else entity
        else:
            label_map[entity] = entity
    return entities, label_map


def _has_group_values(series: pd.Series) -> bool:
    values = series.astype("string").str.strip().replace("", pd.NA)
    return values.notna().any()


def _resident_metric_label(metric: str) -> str:
    return str(RESIDENT_METRIC_CONFIG.get(metric, {}).get("label", metric))


def _resident_display_column(column: str) -> str:
    return RESIDENT_DISPLAY_COLUMN_ALIASES.get(column, column)


def _resident_display_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        target = df.copy()
    elif columns is None:
        target = df.copy()
    else:
        target = df.loc[:, columns].copy()
    rename_map = {column: _resident_display_column(column) for column in target.columns if column in RESIDENT_DISPLAY_COLUMN_ALIASES}
    return target.rename(columns=rename_map)


def _resident_metric_kind(metric: str) -> str:
    return str(RESIDENT_METRIC_CONFIG.get(metric, {}).get("kind", "count"))


def _resident_metric_is_percent(metric: str) -> bool:
    return _resident_metric_kind(metric) == "percent"


def _resident_metric_is_float(metric: str) -> bool:
    return _resident_metric_kind(metric) == "float"


def _resident_metric_is_signed_count(metric: str) -> bool:
    return _resident_metric_kind(metric) == "signed_count"


def _resident_metric_requires_age_stats(metric: str) -> bool:
    return bool(RESIDENT_METRIC_CONFIG.get(metric, {}).get("requires_age_stats", False))


def _format_resident_metric_value(metric: str, value: object) -> str:
    if _resident_metric_is_percent(metric):
        return format_percent(value)
    if _resident_metric_is_float(metric):
        if pd.isna(value):
            return "-"
        suffix = str(RESIDENT_METRIC_CONFIG.get(metric, {}).get("suffix", ""))
        return f"{float(value):,.1f}{suffix}"
    if _resident_metric_is_signed_count(metric):
        if pd.isna(value):
            return "-"
        return f"{float(value):+,.0f}"
    return format_int(value)


def _format_resident_metric_delta(metric: str, delta: object) -> str | None:
    if pd.isna(delta):
        return None
    if _resident_metric_is_percent(metric):
        return format_delta_pp(delta)
    if _resident_metric_is_float(metric):
        suffix = str(RESIDENT_METRIC_CONFIG.get(metric, {}).get("suffix", ""))
        return f"{float(delta):+.1f}{suffix}"
    if _resident_metric_is_signed_count(metric):
        return f"{float(delta):+,.0f}"
    return format_int(delta)


def _build_resident_metric_sentence(metric: str, current: object, delta: object, latest_label: str) -> str:
    metric_label = _resident_metric_label(metric)
    current_text = _format_resident_metric_value(metric, current)
    if pd.isna(current):
        return f"{metric_label} 기준이 부족합니다."
    if pd.isna(delta):
        return f"{latest_label} 기준 {metric_label}은 {current_text}입니다."
    delta_value = float(delta)
    if abs(delta_value) < 1e-12:
        return f"{latest_label} 기준 {metric_label}은 직전 선거와 같은 수준입니다."
    if _resident_metric_is_percent(metric):
        direction = "상승" if delta_value > 0 else "하락"
        return f"{latest_label} 기준 {metric_label}은 {current_text}로 직전 선거 대비 {abs(delta_value) * 100:.2f}%p {direction}했습니다."
    if _resident_metric_is_float(metric):
        direction = "높아졌습니다" if delta_value > 0 else "낮아졌습니다"
        return f"{latest_label} 기준 {metric_label}은 {current_text}로 직전 선거 대비 {abs(delta_value):.1f}세 {direction}"
    direction = "증가" if delta_value > 0 else "감소"
    return f"{latest_label} 기준 {metric_label}은 {current_text}로 직전 선거 대비 {format_int(abs(delta_value))}{direction}했습니다."


def _to_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _sum_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan
    total = pd.to_numeric(df[column], errors="coerce").sum(min_count=1)
    return np.nan if pd.isna(total) else float(total)


def _sum_available_values(values: list[object]) -> float:
    numeric_values = [float(value) for value in values if pd.notna(value)]
    return float(sum(numeric_values)) if numeric_values else np.nan


def _sum_exact_resident_age_counts(df: pd.DataFrame, start_age: int, end_age: int = 100) -> float:
    values: list[float] = []
    for age in range(start_age, min(end_age, 99) + 1):
        value = _sum_numeric(df, f"{age}세")
        if pd.notna(value):
            values.append(float(value))
    if end_age >= 100:
        value = _sum_numeric(df, "100세이상")
        if pd.notna(value):
            values.append(float(value))
    return float(sum(values)) if values else np.nan


def _exact_resident_electorate_stats(df: pd.DataFrame, age_basis: int) -> tuple[float, float, float]:
    required_columns = [f"{age}세" for age in range(max(age_basis, 0), 100)] + ["100세이상"]
    if any(column not in df.columns for column in required_columns):
        return np.nan, np.nan, np.nan

    counts: list[tuple[int, float]] = []
    for age in range(max(age_basis, 0), 100):
        value = _sum_numeric(df, f"{age}세")
        if pd.notna(value):
            counts.append((age, float(value)))
    over_100 = _sum_numeric(df, "100세이상")
    if pd.notna(over_100):
        counts.append((100, float(over_100)))
    if not counts:
        return np.nan, np.nan, np.nan

    electorate_population = sum(count for _, count in counts)
    electorate_youth = sum(count for age, count in counts if age < 30)
    electorate_average_age = (
        sum(age * count for age, count in counts) / electorate_population
        if electorate_population > 0
        else np.nan
    )
    return electorate_population, electorate_youth, electorate_average_age


def _electorate_youth_age_range_label(row: pd.Series | dict[str, object] | None) -> str:
    if row is None:
        return "18/19~29세"
    value = row.get("선거연령기준", np.nan)
    if pd.isna(value):
        return "18/19~29세"
    try:
        age_basis = int(float(value))
    except (TypeError, ValueError):
        return "18/19~29세"
    if age_basis == 18:
        return "18~29세"
    if age_basis == 19:
        return "19~29세"
    return f"{age_basis}~29세"


def _electorate_youth_chart_label(row: pd.Series | dict[str, object] | None) -> str:
    return _electorate_youth_age_range_label(row)


def _weighted_average(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if value_col not in df.columns or weight_col not in df.columns:
        return np.nan
    value = pd.to_numeric(df[value_col], errors="coerce")
    weight = pd.to_numeric(df[weight_col], errors="coerce")
    valid = value.notna() & weight.notna() & weight.gt(0)
    if not valid.any():
        return np.nan
    return float((value.loc[valid] * weight.loc[valid]).sum() / weight.loc[valid].sum())


def _resident_level_group_columns(df: pd.DataFrame, level: str) -> list[str]:
    return [
        column
        for column in RESIDENT_LEVEL_GROUP_COLUMNS[level]
        if column in df.columns and _has_group_values(df[column])
    ]


def _resident_level_label_columns(df: pd.DataFrame, level: str) -> list[str]:
    return [
        column
        for column in RESIDENT_LEVEL_LABEL_COLUMNS[level]
        if column in df.columns and _has_group_values(df[column])
    ]


def _attach_resident_geo_id(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    rowtype = result["RowType"].astype("string") if "RowType" in result.columns else pd.Series("", index=result.index, dtype="string")

    emd_label = build_region_label(result, ["시도명", "구시군명", "일반구명", "읍면동명"]).astype("string")
    sig_label = build_region_label(result, ["시도명", "구시군명"]).astype("string")
    sido_label = result["시도명"].astype("string").fillna("전국") if "시도명" in result.columns else pd.Series("전국", index=result.index, dtype="string")
    emd_key = result["읍면동KEY"].astype("string") if "읍면동KEY" in result.columns else pd.Series(pd.NA, index=result.index, dtype="string")
    sig_key = result["구시군KEY"].astype("string") if "구시군KEY" in result.columns else pd.Series(pd.NA, index=result.index, dtype="string")

    emd_id = ("EMD|" + emd_key).where(emd_key.notna(), "EMD|" + emd_label)
    sig_id = ("SIG|" + sig_key).where(sig_key.notna(), "SIG|" + sig_label)
    sido_id = "SIDO|" + sido_label

    geo_id = pd.Series("ALL", index=result.index, dtype="string")
    geo_id = geo_id.where(~rowtype.eq("시도"), sido_id)
    geo_id = geo_id.where(~rowtype.eq("구시군"), sig_id)
    geo_id = geo_id.where(~rowtype.eq("읍면동"), emd_id)
    result["__resident_geo_id"] = geo_id
    return result


def _select_resident_scope_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "선거KEY" not in df.columns or "RowType" not in df.columns:
        return df.iloc[0:0].copy()

    selected_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("선거KEY", observed=True, sort=False):
        for rowtype, key_col in (("합계", None), ("시도", "시도명"), ("구시군", "구시군KEY"), ("읍면동", "읍면동KEY")):
            matched = group.loc[group["RowType"].astype("string") == rowtype].copy()
            if key_col is not None and key_col in matched.columns:
                matched = matched.loc[matched[key_col].astype("string").notna()].copy()
            if not matched.empty:
                selected_parts.append(matched)
                break

    if not selected_parts:
        return df.iloc[0:0].copy()
    return pd.concat(selected_parts, ignore_index=True)


def _select_resident_rows_by_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()

    if level == "시도":
        result = df.loc[df["RowType"].astype("string") == "시도"].copy()
    elif level == "구시군":
        result = df.loc[(df["RowType"].astype("string") == "구시군") & df["구시군KEY"].astype("string").notna()].copy()
    else:
        result = df.loc[(df["RowType"].astype("string") == "읍면동") & df["읍면동명"].astype("string").notna()].copy()
    return result.reset_index(drop=True)


def _dedupe_resident_period_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "선거시점" not in df.columns:
        return df
    prepared = _attach_resident_geo_id(df)
    sort_cols = [column for column in ["선거시점", "선거KEY", "행정기관코드"] if column in df.columns]
    return (
        prepared.sort_values(by=sort_cols, kind="stable")
        .drop_duplicates(subset=["선거시점", "__resident_geo_id"], keep="first")
        .drop(columns="__resident_geo_id")
        .reset_index(drop=True)
    )


def _aggregate_resident_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, *RESIDENT_METRIC_OPTIONS, "기준월", "기준월라벨", "선거축라벨", "선거라벨"])

    rows: list[dict[str, object]] = []
    group_iter = [((), df)] if not group_cols else df.groupby(group_cols, observed=True, sort=False, dropna=False)

    for key_values, group in group_iter:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {column: value for column, value in zip(group_cols, key_values)}
        representative = group.sort_values(by=[column for column in ["선거KEY", "행정기관코드"] if column in group.columns], kind="stable").iloc[0]

        for column in ["선거KEY", "선거시점", "선거명", "선거종류", "선거연령기준", "기준월", "기준월라벨", "시도명", "구시군명", "구시군KEY", "일반구명", "읍면동명", "읍면동KEY", "행정기관코드"]:
            if column in group.columns and column not in row:
                row[column] = representative[column]

        total_population = _sum_numeric(group, "총인구수")
        households = _sum_numeric(group, "세대수")
        male_population = _sum_numeric(group, "남자인구수")
        female_population = _sum_numeric(group, "여자인구수")
        previous_population = _sum_numeric(group, "전월인구수")
        population_delta = _sum_numeric(group, "인구증감")
        children = _sum_numeric(group, "아동인구")
        teens = _sum_numeric(group, "청소년인구")
        youth = _sum_numeric(group, "청년인구")
        elderly = _sum_numeric(group, "고령인구")
        single_households = _sum_numeric(group, "1인가구수")
        thirties = _sum_numeric(group, "30대인구")
        forties = _sum_numeric(group, "40대인구")
        fifties = _sum_numeric(group, "50대인구")
        sixties = _sum_numeric(group, "60대인구")
        seventies_plus = _sum_numeric(group, "70세이상인구")
        try:
            electorate_age_basis = int(float(representative.get("선거연령기준", 18)))
        except (TypeError, ValueError):
            electorate_age_basis = 18
        (
            exact_electorate_population,
            exact_electorate_youth,
            exact_electorate_average_age,
        ) = _exact_resident_electorate_stats(group, electorate_age_basis)
        electorate_youth = (
            exact_electorate_youth
            if pd.notna(exact_electorate_youth)
            else _sum_numeric(group, "선거연령청년인구_추정")
        )
        electorate_age_population = (
            exact_electorate_population
            if pd.notna(exact_electorate_population)
            else _sum_available_values([electorate_youth, thirties, forties, fifties, sixties, seventies_plus])
        )
        electorate_under_sixty = _sum_exact_resident_age_counts(group, electorate_age_basis, 59)
        if pd.isna(electorate_under_sixty):
            electorate_under_sixty = _sum_available_values([electorate_youth, thirties, forties, fifties])
        electorate_sixty_plus = _sum_exact_resident_age_counts(group, 60, 100)
        if pd.isna(electorate_sixty_plus):
            electorate_sixty_plus = _sum_available_values([sixties, seventies_plus])
        average_age = _weighted_average(group, "평균연령", "총인구수")
        electorate_average_age = exact_electorate_average_age
        if pd.isna(electorate_average_age):
            electorate_youth_midpoint = 23.5 if electorate_age_basis <= 18 else 24.0
            seventies = _sum_numeric(group, "70대인구")
            eighties = _sum_numeric(group, "80대인구")
            nineties = _sum_numeric(group, "90대인구")
            hundreds = _sum_numeric(group, "100세이상인구")
            older_counts = (
                [(seventies, 74.5), (eighties, 84.5), (nineties, 94.5), (hundreds, 100.0)]
                if any(pd.notna(value) for value in [seventies, eighties, nineties, hundreds])
                else [(seventies_plus, 75.0)]
            )
            electorate_age_counts = [
                (electorate_youth, electorate_youth_midpoint),
                (thirties, 34.5),
                (forties, 44.5),
                (fifties, 54.5),
                (sixties, 64.5),
                *older_counts,
            ]
            electorate_age_weight = sum(float(count) for count, _ in electorate_age_counts if pd.notna(count))
            electorate_average_age = (
                sum(float(count) * midpoint for count, midpoint in electorate_age_counts if pd.notna(count)) / electorate_age_weight
                if electorate_age_weight > 0
                else np.nan
            )
        if pd.isna(electorate_average_age) and pd.notna(average_age) and pd.notna(total_population) and pd.notna(electorate_age_population):
            under_electorate_population = float(total_population) - float(electorate_age_population)
            if under_electorate_population >= 0 and float(electorate_age_population) > 0:
                under_electorate_midpoint = max((electorate_age_basis - 1) / 2, 0)
                total_age_weight = float(average_age) * float(total_population)
                electorate_average_age = (
                    total_age_weight - under_electorate_population * under_electorate_midpoint
                ) / float(electorate_age_population)

        age_stats_mask = _to_numeric_series(group, "1인가구연령통계가용여부").fillna(0).gt(0)
        age_stats_group = group.loc[age_stats_mask].copy()
        young_single_households = _sum_numeric(age_stats_group, "청년1인가구수_추정")
        old_single_households = _sum_numeric(age_stats_group, "노년1인가구수_추정")
        single_households_with_age_stats = _sum_numeric(age_stats_group, "1인가구수")

        row["총인구수"] = total_population
        row["세대수"] = households
        row["남자인구수"] = male_population
        row["여자인구수"] = female_population
        row["평균연령"] = average_age
        row["전월인구수"] = previous_population
        row["인구증감"] = population_delta
        row["인구증감률"] = safe_divide(population_delta, previous_population)
        row["아동인구"] = children
        row["청소년인구"] = teens
        row["청년인구"] = youth
        row["고령인구"] = elderly
        row["1인가구수"] = single_households
        row["청년1인가구수_추정"] = young_single_households
        row["노년1인가구수_추정"] = old_single_households
        row["30대인구"] = thirties
        row["40대인구"] = forties
        row["50대인구"] = fifties
        row["60대인구"] = sixties
        row["70세이상인구"] = seventies_plus
        row["선거연령인구_추정"] = electorate_age_population
        row["선거인수_주민등록기준"] = electorate_age_population
        row["50대이하선거인수_주민등록기준"] = electorate_under_sixty
        row["60세이상선거인수_주민등록기준"] = electorate_sixty_plus
        row["고령비고령선거인격차_주민등록기준"] = (
            float(electorate_sixty_plus) - float(electorate_under_sixty)
            if pd.notna(electorate_sixty_plus) and pd.notna(electorate_under_sixty)
            else np.nan
        )
        row["선거연령청년인구_추정"] = electorate_youth
        row["남성구성비"] = safe_divide(male_population, total_population)
        row["여성구성비"] = safe_divide(female_population, total_population)
        row["선거인수비중"] = safe_divide(electorate_age_population, total_population)
        row["인구대비선거인비중_주민등록기준"] = row["선거인수비중"]
        row["선거인평균연령_주민등록기준"] = electorate_average_age
        row["50대이하선거인비중_주민등록기준"] = safe_divide(electorate_under_sixty, electorate_age_population)
        row["60세이상선거인비중_주민등록기준"] = safe_divide(electorate_sixty_plus, electorate_age_population)
        row["선거연령청년구성비_추정"] = safe_divide(electorate_youth, electorate_age_population)
        row["30대구성비"] = safe_divide(thirties, electorate_age_population)
        row["40대구성비"] = safe_divide(forties, electorate_age_population)
        row["50대구성비"] = safe_divide(fifties, electorate_age_population)
        row["60대구성비"] = safe_divide(sixties, electorate_age_population)
        row["70세이상구성비"] = safe_divide(seventies_plus, electorate_age_population)
        row["아동구성비"] = safe_divide(children, total_population)
        row["청소년구성비"] = safe_divide(teens, total_population)
        row["청년구성비"] = safe_divide(youth, total_population)
        row["고령구성비"] = safe_divide(elderly, total_population)
        row["1인가구비율"] = safe_divide(single_households, households)
        row["청년1인가구비율_추정"] = safe_divide(young_single_households, single_households_with_age_stats)
        row["노년1인가구비율_추정"] = safe_divide(old_single_households, single_households_with_age_stats)
        row["1인가구연령통계가용여부"] = 1 if age_stats_mask.any() else 0

        event_name = _event_name_from_keys(group["선거KEY"]) if "선거KEY" in group.columns else "선거"
        row["선거이벤트"] = event_name
        row["선거축라벨"] = _build_period_label(row.get("선거시점"), event_name)
        row["선거라벨"] = row["선거축라벨"]
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    sort_columns = [column for column in [*group_cols, "선거시점", "선거KEY"] if column in result.columns]
    if sort_columns:
        result = result.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)
    return result


@st.cache_data(show_spinner=False)
def _build_resident_period_trend_frame(resident_df: pd.DataFrame) -> pd.DataFrame:
    base = _select_resident_scope_rows(resident_df)
    if base.empty:
        return base
    deduped = _dedupe_resident_period_rows(base)
    result = _aggregate_resident_metrics(deduped, ["선거시점"])
    return result.sort_values(by=["선거시점"], ascending=True, kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _build_resident_region_period_trend_frame(resident_df: pd.DataFrame, level: str) -> pd.DataFrame:
    base = _select_resident_rows_by_level(resident_df, level)
    if base.empty:
        return base
    deduped = _dedupe_resident_period_rows(base)
    group_cols = ["선거시점", *_resident_level_group_columns(deduped, level)]
    result = _aggregate_resident_metrics(deduped, group_cols)
    label_cols = _resident_level_label_columns(result, level)
    if label_cols:
        result["지역"] = build_region_label(result, label_cols)
    else:
        result["지역"] = "지역"
    return result.sort_values(by=["선거시점", "지역"], ascending=[True, True], kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _get_latest_metric_state(df: pd.DataFrame, metric: str) -> dict[str, object]:
    if df.empty or metric not in df.columns:
        return {"current": np.nan, "previous": np.nan, "delta": np.nan, "latest_row": None}
    ordered = df.sort_values(by=[column for column in ["선거시점", "선거KEY"] if column in df.columns], ascending=False, kind="stable")
    available = ordered.loc[ordered[metric].notna()].reset_index(drop=True)
    if available.empty:
        return {"current": np.nan, "previous": np.nan, "delta": np.nan, "latest_row": None}
    latest_row = available.iloc[0]
    previous_row = available.iloc[1] if len(available) > 1 else None
    current = latest_row[metric]
    previous = np.nan if previous_row is None else previous_row[metric]
    return {
        "current": current,
        "previous": previous,
        "delta": np.nan if previous_row is None else current - previous,
        "latest_row": latest_row,
    }


def _build_latest_snapshot_frame(latest_row: pd.Series, items: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"항목": label, "값": latest_row.get(metric, np.nan)} for label, metric in items]
    )


def _resident_row_text(row: pd.Series | None, column: str, fallback: str = "-") -> str:
    if row is None or column not in row.index:
        return fallback
    value = row.get(column)
    if pd.isna(value):
        return fallback
    return str(value)


def _sum_row_metrics(row: pd.Series, metrics: list[str]) -> float:
    values: list[float] = []
    for metric in metrics:
        if metric not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
        if pd.notna(value):
            values.append(float(value))
    return float(sum(values)) if values else np.nan


def _resident_metric_numeric(row: pd.Series | None, metric: str) -> float:
    if row is None:
        return np.nan
    if metric == "50대이하선거인수_주민등록기준":
        direct_value = _resident_metric_numeric(row, "__direct_50대이하선거인수_주민등록기준")
        if pd.notna(direct_value):
            return direct_value
        return _sum_row_metrics(row, ["선거연령청년인구_추정", "30대인구", "40대인구", "50대인구"])
    if metric == "60세이상선거인수_주민등록기준":
        direct_value = _resident_metric_numeric(row, "__direct_60세이상선거인수_주민등록기준")
        if pd.notna(direct_value):
            return direct_value
        return _sum_row_metrics(row, ["60대인구", "70세이상인구"])
    if metric == "고령비고령선거인격차_주민등록기준":
        direct_value = _resident_metric_numeric(row, "__direct_고령비고령선거인격차_주민등록기준")
        if pd.notna(direct_value):
            return direct_value
        older = _resident_metric_numeric(row, "60세이상선거인수_주민등록기준")
        younger = _resident_metric_numeric(row, "50대이하선거인수_주민등록기준")
        return older - younger if pd.notna(older) and pd.notna(younger) else np.nan
    if metric == "50대이하선거인비중_주민등록기준":
        numerator = _resident_metric_numeric(row, "50대이하선거인수_주민등록기준")
        denominator = _resident_metric_numeric(row, "선거인수_주민등록기준")
        return safe_divide(numerator, denominator)
    if metric == "60세이상선거인비중_주민등록기준":
        numerator = _resident_metric_numeric(row, "60세이상선거인수_주민등록기준")
        denominator = _resident_metric_numeric(row, "선거인수_주민등록기준")
        return safe_divide(numerator, denominator)
    if metric == "__direct_50대이하선거인수_주민등록기준":
        metric = "50대이하선거인수_주민등록기준"
    elif metric == "__direct_60세이상선거인수_주민등록기준":
        metric = "60세이상선거인수_주민등록기준"
    elif metric == "__direct_고령비고령선거인격차_주민등록기준":
        metric = "고령비고령선거인격차_주민등록기준"
    if metric not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
    return np.nan if pd.isna(value) else float(value)


def _resident_metric_change(latest_row: pd.Series | None, reference_row: pd.Series | None, metric: str) -> float:
    latest_value = _resident_metric_numeric(latest_row, metric)
    reference_value = _resident_metric_numeric(reference_row, metric)
    if pd.isna(latest_value) or pd.isna(reference_value):
        return np.nan
    return float(latest_value - reference_value)


def _format_resident_metric_change(metric: str, delta: object) -> str | None:
    if pd.isna(delta):
        return None
    if _resident_metric_is_percent(metric):
        return format_delta_pp(delta)
    if _resident_metric_is_float(metric):
        suffix = str(RESIDENT_METRIC_CONFIG.get(metric, {}).get("suffix", ""))
        return f"{float(delta):+.1f}{suffix}"
    return f"{float(delta):+,.0f}"


def _format_signed_percent(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:+.2f}%"


def _period_month_delta(latest_period: object, reference_period: object) -> int | None:
    try:
        latest_text = str(latest_period)
        reference_text = str(reference_period)
        latest_year, latest_month = int(latest_text[:4]), int(latest_text[4:6])
        reference_year, reference_month = int(reference_text[:4]), int(reference_text[4:6])
    except (TypeError, ValueError):
        return None
    return (latest_year - reference_year) * 12 + (latest_month - reference_month)


def _resident_change_items(latest_row: pd.Series | None) -> dict[str, list[tuple[str, str]]]:
    youth_age_label = _electorate_youth_chart_label(latest_row)
    return {
        "핵심": [
            ("총인구수", "총인구수"),
            ("세대수", "세대수"),
            ("평균연령", "평균연령"),
            ("인구 대비 선거인 비중", "선거인수비중"),
            ("1인가구 수", "1인가구수"),
            ("1인가구 비율", "1인가구비율"),
            ("청년(19~34세) 구성비", "청년구성비"),
            ("고령(65세 이상) 구성비", "고령구성비"),
        ],
        "총량": [
            ("총인구수", "총인구수"),
            ("세대수", "세대수"),
            ("1인가구 수", "1인가구수"),
            ("아동(0~17세) 인구", "아동인구"),
            ("청년(19~34세) 인구", "청년인구"),
            ("고령(65세 이상) 인구", "고령인구"),
        ],
        "선거연령대": [
            (youth_age_label, "선거연령청년구성비_추정"),
            ("30대", "30대구성비"),
            ("40대", "40대구성비"),
            ("50대", "50대구성비"),
            ("60대", "60대구성비"),
            ("70세 이상", "70세이상구성비"),
        ],
        "생애주기": [
            ("아동(0~17세)", "아동구성비"),
            ("청소년(9~24세)", "청소년구성비"),
            ("청년(19~34세)", "청년구성비"),
            ("고령(65세 이상)", "고령구성비"),
        ],
        "가구·성별": [
            ("남성", "남성구성비"),
            ("여성", "여성구성비"),
            ("1인가구 비율", "1인가구비율"),
            ("청년 1인가구 비율", "청년1인가구비율_추정"),
            ("노년 1인가구 비율", "노년1인가구비율_추정"),
        ],
    }


def _resident_overview_metric_items() -> list[tuple[str, str]]:
    return [
        ("총 인구수", "총인구수"),
        ("평균 연령", "평균연령"),
        ("세대 수", "세대수"),
        ("1인가구 수", "1인가구수"),
        ("1인가구 비율", "1인가구비율"),
        ("선거인 수(주민등록인구 기준)", "선거인수_주민등록기준"),
        ("선거인 평균연령", "선거인평균연령_주민등록기준"),
        ("인구 대비 선거인 비중", "선거인수비중"),
        ("50대 이하 선거인수", "50대이하선거인수_주민등록기준"),
        ("50대 이하 선거인수 비중", "50대이하선거인비중_주민등록기준"),
        ("60세 이상 선거인수", "60세이상선거인수_주민등록기준"),
        ("60세 이상 선거인수 비중", "60세이상선거인비중_주민등록기준"),
        ("고령-비고령 선거인 격차", "고령비고령선거인격차_주민등록기준"),
    ]


def _resident_age_band_items(display_row: pd.Series | None) -> list[tuple[str, str, str]]:
    youth_age_label = _electorate_youth_chart_label(display_row)
    return [
        (f"20대 이하({youth_age_label})", "선거연령청년인구_추정", "선거연령청년구성비_추정"),
        ("30대", "30대인구", "30대구성비"),
        ("40대", "40대인구", "40대구성비"),
        ("50대", "50대인구", "50대구성비"),
        ("60대", "60대인구", "60대구성비"),
        ("70세 이상", "70세이상인구", "70세이상구성비"),
    ]


def _build_resident_age_band_frame(display_row: pd.Series | None, reference_row: pd.Series | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, population_metric, share_metric in _resident_age_band_items(display_row):
        population = _resident_metric_numeric(display_row, population_metric)
        share = _resident_metric_numeric(display_row, share_metric)
        population_delta = _resident_metric_change(display_row, reference_row, population_metric)
        share_delta = _resident_metric_change(display_row, reference_row, share_metric)
        rows.append(
            {
                "연령 구간": label,
                "인구수(명)": format_int(population),
                "인구수 변화": _format_resident_metric_change(population_metric, population_delta) or "-",
                "선거인 구성비(%)": format_percent(share),
                "구성비 변화": _format_resident_metric_change(share_metric, share_delta) or "-",
            }
        )
    return pd.DataFrame(rows)


def _resident_emd_label(row: pd.Series) -> str:
    value = row.get("읍면동명", np.nan)
    if pd.notna(value) and str(value).strip():
        return str(value)
    value = row.get("지역", np.nan)
    if pd.notna(value) and str(value).strip():
        return str(value)
    return "-"


def _resident_alias_emd_merge_key(row: pd.Series) -> str | None:
    sido = str(row.get("시도명", ""))
    gusigun = str(row.get("구시군명", ""))
    emd = str(row.get("읍면동명", ""))
    for alias in RESIDENT_ADMIN_EMD_RENAME_ALIASES:
        if (
            sido == str(alias["시도명"])
            and gusigun == str(alias["구시군명"])
            and emd in {str(alias["from"]), str(alias["to"])}
        ):
            return f"ALIAS|{sido}|{gusigun}|{alias['to']}"
    return None


def _resident_emd_merge_key(row: pd.Series) -> str:
    alias_key = _resident_alias_emd_merge_key(row)
    if alias_key is not None:
        return alias_key
    key = row.get("읍면동KEY", np.nan)
    if pd.notna(key) and str(key).strip():
        return f"KEY|{key}"
    return f"LABEL|{_resident_emd_label(row)}"


def _resident_period_rows(df: pd.DataFrame, period: object) -> pd.DataFrame:
    if df.empty or "선거시점" not in df.columns or period is None:
        return df.iloc[0:0].copy()
    return df.loc[df["선거시점"].astype("string") == str(period)].copy()


def _build_resident_emd_composition_frame(
    emd_trend_df: pd.DataFrame,
    display_row: pd.Series | None,
    reference_row: pd.Series | None,
) -> pd.DataFrame:
    if emd_trend_df.empty or display_row is None:
        return pd.DataFrame()

    display_period = display_row.get("선거시점", None)
    reference_period = None if reference_row is None else reference_row.get("선거시점", None)
    display_rows = _resident_period_rows(emd_trend_df, display_period)
    reference_rows = _resident_period_rows(emd_trend_df, reference_period)
    if display_rows.empty:
        return pd.DataFrame()

    reference_lookup = {
        _resident_emd_merge_key(row): row
        for _, row in reference_rows.iterrows()
    }
    display_total_electorate = _resident_metric_numeric(display_row, "선거인수_주민등록기준")
    reference_total_electorate = _resident_metric_numeric(reference_row, "선거인수_주민등록기준")

    rows: list[dict[str, object]] = []
    for _, display_emd_row in display_rows.iterrows():
        reference_emd_row = reference_lookup.get(_resident_emd_merge_key(display_emd_row))
        electorate = _resident_metric_numeric(display_emd_row, "선거인수_주민등록기준")
        reference_electorate = _resident_metric_numeric(reference_emd_row, "선거인수_주민등록기준")
        share = safe_divide(electorate, display_total_electorate)
        reference_share = safe_divide(reference_electorate, reference_total_electorate)
        share_delta = share - reference_share if pd.notna(share) and pd.notna(reference_share) else np.nan
        rows.append(
            {
                "읍면동": _resident_emd_label(display_emd_row),
                "선거인수(명)": format_int(electorate),
                "선거인수 변화": (
                    _format_resident_metric_change("선거인수_주민등록기준", electorate - reference_electorate)
                    if pd.notna(electorate) and pd.notna(reference_electorate)
                    else "-"
                ),
                "선거인 구성비(%)": format_percent(share),
                "구성비 변화": _format_resident_metric_change("선거인수비중", share_delta) or "-",
            }
        )

    return pd.DataFrame(rows).sort_values(by=["읍면동"], kind="stable").reset_index(drop=True)


def _build_resident_change_summary_frame(
    latest_row: pd.Series | None,
    reference_row: pd.Series | None,
    item_groups: dict[str, list[tuple[str, str]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen_metrics: set[str] = set()
    for group_name, items in item_groups.items():
        for label, metric in items:
            if metric in seen_metrics and group_name != "핵심":
                continue
            seen_metrics.add(metric)
            latest_value = _resident_metric_numeric(latest_row, metric)
            reference_value = _resident_metric_numeric(reference_row, metric)
            delta = _resident_metric_change(latest_row, reference_row, metric)
            relative_delta = safe_divide(delta, reference_value) if not (_resident_metric_is_percent(metric) or _resident_metric_is_float(metric)) else np.nan
            rows.append(
                {
                    "구분": group_name,
                    "항목": label,
                    "선거시점 기준": _format_resident_metric_value(metric, reference_value),
                    "최신 기준": _format_resident_metric_value(metric, latest_value),
                    "변화": _format_resident_metric_change(metric, delta) or "-",
                    "변화율": _format_signed_percent(relative_delta) if pd.notna(relative_delta) else "-",
                }
            )
    return pd.DataFrame(rows)


def _build_resident_change_chart_frame(
    latest_row: pd.Series | None,
    reference_row: pd.Series | None,
    items: list[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for label, metric in items:
        delta = _resident_metric_change(latest_row, reference_row, metric)
        if pd.isna(delta):
            continue
        rows.append({"항목": label, "변화": delta, "지표": label})
    return pd.DataFrame(rows)


def _render_resident_change_metric_grid(
    latest_row: pd.Series | None,
    reference_row: pd.Series | None,
    items: list[tuple[str, str]],
) -> None:
    for start in range(0, len(items), 4):
        columns = st.columns(4)
        for column, (label, metric) in zip(columns, items[start : start + 4]):
            latest_value = _resident_metric_numeric(latest_row, metric)
            delta = _resident_metric_change(latest_row, reference_row, metric)
            column.metric(
                label,
                _format_resident_metric_value(metric, latest_value),
                delta=_format_resident_metric_change(metric, delta),
            )


def _render_resident_overview_card_grid(
    display_row: pd.Series | None,
    reference_row: pd.Series | None,
    display_month: str,
    reference_month: str,
) -> None:
    card_items: list[tuple[str, str, str | None]] = [
        ("주민등록 기준시점", display_month, None),
        ("비교 대상 기준시점", reference_month, None),
    ]
    for label, metric in _resident_overview_metric_items():
        latest_value = _resident_metric_numeric(display_row, metric)
        delta = _resident_metric_change(display_row, reference_row, metric)
        card_items.append(
            (
                label,
                _format_resident_metric_value(metric, latest_value),
                _format_resident_metric_change(metric, delta),
            )
        )

    for start in range(0, len(card_items), 5):
        columns = st.columns(5)
        for column, (label, value, delta) in zip(columns, card_items[start : start + 5]):
            column.metric(label, value, delta=delta)


def _render_resident_change_overview_section(
    display_row: pd.Series | None,
    reference_row: pd.Series | None,
    reference_election_label: str,
    emd_trend_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if display_row is None or reference_row is None:
        st.info("선택된 최근 선거 시점과 비교할 주민등록인구 데이터가 부족합니다.")
        return pd.DataFrame()

    display_month = _resident_row_text(display_row, "기준월라벨")
    reference_month = _resident_row_text(reference_row, "기준월라벨")
    _render_resident_overview_card_grid(display_row, reference_row, display_month, reference_month)
    st.caption("카드의 증감값은 주민등록 기준시점 값에서 비교 대상 기준시점 값을 뺀 값입니다.")
    st.caption("선거인 수와 인구 대비 선거인 비중은 확정선거인수 대신 주민등록 인구의 선거연령 이상 인구로 계산합니다.")
    st.caption("고령-비고령 선거인 격차는 `60세 이상 선거인 수 - 50대 이하 선거인 수`이며, 양수는 60세 이상 우위, 음수는 50대 이하 우위입니다.")

    with st.expander("연령 기준", expanded=False):
        st.markdown(_resident_age_basis_markdown())

    st.subheader("연령 구간별 인구수 및 선거인 구성비")
    age_band_frame = _build_resident_age_band_frame(display_row, reference_row)
    st.dataframe(age_band_frame, use_container_width=True, hide_index=True)
    st.caption("20대 이하는 해당 시점의 선거연령 기준에 따라 18~29세 또는 19~29세로 계산합니다.")
    st.caption("선거인 평균연령은 선거연령 기준(만 18세 또는 만 19세 이상)에 맞춰 주민등록 1세 단위 인구를 직접 합산해 계산합니다.")

    st.subheader("읍면동별 인구수 및 선거인 구성비")
    emd_frame = _build_resident_emd_composition_frame(
        emd_trend_df if emd_trend_df is not None else pd.DataFrame(),
        display_row,
        reference_row,
    )
    if emd_frame.empty:
        st.info("현재 선택 범위에서 표시할 읍면동별 주민등록인구 데이터가 없습니다.")
    else:
        st.dataframe(emd_frame, use_container_width=True, hide_index=True)
        st.caption("읍면동별 선거인 구성비는 선택 범위 전체 선거인 수 대비 해당 읍면동의 주민등록인구 기준 선거인 수 비중입니다.")

    if int(display_row.get("1인가구연령통계가용여부", 0) or 0) == 0:
        st.caption("청년·노년 1인가구 지표는 행정안전부 1인가구 연령 통계가 제공되는 시점부터만 계산됩니다.")

    frames = [age_band_frame.assign(표="연령 구간별")]
    if not emd_frame.empty:
        frames.append(emd_frame.assign(표="읍면동별"))
    return pd.concat(frames, ignore_index=True, sort=False)


def _render_resident_overview_section(latest_row: pd.Series | None, prefix: str = "최신") -> None:
    if latest_row is None:
        st.info("표시할 주민등록인구 개요 데이터가 없습니다.")
        return

    electorate_youth_chart_label = _electorate_youth_chart_label(latest_row)

    resident_overview_col1, resident_overview_col2, resident_overview_col3, resident_overview_col4 = st.columns(4)
    resident_overview_col1.metric(
        "기준월",
        str(latest_row["기준월라벨"]) if "기준월라벨" in latest_row.index else "-",
    )
    resident_overview_col2.metric(
        "총인구수",
        format_int(latest_row["총인구수"]),
    )
    resident_overview_col3.metric(
        "세대수",
        format_int(latest_row["세대수"]),
    )
    resident_overview_col4.metric(
        "평균연령",
        _format_resident_metric_value("평균연령", latest_row["평균연령"]),
    )

    resident_overview_col5, resident_overview_col6, resident_overview_col7, resident_overview_col8 = st.columns(4)
    resident_overview_col5.metric(
        "1인가구 수",
        format_int(latest_row["1인가구수"]),
    )
    resident_overview_col6.metric(
        "1인가구 비율",
        _format_resident_metric_value("1인가구비율", latest_row["1인가구비율"]),
    )
    resident_overview_col7.metric(
        f"{electorate_youth_chart_label} 수",
        format_int(latest_row["선거연령청년인구_추정"]),
    )
    resident_overview_col8.metric(
        f"{electorate_youth_chart_label} 구성비",
        _format_resident_metric_value("선거연령청년구성비_추정", latest_row["선거연령청년구성비_추정"]),
    )

    with st.expander("연령 기준", expanded=False):
        st.markdown(_resident_age_basis_markdown())

    overview_chart_col1, overview_chart_col2 = st.columns(2)
    electorate_age_frame = _build_latest_snapshot_frame(
        latest_row,
        [
            (electorate_youth_chart_label, "선거연령청년구성비_추정"),
            ("30대", "30대구성비"),
            ("40대", "40대구성비"),
            ("50대", "50대구성비"),
            ("60대", "60대구성비"),
            ("70세 이상", "70세이상구성비"),
        ],
    )
    overview_chart_col1.plotly_chart(
        grouped_bar_chart(
            electorate_age_frame,
            x_col="항목",
            y_col="값",
            color_col="항목",
            title=f"{prefix} 선거연령대 구성비",
            percent=True,
            single_color=OVERVIEW_SNAPSHOT_COLORS["age"],
        ),
        use_container_width=True,
    )

    lifecycle_frame = _build_latest_snapshot_frame(
        latest_row,
        [
            ("아동(0~17세)", "아동구성비"),
            ("청소년(9~24세)", "청소년구성비"),
            ("청년(19~34세)", "청년구성비"),
            ("고령(65세 이상)", "고령구성비"),
        ],
    )
    overview_chart_col2.plotly_chart(
        grouped_bar_chart(
            lifecycle_frame,
            x_col="항목",
            y_col="값",
            color_col="항목",
            title=f"{prefix} 생애주기 구성비",
            percent=True,
            single_color=OVERVIEW_SNAPSHOT_COLORS["life"],
        ),
        use_container_width=True,
    )

    overview_chart_col3, overview_chart_col4 = st.columns(2)
    gender_frame = _build_latest_snapshot_frame(
        latest_row,
        [
            ("남성", "남성구성비"),
            ("여성", "여성구성비"),
        ],
    )
    overview_chart_col3.plotly_chart(
        grouped_bar_chart(
            gender_frame,
            x_col="항목",
            y_col="값",
            color_col="항목",
            title=f"{prefix} 성별 구성비",
            percent=True,
            single_color=OVERVIEW_SNAPSHOT_COLORS["gender"],
        ),
        use_container_width=True,
    )

    household_frame = _build_latest_snapshot_frame(
        latest_row,
        [
            ("1인가구 비율", "1인가구비율"),
            ("청년 1인가구 비율", "청년1인가구비율_추정"),
            ("노년 1인가구 비율", "노년1인가구비율_추정"),
        ],
    )
    overview_chart_col4.plotly_chart(
        grouped_bar_chart(
            household_frame,
            x_col="항목",
            y_col="값",
            color_col="항목",
            title=f"{prefix} 1인가구 구조",
            percent=True,
            single_color=OVERVIEW_SNAPSHOT_COLORS["household"],
        ),
        use_container_width=True,
    )

    if int(latest_row.get("1인가구연령통계가용여부", 0) or 0) == 0:
        st.caption("청년·노년 1인가구 지표는 행정안전부 1인가구 연령 통계가 제공되는 시점부터만 계산됩니다.")


def _resident_age_basis_markdown() -> str:
    return "\n".join(
        [
            "- `아동`: 0~17세",
            "- `청소년`: 9~24세",
            "- `청년`: 19~34세",
            "- `고령`: 65세 이상",
            "- `18/19~29세`: 선거연령이 19세인 선거는 19~29세, 18세인 선거는 18~29세",
            "- `선거연령대 구성비`: 선거별 선거연령 기준에 맞춰 18세 이상 또는 19세 이상 인구를 모수로 계산합니다.",
            "- `선거인 수`, `선거인 평균연령`: 확정선거인수 대신 주민등록 1세 단위 인구를 직접 합산해 계산합니다.",
            "- `청년 1인가구`, `노년 1인가구`: 각각 19~34세, 65세 이상 1인가구",
            "- 연령 기준 지표는 `1세 단위 원자료`를 직접 합산한 값입니다.",
        ]
    )


def _metric_group_from_metric(metric: str) -> str:
    for group_name, metrics in RESIDENT_METRIC_GROUPS.items():
        if metric in metrics:
            return group_name
    return next(iter(RESIDENT_METRIC_GROUPS))


def _render_resident_metric_group_picker(default_group: str) -> str:
    group_options = list(RESIDENT_METRIC_GROUPS.keys())
    default_index = group_options.index(default_group) if default_group in group_options else 0
    if st.session_state.get("resident_metric_group_selector") not in group_options:
        st.session_state["resident_metric_group_selector"] = group_options[default_index]
    if hasattr(st, "segmented_control"):
        return str(
            st.segmented_control(
                "지표 유형",
                options=group_options,
                default=group_options[default_index],
                key="resident_metric_group_selector",
            )
        )
    return str(
        st.radio(
            "지표 유형",
            options=group_options,
            index=default_index,
            horizontal=True,
            key="resident_metric_group_selector",
        )
    )


def _render_resident_metric_picker(group_name: str, default_metric: str) -> str:
    detail_options = RESIDENT_METRIC_GROUPS[group_name]
    if not detail_options:
        return default_metric
    default_value = default_metric if default_metric in detail_options else detail_options[0]
    if st.session_state.get("resident_metric_selector") not in detail_options:
        st.session_state["resident_metric_selector"] = default_value
    if hasattr(st, "pills"):
        value = st.pills(
            "세부 지표",
            options=detail_options,
            selection_mode="single",
            default=default_value,
            format_func=_resident_metric_label,
            key="resident_metric_selector",
        )
        return str(value or default_value)
    if hasattr(st, "segmented_control"):
        return str(
            st.segmented_control(
                "세부 지표",
                options=detail_options,
                default=default_value,
                format_func=_resident_metric_label,
                key="resident_metric_selector",
            )
        )
    return str(
        st.radio(
            "세부 지표",
            options=detail_options,
            index=detail_options.index(default_value),
            format_func=_resident_metric_label,
            key="resident_metric_selector",
        )
    )


def _unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def main() -> None:
    st.title("선거인수")

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
    resident_source = _prepare_resident_metadata(app_data["resident"])
    selected = render_sidebar_filters(filter_options, include_party=True, include_candidate=True)
    filtered_confirmed = apply_common_filters(app_data["confirmed"], selected)
    filtered_resident = apply_common_filters(resident_source, selected)

    if filtered_confirmed.empty:
        st.warning("현재 필터 조건에 맞는 확정선거인수 데이터가 없습니다.")
        st.stop()

    st.caption("현재 필터: " + build_breadcrumb_text(selected, filter_options))
    st.caption("선거인수 추이는 같은 선거시점의 하위 선거를 하나로 묶은 `대통령선거 / 국회의원선거 / 지방선거` 기준으로 표시합니다.")

    trend_df = _build_period_trend_frame(filtered_confirmed)
    if trend_df.empty:
        st.warning("확정선거인수 추이 데이터가 없습니다.")
        st.stop()

    latest_row = trend_df.sort_values(by=["선거시점"], ascending=[False], kind="stable").iloc[0]
    latest = get_latest_and_previous(trend_df.sort_values(by=["선거시점"], ascending=[False], kind="stable"), "확정선거인수")
    default_level = resolve_region_level(selected, default="구시군")

    region_level = st.selectbox(
        "비교 레벨",
        ["시도", "구시군", "읍면동"],
        index=["시도", "구시군", "읍면동"].index(default_level),
        key="electorate_level",
    )

    comparison_selected = build_region_comparison_selection(selected, region_level)
    comparison_confirmed = apply_common_filters(app_data["confirmed"], comparison_selected)
    inferred_sido_scope = _infer_sido_scope_for_gis(filtered_confirmed, selected)
    if region_level == "구시군" and inferred_sido_scope:
        comparison_confirmed = comparison_confirmed.loc[
            comparison_confirmed["시도명"].astype("string").isin(inferred_sido_scope)
        ].copy()
    focus_emd_gusigun_key = _resolve_focus_gusigun_key(filtered_confirmed if region_level == "읍면동" else comparison_confirmed.iloc[0:0].copy())
    if region_level == "읍면동" and focus_emd_gusigun_key and "구시군KEY" in comparison_confirmed.columns:
        comparison_confirmed = comparison_confirmed.loc[
            comparison_confirmed["구시군KEY"].astype("string") == str(focus_emd_gusigun_key)
        ].copy()
    focus_gusigun_key = _resolve_focus_gusigun_key(filtered_confirmed if region_level == "구시군" else comparison_confirmed.iloc[0:0].copy())
    comparison_gusigun_key = _resolve_focus_gusigun_key(comparison_confirmed if region_level == "읍면동" else comparison_confirmed.iloc[0:0].copy())
    region_trend_df = _build_region_period_trend_frame(comparison_confirmed, region_level)
    region_latest_period = _resolve_latest_period(region_trend_df, latest_row["선거시점"])
    region_df = _build_region_current_frame(region_trend_df, region_latest_period)

    resident_metric = str(st.session_state.get("resident_metric_selector", "선거인수비중"))
    if resident_metric not in RESIDENT_METRIC_OPTIONS:
        resident_metric = "선거인수비중"

    resident_trend_df = _build_resident_period_trend_frame(filtered_resident)
    resident_overview_source = apply_common_filters(resident_source, _resident_display_value_selection(selected))
    resident_overview_trend_df = _build_resident_period_trend_frame(resident_overview_source)
    resident_metric_state = _get_latest_metric_state(resident_trend_df, resident_metric)
    resident_latest_scope_row = _resolve_resident_overview_row(resident_trend_df, latest_row["선거시점"])
    resident_latest_population_row = (
        resident_overview_trend_df.sort_values(by=["선거시점"], ascending=[False], kind="stable").iloc[0]
        if not resident_overview_trend_df.empty
        else None
    )

    comparison_resident = apply_common_filters(resident_source, comparison_selected)
    resident_inferred_sido_scope = _infer_sido_scope_for_gis(filtered_resident, selected)
    if region_level == "구시군" and resident_inferred_sido_scope:
        comparison_resident = comparison_resident.loc[
            comparison_resident["시도명"].astype("string").isin(resident_inferred_sido_scope)
        ].copy()
    resident_focus_emd_gusigun_key = _resolve_focus_gusigun_key(filtered_resident if region_level == "읍면동" else comparison_resident.iloc[0:0].copy())
    if region_level == "읍면동" and resident_focus_emd_gusigun_key and "구시군KEY" in comparison_resident.columns:
        comparison_resident = comparison_resident.loc[
            comparison_resident["구시군KEY"].astype("string") == str(resident_focus_emd_gusigun_key)
        ].copy()
    resident_focus_gusigun_key = _resolve_focus_gusigun_key(filtered_resident if region_level == "구시군" else comparison_resident.iloc[0:0].copy())
    resident_comparison_gusigun_key = _resolve_focus_gusigun_key(comparison_resident if region_level == "읍면동" else comparison_resident.iloc[0:0].copy())
    resident_region_trend_df = _build_resident_region_period_trend_frame(comparison_resident, region_level)
    resident_region_latest_period = _resolve_latest_period(
        resident_region_trend_df,
        resident_metric_state["latest_row"]["선거시점"] if resident_metric_state["latest_row"] is not None else latest_row["선거시점"],
    )
    resident_region_df = _build_region_current_frame(resident_region_trend_df, resident_region_latest_period)
    if resident_metric in resident_region_df.columns:
        resident_region_df = resident_region_df.sort_values(
            by=[resident_metric, "지역"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["개요", "추이분석", "지역 비교", "주민등록인구 구성", "원자료"])

    with tab1:
        st.info(_build_count_delta_sentence(latest["current"], latest["delta"]))

        overview_col1, overview_col2, overview_col3 = st.columns(3)
        overview_col1.metric("최신 선거시점", str(latest_row["선거축라벨"]))
        overview_col2.metric("확정선거인수", format_int(latest_row["확정선거인수"]))
        overview_col3.metric(
            "직전 대비 증감",
            format_int(latest_row["확정선거인수"]),
            delta=None if pd.isna(latest["delta"]) else format_int(latest["delta"]),
        )

        if resident_latest_scope_row is not None:
            st.divider()
            st.subheader("주민등록인구 개요")
            _render_resident_overview_section(resident_latest_scope_row, prefix="최신")

    with tab2:
        st.caption("선거축은 선거시점 순서를 따르고, 축 라벨에는 `시점 + 선거명칭`을 표시합니다.")
        st.plotly_chart(
            line_metric_chart(
                trend_df,
                x_col="선거축라벨",
                y_col="확정선거인수",
                title="선거시점별 확정선거인수 추이",
            ),
            use_container_width=True,
        )
        st.dataframe(
            trend_df.loc[:, [column for column in ["선거축라벨", "확정선거인수"] if column in trend_df.columns]],
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        if comparison_selected != selected:
            st.caption(f"{region_level} 비교는 선택된 상위 지역 범위를 유지하고 하위 지역 필터는 풀어서 비교합니다.")
        if region_level == "구시군" and inferred_sido_scope:
            st.caption("구시군 GIS 히트맵은 현재 필터된 구시군이 속한 시도 범위까지만 표시합니다.")
        if region_level == "읍면동":
            if comparison_gusigun_key:
                st.caption("읍면동 비교는 현재 필터된 구시군 범위 안의 읍면동만 랭킹, GIS, 추이 범례에 표시합니다.")
            else:
                st.caption("단일 구시군이 선택되지 않아 현재 필터 범위 전체 읍면동을 비교합니다.")
        if region_latest_period != latest_row["선거시점"]:
            st.caption(f"지역 비교는 해당 레벨에서 실제 데이터가 있는 최신 시점인 {region_latest_period} 기준으로 표시합니다.")

        selection = st.plotly_chart(
            lollipop_chart(
                region_df,
                category_col="지역",
                value_col="확정선거인수",
                title=f"{region_level}별 확정선거인수 랭킹",
            ),
            use_container_width=True,
            key="electorate_region_rank_chart",
            on_select="rerun",
        )
        selected_region = _extract_selection_value(selection, ("y",))

        if region_level == "구시군" and "구시군KEY" in region_df.columns:
            geometry_context = load_geometry_context("구시군")
            st.caption("구시군 경계는 통계지리정보서비스(SGIS) 공개 자료에서 확인한 최신 공개 예시인 2024년 2분기 경계 계열 기준으로 표시합니다.")
            st.plotly_chart(
                build_region_choropleth(
                    region_df,
                    level="구시군",
                    key_col="구시군KEY",
                    value_col="확정선거인수",
                    title="구시군별 확정선거인수 GIS 히트맵",
                    geometry_context=geometry_context,
                    label_col="구시군명" if "구시군명" in region_df.columns else "지역",
                ),
                use_container_width=True,
            )
        elif region_level == "읍면동" and "읍면동KEY" in region_df.columns:
            geometry_context = load_geometry_context("읍면동")
            st.caption("읍면동 GIS 히트맵은 현재 필터 범위에서 집계된 읍면동 경계를 표시합니다.")
            st.plotly_chart(
                build_region_choropleth(
                    region_df,
                    level="읍면동",
                    key_col="읍면동KEY",
                    value_col="확정선거인수",
                    title="읍면동별 확정선거인수 GIS 히트맵",
                    geometry_context=geometry_context,
                    label_col="읍면동명" if "읍면동명" in region_df.columns else "지역",
                ),
                use_container_width=True,
            )
        else:
            heatmap_source = region_df.head(25).copy()
            heatmap_source["지표"] = "확정선거인수"
            st.plotly_chart(
                heatmap_chart(
                    heatmap_source,
                    x_col="지표",
                    y_col="지역",
                    value_col="확정선거인수",
                    title=f"{region_level}별 확정선거인수 heatmap",
                ),
                use_container_width=True,
            )

        compare_source = region_trend_df.copy()
        end_label_regions, end_label_text_map = _resolve_chart_end_labels(selected, region_df, preferred_level=region_level)
        legend_order = region_df["지역"].dropna().astype("string").tolist() if "지역" in region_df.columns else None
        compare_fig = line_metric_chart(
            compare_source,
            x_col="선거축라벨",
            y_col="확정선거인수",
            color_col="지역",
            title=f"{region_level} 다중 비교: 확정선거인수",
            legend_order=legend_order,
            end_label_entities=end_label_regions or None,
            end_label_text_map=end_label_text_map or None,
        )
        if region_level == "구시군" and focus_gusigun_key and "구시군KEY" in region_df.columns:
            focus_region_names = set(
                region_df.loc[
                    region_df["구시군KEY"].astype("string") == str(focus_gusigun_key),
                    "지역",
                ].dropna().astype("string").tolist()
            )
            selected_adjacent_regions = set(get_selected_adjacent_gusigun_names(selected))
            if not selected_adjacent_regions:
                adjacent_keys = set(get_adjacent_gusigun_keys(focus_gusigun_key))
                selected_adjacent_regions = set(
                    region_df.loc[
                        region_df["구시군KEY"].astype("string").isin(adjacent_keys),
                        "지역",
                    ].dropna().astype("string").tolist()
                )
            default_visible_regions = set(
                [*focus_region_names, *selected_adjacent_regions]
            )
            if default_visible_regions:
                for trace in compare_fig.data:
                    trace.visible = True if str(trace.name) in default_visible_regions else "legendonly"
                st.caption("다중 비교 범례에는 시도 내 모든 구시군이 표시되며, 처음에는 선택 구시군과 인접 구시군만 활성화합니다.")
        st.plotly_chart(compare_fig, use_container_width=True)

        if selected_region:
            st.caption(f"선택된 지역 drill-down: {selected_region}")
            region_detail = region_trend_df.loc[region_trend_df["지역"].astype("string") == selected_region].copy()
            st.plotly_chart(
                line_metric_chart(
                    region_detail,
                    x_col="선거축라벨",
                    y_col="확정선거인수",
                    title=f"{selected_region}의 선거시점별 확정선거인수",
                ),
                use_container_width=True,
            )
            st.dataframe(
                region_detail.loc[:, [column for column in ["선거축라벨", "지역", "확정선거인수"] if column in region_detail.columns]],
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        st.caption("주민등록인구 구성은 행정안전부 주민등록인구통계 기준월 데이터를 선거 기준월과 최신 기준월까지 함께 표시합니다.")
        if resident_trend_df.empty:
            st.info("현재 필터 조건에 맞는 주민등록인구 구성 데이터가 없습니다.")
        else:
            default_metric_group = _metric_group_from_metric(resident_metric)
            metric_picker_col1, metric_picker_col2 = st.columns([1.8, 2.2])
            with metric_picker_col1:
                resident_metric_group = _render_resident_metric_group_picker(default_metric_group)
            if resident_metric_group != "개요":
                with metric_picker_col2:
                    resident_metric = _render_resident_metric_picker(resident_metric_group, resident_metric)
            else:
                resident_overview_basis_source = resident_overview_trend_df if not resident_overview_trend_df.empty else resident_trend_df
                resident_overview_basis_options = (
                    resident_overview_basis_source.sort_values(
                        by=[column for column in ["선거시점", "선거KEY"] if column in resident_overview_basis_source.columns],
                        ascending=False,
                        kind="stable",
                    )
                    .reset_index(drop=True)
                )
                resident_overview_basis_option_ids = list(range(len(resident_overview_basis_options)))
                if st.session_state.get("resident_overview_basis_selector") not in resident_overview_basis_option_ids:
                    st.session_state["resident_overview_basis_selector"] = 0
                with metric_picker_col2:
                    resident_overview_basis_index = st.selectbox(
                        "표시 수치 기준",
                        resident_overview_basis_option_ids,
                        index=0,
                        format_func=lambda idx: (
                            f"{_resident_row_text(resident_overview_basis_options.iloc[idx], '기준월라벨')} "
                            f"({_resident_row_text(resident_overview_basis_options.iloc[idx], '선거축라벨')})"
                        ),
                        key="resident_overview_basis_selector",
                    )
                resident_latest_population_row = resident_overview_basis_options.iloc[resident_overview_basis_index]

            if resident_metric_group == "개요":
                st.subheader("주민등록인구 구성 변화 개요")
                reference_election_label = str(latest_row["선거축라벨"]) if "선거축라벨" in latest_row.index else "선택된 최근 선거"
                resident_overview_emd_trend_df = _build_resident_region_period_trend_frame(
                    resident_overview_source,
                    "읍면동",
                )
                st.caption(
                    "선택된 최근 선거 시점의 주민등록인구 구성을 기준으로, "
                    "선택한 주민등록 기준월의 총량·선거인·연령구간 지표를 표시합니다. "
                    "선거인 지표는 주민등록 인구 기준의 선거연령 이상 인구로 계산하며, 기본값은 최신 주민등록 기준월입니다."
                )
                resident_overview_change_frame = _render_resident_change_overview_section(
                    resident_latest_population_row,
                    resident_latest_scope_row,
                    reference_election_label,
                    resident_overview_emd_trend_df,
                )
                st.download_button(
                    "주민등록 구성 변화 개요 CSV 다운로드",
                    dataframe_to_csv_bytes(resident_overview_change_frame),
                    file_name="resident_composition_change_overview.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                resident_metric_state = _get_latest_metric_state(resident_trend_df, resident_metric)
                resident_latest_metric_row = resident_metric_state["latest_row"]
                latest_metric_label = (
                    str(resident_latest_metric_row["선거축라벨"])
                    if resident_latest_metric_row is not None and "선거축라벨" in resident_latest_metric_row.index
                    else "최신 시점"
                )
                resident_region_latest_period = _resolve_latest_period(
                    resident_region_trend_df,
                    resident_metric_state["latest_row"]["선거시점"] if resident_metric_state["latest_row"] is not None else latest_row["선거시점"],
                )
                resident_region_df = _build_region_current_frame(resident_region_trend_df, resident_region_latest_period)
                if resident_metric in resident_region_df.columns:
                    resident_region_df = resident_region_df.sort_values(
                        by=[resident_metric, "지역"],
                        ascending=[False, True],
                        kind="stable",
                    ).reset_index(drop=True)

                st.info(
                    _build_resident_metric_sentence(
                        resident_metric,
                        resident_metric_state["current"],
                        resident_metric_state["delta"],
                        latest_metric_label,
                    )
                )

                resident_subtab1, resident_subtab2, resident_subtab3, resident_subtab4 = st.tabs(
                    ["추이분석", "랭킹", "GIS", "다중 비교"]
                )

                with resident_subtab1:
                    st.plotly_chart(
                        line_metric_chart(
                            resident_trend_df,
                            x_col="선거축라벨",
                            y_col=resident_metric,
                            title=f"선거시점별 {_resident_metric_label(resident_metric)} 추이",
                            percent=_resident_metric_is_percent(resident_metric),
                        ),
                        use_container_width=True,
                    )
                    st.dataframe(
                        _resident_display_frame(
                            resident_trend_df,
                            _unique_columns(
                                [
                                    column
                                    for column in ["선거축라벨", resident_metric, "총인구수", "세대수", "1인가구수", "평균연령"]
                                    if column in resident_trend_df.columns
                                ]
                            ),
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                with resident_subtab2:
                    if comparison_selected != selected:
                        st.caption(f"{region_level} 비교는 선택된 상위 지역 범위를 유지하고 하위 지역 필터는 풀어서 비교합니다.")
                    if region_level == "읍면동":
                        if resident_comparison_gusigun_key:
                            st.caption("읍면동 비교는 현재 필터된 구시군 범위 안의 읍면동만 랭킹에 표시합니다.")
                        else:
                            st.caption("단일 구시군이 선택되지 않아 현재 필터 범위 전체 읍면동을 비교합니다.")
                    if resident_region_latest_period != (resident_latest_scope_row["선거시점"] if resident_latest_scope_row is not None else latest_row["선거시점"]):
                        st.caption(f"랭킹은 해당 레벨에서 실제 데이터가 있는 최신 시점인 {resident_region_latest_period} 기준입니다.")
                    if _resident_metric_requires_age_stats(resident_metric):
                        st.caption("청년·노년 1인가구 지표는 1인가구 연령 통계가 있는 선거시점만 표시됩니다.")

                    resident_selection = st.plotly_chart(
                        lollipop_chart(
                            resident_region_df,
                            category_col="지역",
                            value_col=resident_metric,
                            title=f"{region_level}별 {_resident_metric_label(resident_metric)} 랭킹",
                            percent=_resident_metric_is_percent(resident_metric),
                        ),
                        use_container_width=True,
                        key="resident_region_rank_chart",
                        on_select="rerun",
                    )
                    selected_resident_region = _extract_selection_value(resident_selection, ("y",))

                    st.dataframe(
                        _resident_display_frame(
                            resident_region_df,
                            _unique_columns(
                                [
                                    column
                                    for column in ["지역", resident_metric, "총인구수", "세대수", "1인가구수", "평균연령"]
                                    if column in resident_region_df.columns
                                ]
                            ),
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if selected_resident_region:
                        resident_region_detail = resident_region_trend_df.loc[
                            resident_region_trend_df["지역"].astype("string") == selected_resident_region
                        ].copy()
                        st.caption(f"선택된 지역 drill-down: {selected_resident_region}")
                        st.plotly_chart(
                            line_metric_chart(
                                resident_region_detail,
                                x_col="선거축라벨",
                                y_col=resident_metric,
                                title=f"{selected_resident_region}의 선거시점별 {_resident_metric_label(resident_metric)}",
                                percent=_resident_metric_is_percent(resident_metric),
                            ),
                            use_container_width=True,
                        )

                with resident_subtab3:
                    if region_level == "구시군" and resident_inferred_sido_scope:
                        st.caption("구시군 GIS 히트맵은 현재 필터된 구시군이 속한 시도 범위까지만 표시합니다.")
                    if region_level == "읍면동":
                        if resident_comparison_gusigun_key:
                            st.caption("읍면동 GIS는 현재 필터된 구시군 범위 안의 읍면동만 표시합니다.")
                        else:
                            st.caption("단일 구시군이 선택되지 않아 현재 필터 범위 전체 읍면동을 표시합니다.")
                    if region_level == "구시군" and "구시군KEY" in resident_region_df.columns:
                        geometry_context = load_geometry_context("구시군")
                        st.plotly_chart(
                            build_region_choropleth(
                                resident_region_df,
                                level="구시군",
                                key_col="구시군KEY",
                                value_col=resident_metric,
                                title=f"구시군별 {_resident_metric_label(resident_metric)} GIS 히트맵",
                                geometry_context=geometry_context,
                                percent=_resident_metric_is_percent(resident_metric),
                                label_col="구시군명" if "구시군명" in resident_region_df.columns else "지역",
                            ),
                            use_container_width=True,
                        )
                    elif region_level == "읍면동" and "읍면동KEY" in resident_region_df.columns:
                        geometry_context = load_geometry_context("읍면동")
                        st.plotly_chart(
                            build_region_choropleth(
                                resident_region_df,
                                level="읍면동",
                                key_col="읍면동KEY",
                                value_col=resident_metric,
                                title=f"읍면동별 {_resident_metric_label(resident_metric)} GIS 히트맵",
                                geometry_context=geometry_context,
                                percent=_resident_metric_is_percent(resident_metric),
                                label_col="읍면동명" if "읍면동명" in resident_region_df.columns else "지역",
                            ),
                            use_container_width=True,
                        )
                    else:
                        resident_heatmap = resident_region_df.head(25).copy()
                        resident_heatmap["지표"] = _resident_metric_label(resident_metric)
                        st.plotly_chart(
                            heatmap_chart(
                                resident_heatmap,
                                x_col="지표",
                                y_col="지역",
                                value_col=resident_metric,
                                title=f"{region_level}별 {_resident_metric_label(resident_metric)} heatmap",
                                percent=_resident_metric_is_percent(resident_metric),
                            ),
                            use_container_width=True,
                        )

                with resident_subtab4:
                    resident_end_label_regions, resident_end_label_text_map = _resolve_chart_end_labels(
                        selected,
                        resident_region_df,
                        preferred_level=region_level,
                    )
                    if comparison_selected != selected:
                        st.caption(f"{region_level} 비교는 선택된 상위 지역 범위를 유지하고 하위 지역 필터는 풀어서 비교합니다.")
                    resident_legend_order = resident_region_df["지역"].dropna().astype("string").tolist() if "지역" in resident_region_df.columns else None
                    resident_compare_fig = line_metric_chart(
                        resident_region_trend_df,
                        x_col="선거축라벨",
                        y_col=resident_metric,
                        color_col="지역",
                        title=f"{region_level} 다중 비교: {_resident_metric_label(resident_metric)}",
                        percent=_resident_metric_is_percent(resident_metric),
                        legend_order=resident_legend_order,
                        end_label_entities=resident_end_label_regions or None,
                        end_label_text_map=resident_end_label_text_map or None,
                    )
                    if region_level == "구시군" and resident_focus_gusigun_key and "구시군KEY" in resident_region_df.columns:
                        focus_region_names = set(
                            resident_region_df.loc[
                                resident_region_df["구시군KEY"].astype("string") == str(resident_focus_gusigun_key),
                                "지역",
                            ].dropna().astype("string").tolist()
                        )
                        selected_adjacent_regions = set(get_selected_adjacent_gusigun_names(selected))
                        if not selected_adjacent_regions:
                            adjacent_keys = set(get_adjacent_gusigun_keys(resident_focus_gusigun_key))
                            selected_adjacent_regions = set(
                                resident_region_df.loc[
                                    resident_region_df["구시군KEY"].astype("string").isin(adjacent_keys),
                                    "지역",
                                ].dropna().astype("string").tolist()
                            )
                        default_visible_regions = set(
                            [*focus_region_names, *selected_adjacent_regions]
                        )
                        if default_visible_regions:
                            for trace in resident_compare_fig.data:
                                trace.visible = True if str(trace.name) in default_visible_regions else "legendonly"
                            st.caption("주민등록인구 다중 비교 범례에는 시도 내 모든 구시군이 표시되며, 처음에는 선택 구시군과 인접 구시군만 활성화합니다.")
                    st.plotly_chart(resident_compare_fig, use_container_width=True)

                resident_download_col1, resident_download_col2, resident_download_col3 = st.columns(3)
                with resident_download_col1:
                    st.download_button(
                        "주민등록 추이 CSV 다운로드",
                        dataframe_to_csv_bytes(_resident_display_frame(resident_trend_df)),
                        file_name="resident_composition_trend.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with resident_download_col2:
                    st.download_button(
                        "주민등록 지역 비교 CSV 다운로드",
                        dataframe_to_csv_bytes(_resident_display_frame(resident_region_df)),
                        file_name="resident_composition_region_compare.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with resident_download_col3:
                    st.download_button(
                        "주민등록 지역 추이 CSV 다운로드",
                        dataframe_to_csv_bytes(_resident_display_frame(resident_region_trend_df)),
                        file_name="resident_composition_region_trend.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    with tab5:
        raw_tab1, raw_tab2 = st.tabs(["확정선거인수", "주민등록인구"])
        with raw_tab1:
            st.dataframe(
                region_df.sort_values(by="확정선거인수", ascending=False, kind="stable"),
                use_container_width=True,
                hide_index=True,
            )
        with raw_tab2:
            if resident_region_df.empty:
                st.info("표시할 주민등록인구 원자료가 없습니다.")
            else:
                resident_columns = [
                    column
                    for column in ["선거축라벨", "지역", resident_metric, "총인구수", "세대수", "1인가구수", "평균연령"]
                    if column in resident_region_df.columns
                ]
                st.dataframe(
                    _resident_display_frame(resident_region_df, _unique_columns(resident_columns)),
                    use_container_width=True,
                    hide_index=True,
                )
                st.dataframe(
                    _resident_display_frame(
                        resident_trend_df,
                        _unique_columns(
                            [
                                column
                                for column in ["선거축라벨", resident_metric, "총인구수", "세대수", "1인가구수", "평균연령"]
                                if column in resident_trend_df.columns
                            ]
                        ),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        st.download_button(
            "확정선거인수 추이 CSV 다운로드",
            dataframe_to_csv_bytes(trend_df),
            file_name="confirmed_electorate_trend.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "지역 비교 CSV 다운로드",
            dataframe_to_csv_bytes(region_df),
            file_name="confirmed_electorate_region_compare.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col3:
        st.download_button(
            "지역 추이 CSV 다운로드",
            dataframe_to_csv_bytes(region_trend_df),
            file_name="confirmed_electorate_region_trend.csv",
            mime="text/csv",
            use_container_width=True,
        )


main()
