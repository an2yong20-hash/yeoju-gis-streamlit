from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import Point

from src.loaders import (
    get_cache_file_signatures,
    load_cached_dims,
    load_fact_confirmed_electorate_enriched,
    load_fact_turnout_enriched,
    load_fact_votes_enriched,
)
from src.maps import empty_map_figure, load_geometry_context, resolve_basemap_style
from src.metrics import calc_polling_station_metrics, dataframe_to_csv_bytes

st.set_page_config(page_title="ElectionInfo BI", layout="wide", initial_sidebar_state="expanded")

CURRENT_ELECTION_KEY = "202506-P"
CURRENT_PERIOD = CURRENT_ELECTION_KEY[:6]
CURRENT_ELECTION_LABEL = "202506-P | 제21대 대선"
TARGET_SIDO = "경기도"
TARGET_GUSIGUN = "여주시"
POLLING_JURISDICTION_FILE = Path.home() / "Desktop" / "여주시 투표구 관할구역.xlsx"

PRIORITY_STYLES = {
    1: {"color": "#D64550", "label": "1순위"},
    2: {"color": "#F2A541", "label": "2순위"},
    3: {"color": "#2F80ED", "label": "3순위"},
    4: {"color": "#6F5BD5", "label": "4순위"},
}
BOUNDARY_FILL_COLOR = "rgba(234, 243, 240, 0)"
BOUNDARY_LINE_COLOR = "#245B5A"
VWORLD_BASE_TILE_URL = "https://xdworld.vworld.kr/2d/Base/service/{z}/{x}/{y}.png"
LEE_PHOTO_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Lee_Jae-myung%27s_Portrait_%282024.2%29_%28cropped%29.jpg?width=128"
KIM_PHOTO_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Kim_Moon-soo%27s_Portrait_%282025%29.png?width=128"
CANDIDATE_PAIR = ("이재명", "김문수")
ADVANCE_POLLING_LEGEND_NAME = "○ 사전투표소"
ADVANCE_POLLING_DATA_TRACE_NAME = "__advance_polling_places__"
ADVANCE_POLLING_EXPORT_TRACE_NAME = "__advance_polling_export_rings__"
ADVANCE_POLLING_MARKER_SIZE_PX = 21
ADVANCE_POLLING_RING_COLOR = "rgba(17, 24, 39, 0.78)"
ADVANCE_POLLING_RING_WIDTH = 2
POLLING_PLACE_COORDINATE_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    ("세종대왕면", "사전투표소"): (37.29709718, 127.5838806),
    ("세종대왕면", "제1투"): (37.297945, 127.572094),
    ("세종대왕면", "제2투"): (37.273870, 127.557580),
}
DEFAULT_POLLING_PLACE_COORDINATE_OVERRIDES = dict(POLLING_PLACE_COORDINATE_OVERRIDES)
HISTORICAL_SUMMARY_ELECTIONS = [
    {"key": "201806-L1", "label": "18도지사"},
    {"key": "202404-N1", "label": "24지역구"},
    {"key": "202506-P", "label": "25대선"},
]
DEMOCRATIC_PARTY_NAMES = {"더불어민주당"}
CONSERVATIVE_PARTY_NAMES = {"국민의힘", "자유한국당"}
HISTORICAL_EMD_NAME_RENAMES = {"능서면": "세종대왕면"}

GIS_DIM_COLUMNS = {
    "DimPollingPlace": [
        "투표소KEY",
        "선거시점",
        "시도명_F",
        "구시군명_F",
        "일반구명_F",
        "읍면동명_F",
        "투표소명_F",
        "장소명",
        "주소",
        "위도",
        "경도",
        "읍면동KEY_D",
    ],
    "DimDongAlias": [
        "시도명_F",
        "구시군명_F",
        "일반구명_F",
        "읍면동명_F",
        "시도명_D",
        "구시군명_D",
        "일반구명_D",
        "읍면동명_D",
        "읍면동KEY_F",
        "읍면동KEY_D",
    ],
}

TURNOUT_COLUMNS = (
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
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
)

CONFIRMED_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "읍면동KEY",
    "투표구명",
    "투표소KEY",
    "RowType",
    "확정선거인수",
)

VOTES_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "정당KEY",
    "정당명",
    "후보명",
    "후보라벨",
    "선거구명",
    "시도명",
    "구시군명",
    "구시군KEY",
    "일반구명",
    "읍면동명",
    "읍면동KEY",
    "투표소KEY",
    "구분",
    "RowType",
    "유효투표수",
    "득표수",
)

CUSTOM_GIS_AREAS: list[dict[str, Any]] = [
    {
        "id": "core_win",
        "title": "핵심 승리권역",
        "description": "중앙동, 오학동, 여흥동, 가남읍",
        "emd_names": ["중앙동", "오학동", "여흥동", "가남읍"],
        "priority_groups": [
            {
                "rank": 1,
                "profile": "권역 내 선거일 득표수 높음 / 사전투표율 높음",
                "places": [
                    ("중앙동", ["제2투", "제3투", "제5투"]),
                    ("오학동", ["제1투", "제2투", "제3투"]),
                    ("여흥동", ["제5투"]),
                    ("가남읍", ["제5투"]),
                ],
            },
            {
                "rank": 2,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 높음",
                "places": [
                    ("중앙동", ["제1투", "제4투"]),
                    ("여흥동", ["제1투", "제3투", "제6투"]),
                    ("가남읍", ["제1투", "제6투"]),
                ],
            },
            {
                "rank": 3,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 낮음",
                "places": [
                    ("여흥동", ["제2투", "제4투"]),
                    ("가남읍", ["제2투", "제3투", "제4투"]),
                ],
            },
        ],
    },
    {
        "id": "expanded_weak",
        "title": "확장형 열세권역",
        "description": "강천면, 점동면",
        "emd_names": ["강천면", "점동면"],
        "priority_groups": [
            {
                "rank": 1,
                "profile": "권역 내 선거일 득표수 높음 / 사전투표율 높음",
                "places": [("점동면", ["제1투"])],
            },
            {
                "rank": 2,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 높음",
                "places": [
                    ("점동면", ["제3투"]),
                    ("강천면", ["제1투", "제2투", "제3투", "제4투"]),
                ],
            },
            {
                "rank": 3,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 낮음",
                "places": [("점동면", ["제2투"])],
            },
        ],
    },
    {
        "id": "loss_control",
        "title": "손실관리지역",
        "description": "세종대왕면, 대신면, 북내면, 흥천면",
        "emd_names": ["세종대왕면", "대신면", "북내면", "흥천면"],
        "priority_groups": [
            {
                "rank": 1,
                "profile": "권역 내 선거일 득표수 높음 / 사전투표율 높음",
                "places": [
                    ("북내면", ["제1투"]),
                    ("흥천면", ["제1투"]),
                ],
            },
            {
                "rank": 2,
                "profile": "권역 내 선거일 득표수 높음 / 사전투표율 낮음",
                "places": [
                    ("세종대왕면", ["제1투", "제2투"]),
                    ("대신면", ["제2투"]),
                    ("흥천면", ["제2투"]),
                ],
            },
            {
                "rank": 3,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 높음",
                "places": [
                    ("대신면", ["제1투", "제4투"]),
                    ("북내면", ["제2투"]),
                ],
            },
            {
                "rank": 4,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 낮음",
                "places": [
                    ("세종대왕면", ["제3투"]),
                    ("대신면", ["제3투"]),
                    ("북내면", ["제3투"]),
                ],
            },
        ],
    },
    {
        "id": "low_efficiency",
        "title": "저효율 관리지역",
        "description": "금사면, 산북면",
        "emd_names": ["금사면", "산북면"],
        "priority_groups": [
            {
                "rank": 1,
                "profile": "권역 내 선거일 득표수 높음 / 사전투표율 높음",
                "places": [("산북면", ["투표소"])],
            },
            {
                "rank": 2,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 높음",
                "places": [("금사면", ["제1투"])],
            },
            {
                "rank": 3,
                "profile": "권역 내 선거일 득표수 낮음 / 사전투표율 낮음",
                "places": [("금사면", ["제2투"])],
            },
        ],
    },
]


def _load_bundle() -> dict[str, pd.DataFrame]:
    cache_signature = get_cache_file_signatures()
    dims = load_cached_dims(cache_signature, dim_names=("DimPollingPlace", "DimDongAlias"), columns_map=GIS_DIM_COLUMNS)
    return {
        "polling": dims["DimPollingPlace"],
        "dong_alias": dims["DimDongAlias"],
        "confirmed": load_fact_confirmed_electorate_enriched(cache_signature, CONFIRMED_COLUMNS),
        "turnout": load_fact_turnout_enriched(cache_signature, TURNOUT_COLUMNS),
        "votes": load_fact_votes_enriched(cache_signature, VOTES_COLUMNS),
    }


@st.cache_data(show_spinner=False)
def _read_polling_jurisdiction_frame(path_text: str, modified_time: float) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame(columns=["투표소명_F", "관할구역"])

    frame = pd.read_excel(path, dtype=str)
    required_columns = {"투표구명", "관할구역"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return pd.DataFrame(columns=["투표소명_F", "관할구역"])

    result = frame.loc[:, ["투표구명", "관할구역"]].copy()
    result["투표소명_F"] = result["투표구명"].astype("string").str.strip()
    result["관할구역"] = result["관할구역"].astype("string").str.strip()
    result = result.dropna(subset=["투표소명_F"])
    result = result.loc[result["투표소명_F"].ne("")]
    result["관할구역"] = result["관할구역"].fillna("-").replace("", "-")
    return result.loc[:, ["투표소명_F", "관할구역"]].drop_duplicates(subset=["투표소명_F"], keep="last").reset_index(drop=True)


def _load_polling_jurisdiction_frame() -> pd.DataFrame:
    if not POLLING_JURISDICTION_FILE.exists():
        return pd.DataFrame(columns=["투표소명_F", "관할구역"])
    return _read_polling_jurisdiction_frame(str(POLLING_JURISDICTION_FILE), POLLING_JURISDICTION_FILE.stat().st_mtime)


def _current_election_frame(df: pd.DataFrame, gusigun_col: str = "구시군명") -> pd.DataFrame:
    if df.empty or "선거KEY" not in df.columns or gusigun_col not in df.columns:
        return df.iloc[0:0].copy()
    mask = df["선거KEY"].astype("string").eq(CURRENT_ELECTION_KEY) & df[gusigun_col].astype("string").eq(TARGET_GUSIGUN)
    if "시도명" in df.columns:
        mask = mask & df["시도명"].astype("string").eq(TARGET_SIDO)
    return df.loc[mask].copy()


def _current_polling_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    return df.loc[
        df["선거시점"].astype("string").str.startswith(CURRENT_PERIOD, na=False)
        & df["시도명_F"].astype("string").eq(TARGET_SIDO)
        & df["구시군명_F"].astype("string").eq(TARGET_GUSIGUN)
    ].copy()


def _build_priority_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for area in CUSTOM_GIS_AREAS:
        for group in area["priority_groups"]:
            for emd_name, suffixes in group["places"]:
                for suffix in suffixes:
                    rows.append(
                        {
                            "권역ID": area["id"],
                            "권역": area["title"],
                            "우선순위": int(group["rank"]),
                            "우선순위라벨": PRIORITY_STYLES[int(group["rank"])]["label"],
                            "성격": group["profile"],
                            "읍면동명_F": emd_name,
                            "투표소명_F": f"{emd_name}{suffix}",
                        }
                    )
    return pd.DataFrame(rows)


def _priority_summary_frame(area: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group in area["priority_groups"]:
        regions = " / ".join(
            f"{emd_name} {', '.join(suffixes)}"
            for emd_name, suffixes in group["places"]
        )
        rows.append(
            {
                "구분": f"{group['rank']}순위",
                "지역": regions,
                "성격": group["profile"],
            }
        )
    return pd.DataFrame(rows)


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def _format_percentage_point(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:+.1f}%p"


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def _historical_emd_order() -> dict[str, int]:
    order: dict[str, int] = {}
    for area in CUSTOM_GIS_AREAS:
        for emd_name in area["emd_names"]:
            if emd_name not in order:
                order[emd_name] = len(order)
    return order


def _normalize_historical_emd_names(series: pd.Series) -> pd.Series:
    return series.astype("string").replace(HISTORICAL_EMD_NAME_RENAMES)


def _historical_vote_rate_frame(votes_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "선거KEY",
        "읍면동명",
        "유효투표수",
        "민주당득표수",
        "국민의힘득표수",
        "민주당득표율",
        "국민의힘득표율",
        "격차",
    ]
    required = {"선거KEY", "구시군명", "읍면동명", "RowType", "정당명", "유효투표수", "득표수"}
    if votes_df.empty or not required.issubset(votes_df.columns):
        return pd.DataFrame(columns=output_columns)

    target_keys = {spec["key"] for spec in HISTORICAL_SUMMARY_ELECTIONS}
    base = votes_df.loc[
        votes_df["선거KEY"].astype("string").isin(target_keys)
        & votes_df["시도명"].astype("string").eq(TARGET_SIDO)
        & votes_df["구시군명"].astype("string").eq(TARGET_GUSIGUN)
        & votes_df["RowType"].astype("string").eq("읍면동")
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)

    base["읍면동명"] = _normalize_historical_emd_names(base["읍면동명"])
    base["정당계열"] = pd.NA
    base.loc[base["정당명"].astype("string").isin(DEMOCRATIC_PARTY_NAMES), "정당계열"] = "민주당"
    base.loc[base["정당명"].astype("string").isin(CONSERVATIVE_PARTY_NAMES), "정당계열"] = "국민의힘"
    base = base.loc[base["정당계열"].notna()].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)

    base["득표수"] = pd.to_numeric(base["득표수"], errors="coerce")
    base["유효투표수"] = pd.to_numeric(base["유효투표수"], errors="coerce")
    valid_votes = (
        base.groupby(["선거KEY", "읍면동명"], as_index=False, observed=True)["유효투표수"]
        .max()
        .rename(columns={"유효투표수": "유효투표수"})
    )
    party_votes = (
        base.groupby(["선거KEY", "읍면동명", "정당계열"], observed=True)["득표수"]
        .sum(min_count=1)
        .unstack("정당계열")
        .reset_index()
        .rename(columns={"민주당": "민주당득표수", "국민의힘": "국민의힘득표수"})
    )
    result = valid_votes.merge(party_votes, on=["선거KEY", "읍면동명"], how="left", copy=False)
    for column in ["민주당득표수", "국민의힘득표수"]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["민주당득표율"] = _safe_divide(result["민주당득표수"], result["유효투표수"])
    result["국민의힘득표율"] = _safe_divide(result["국민의힘득표수"], result["유효투표수"])
    result["격차"] = result["민주당득표율"] - result["국민의힘득표율"]
    return result.loc[:, output_columns]


def _flow_label(value: object) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    side = "민주" if value >= 0 else "국힘"
    return f"{side} +{abs(value) * 100:.1f}%p"


def _historical_summary_row(label: str, base: pd.DataFrame, extra: dict[str, object] | None = None) -> dict[str, object]:
    row: dict[str, object] = dict(extra or {})
    row["구분"] = label
    dem_rates: list[float] = []
    conservative_rates: list[float] = []
    gaps: list[float] = []
    for spec in HISTORICAL_SUMMARY_ELECTIONS:
        election = base.loc[base["선거KEY"].astype("string").eq(spec["key"])]
        if election.empty:
            dem_rate = float("nan")
            conservative_rate = float("nan")
            gap = float("nan")
        else:
            valid_votes = pd.to_numeric(election["유효투표수"], errors="coerce").sum(min_count=1)
            dem_votes = pd.to_numeric(election["민주당득표수"], errors="coerce").sum(min_count=1)
            conservative_votes = pd.to_numeric(election["국민의힘득표수"], errors="coerce").sum(min_count=1)
            dem_rate = dem_votes / valid_votes if pd.notna(valid_votes) and valid_votes else float("nan")
            conservative_rate = conservative_votes / valid_votes if pd.notna(valid_votes) and valid_votes else float("nan")
            gap = dem_rate - conservative_rate if pd.notna(dem_rate) and pd.notna(conservative_rate) else float("nan")
        row[f"{spec['label']} 민주"] = dem_rate
        row[f"{spec['label']} 국힘"] = conservative_rate
        dem_rates.append(dem_rate)
        conservative_rates.append(conservative_rate)
        gaps.append(gap)

    row["민주당 평균"] = pd.Series(dem_rates, dtype="float64").mean(skipna=True)
    row["국민의힘 평균"] = pd.Series(conservative_rates, dtype="float64").mean(skipna=True)
    row["평균 격차"] = pd.Series(gaps, dtype="float64").mean(skipna=True)
    gap_by_key = {
        spec["key"]: row[f"{spec['label']} 민주"] - row[f"{spec['label']} 국힘"]
        for spec in HISTORICAL_SUMMARY_ELECTIONS
        if pd.notna(row[f"{spec['label']} 민주"]) and pd.notna(row[f"{spec['label']} 국힘"])
    }
    recent_shift = (
        gap_by_key["202506-P"] - gap_by_key["202404-N1"]
        if "202506-P" in gap_by_key and "202404-N1" in gap_by_key
        else float("nan")
    )
    row["최근 흐름"] = recent_shift
    row["최근 흐름 요약"] = _flow_label(recent_shift)
    return row


def _build_historical_strategy_summary(votes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vote_rates = _historical_vote_rate_frame(votes_df)
    if vote_rates.empty:
        return pd.DataFrame(), pd.DataFrame()

    area_rows: list[dict[str, object]] = []
    emd_rows: list[dict[str, object]] = []
    emd_order = _historical_emd_order()
    for area_index, area in enumerate(CUSTOM_GIS_AREAS, start=1):
        area_emd_names = list(area["emd_names"])
        area_base = vote_rates.loc[vote_rates["읍면동명"].astype("string").isin(area_emd_names)].copy()
        area_rows.append(
            _historical_summary_row(
                str(area["title"]),
                area_base,
                {"대상 읍면동": ", ".join(area_emd_names), "__order": area_index},
            )
        )
        for emd_name in area_emd_names:
            emd_base = vote_rates.loc[vote_rates["읍면동명"].astype("string").eq(emd_name)].copy()
            emd_rows.append(
                _historical_summary_row(
                    emd_name,
                    emd_base,
                    {"권역": area["title"], "__order": emd_order.get(emd_name, 99)},
                )
            )

    def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
        leading = [column for column in ["권역", "구분", "대상 읍면동"] if column in frame.columns]
        trailing = [column for column in ["민주당 평균", "국민의힘 평균", "평균 격차", "최근 흐름", "최근 흐름 요약"] if column in frame.columns]
        middle = [column for column in frame.columns if column not in {*leading, *trailing}]
        return frame.loc[:, [*leading, *middle, *trailing]]

    area_summary = pd.DataFrame(area_rows).sort_values("__order", kind="stable").drop(columns="__order", errors="ignore")
    emd_summary = pd.DataFrame(emd_rows).sort_values(["__order", "권역", "구분"], kind="stable").drop(columns="__order", errors="ignore")
    area_summary = _order_columns(area_summary)
    emd_summary = _order_columns(emd_summary)
    return area_summary.reset_index(drop=True), emd_summary.reset_index(drop=True)


def _style_historical_summary(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    percent_columns = [
        column
        for column in frame.columns
        if column.endswith("민주")
        or column.endswith("국힘")
        or column in {"민주당 평균", "국민의힘 평균", "평균 격차", "최근 흐름"}
    ]

    def _gap_style(value: object) -> str:
        if pd.isna(value):
            return ""
        value = float(value)
        if value >= 0:
            return "color: #1f6f8b; font-weight: 700;"
        return "color: #d00000; font-weight: 700;"

    def _flow_style(value: object) -> str:
        text = str(value)
        if text.startswith("민주"):
            return "background-color: #1f6f8b; color: #ffffff; font-weight: 800;"
        if text.startswith("국힘"):
            return "background-color: #d00000; color: #ffffff; font-weight: 800;"
        return ""

    styler = frame.style.format({column: "{:.1%}" for column in percent_columns}, na_rep="-")
    for column in [col for col in frame.columns if col.endswith("민주") or col == "민주당 평균"]:
        styler = styler.map(lambda value: "color: #1f6f8b; font-weight: 700;" if pd.notna(value) else "", subset=[column])
    for column in [col for col in frame.columns if col.endswith("국힘") or col == "국민의힘 평균"]:
        styler = styler.map(lambda value: "color: #d00000; font-weight: 700;" if pd.notna(value) else "", subset=[column])
    for column in [col for col in ["평균 격차", "최근 흐름"] if col in frame.columns]:
        styler = styler.map(_gap_style, subset=[column])
    if "최근 흐름 요약" in frame.columns:
        styler = styler.map(_flow_style, subset=["최근 흐름 요약"])
    return styler


def _render_historical_strategy_summary(votes_df: pd.DataFrame) -> None:
    area_summary, emd_summary = _build_historical_strategy_summary(votes_df)
    if area_summary.empty or emd_summary.empty:
        st.info("권역별 과거 선거 득표율 요약을 계산할 수 없습니다.")
        return

    st.subheader("권역별 전략 구분")
    st.caption("자유한국당은 국민의힘 계열로 묶고, 읍면동 명칭 변경은 DimDongAlias 기준으로 현재 명칭에 맞춥니다. 최근 흐름은 2024년 지역구선거 대비 2025년 대통령선거의 양당 격차 변화입니다.")
    st.dataframe(_style_historical_summary(area_summary.drop(columns="최근 흐름", errors="ignore")), use_container_width=True, hide_index=True)

    st.subheader("읍면동별 득표율 요약")
    st.dataframe(_style_historical_summary(emd_summary.drop(columns="최근 흐름", errors="ignore")), use_container_width=True, hide_index=True)


def _safe_divide(numerator: object, denominator: object) -> object:
    numerator_series = pd.to_numeric(numerator, errors="coerce")
    denominator_series = pd.to_numeric(denominator, errors="coerce")
    return numerator_series.divide(denominator_series.where(denominator_series.ne(0)))


def _add_candidate_metrics(metrics: pd.DataFrame, votes_df: pd.DataFrame, candidate_name: str = "이재명") -> pd.DataFrame:
    if metrics.empty or votes_df.empty:
        result = metrics.copy()
        result[f"{candidate_name}득표수"] = pd.NA
        result[f"{candidate_name}득표율"] = pd.NA
        return result

    candidate_votes = votes_df.loc[
        votes_df["RowType"].astype("string").eq("투표소")
        & votes_df["후보명"].astype("string").eq(candidate_name)
        & votes_df["투표소KEY"].notna()
    ].copy()
    if candidate_votes.empty:
        result = metrics.copy()
        result[f"{candidate_name}득표수"] = pd.NA
        result[f"{candidate_name}득표율"] = pd.NA
        return result

    grouped = (
        candidate_votes.groupby("투표소KEY", as_index=False, observed=True)[["득표수", "유효투표수"]]
        .sum(min_count=1)
        .rename(columns={"득표수": f"{candidate_name}득표수", "유효투표수": f"{candidate_name}유효투표수"})
    )
    grouped[f"{candidate_name}득표율"] = grouped[f"{candidate_name}득표수"].divide(
        grouped[f"{candidate_name}유효투표수"].where(grouped[f"{candidate_name}유효투표수"].ne(0))
    )
    return metrics.merge(grouped, on="투표소KEY", how="left", copy=False)


def _build_candidate_pair_metrics(votes_df: pd.DataFrame, group_col: str, rowtype: str, prefix: str) -> pd.DataFrame:
    output_columns = [
        group_col,
        f"{prefix}_유효투표수",
        *[
            column
            for candidate_name in CANDIDATE_PAIR
            for column in [f"{prefix}_{candidate_name}득표수", f"{prefix}_{candidate_name}득표율"]
        ],
    ]
    if votes_df.empty or group_col not in votes_df.columns:
        return pd.DataFrame(columns=output_columns)

    base = votes_df.loc[
        votes_df["RowType"].astype("string").eq(rowtype)
        & votes_df["후보명"].astype("string").isin(CANDIDATE_PAIR)
        & votes_df[group_col].notna()
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)

    candidate_votes = (
        base.groupby([group_col, "후보명"], as_index=False, observed=True)["득표수"]
        .sum(min_count=1)
        .pivot(index=group_col, columns="후보명", values="득표수")
    )
    if rowtype == "투표소" and group_col != "투표소KEY" and "투표소KEY" in base.columns:
        valid_source = base.dropna(subset=["투표소KEY"]).drop_duplicates(subset=[group_col, "투표소KEY"])
        valid_votes = pd.to_numeric(valid_source["유효투표수"], errors="coerce").groupby(valid_source[group_col], observed=True).sum(min_count=1)
    else:
        valid_votes = pd.to_numeric(base["유효투표수"], errors="coerce").groupby(base[group_col], observed=True).max()
    result = pd.DataFrame({group_col: valid_votes.index, f"{prefix}_유효투표수": valid_votes.to_numpy()})
    for candidate_name in CANDIDATE_PAIR:
        votes = candidate_votes[candidate_name] if candidate_name in candidate_votes.columns else pd.Series(pd.NA, index=valid_votes.index)
        votes = votes.reindex(valid_votes.index)
        result[f"{prefix}_{candidate_name}득표수"] = votes.to_numpy()
        result[f"{prefix}_{candidate_name}득표율"] = votes.divide(valid_votes.where(valid_votes.ne(0))).to_numpy()
    return result.loc[:, output_columns]


def _build_total_candidate_pair_metrics(votes_df: pd.DataFrame, rowtype: str, prefix: str) -> pd.DataFrame:
    output_columns = [
        f"{prefix}_유효투표수",
        *[
            column
            for candidate_name in CANDIDATE_PAIR
            for column in [f"{prefix}_{candidate_name}득표수", f"{prefix}_{candidate_name}득표율"]
        ],
    ]
    if votes_df.empty:
        return pd.DataFrame(columns=output_columns)

    base = votes_df.loc[
        votes_df["RowType"].astype("string").eq(rowtype)
        & votes_df["후보명"].astype("string").isin(CANDIDATE_PAIR)
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)

    valid_votes = pd.to_numeric(base["유효투표수"], errors="coerce").max()
    row: dict[str, object] = {f"{prefix}_유효투표수": valid_votes}
    candidate_votes = base.groupby("후보명", observed=True)["득표수"].sum(min_count=1)
    for candidate_name in CANDIDATE_PAIR:
        votes = pd.to_numeric(pd.Series([candidate_votes.get(candidate_name, pd.NA)]), errors="coerce").iloc[0]
        row[f"{prefix}_{candidate_name}득표수"] = votes
        row[f"{prefix}_{candidate_name}득표율"] = votes / valid_votes if pd.notna(votes) and pd.notna(valid_votes) and valid_votes else pd.NA
    return pd.DataFrame([row], columns=output_columns)


def _build_emd_turnout_summary_metrics(turnout_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["읍면동KEY", "읍면동실제_투표수"]
    if turnout_df.empty or not {"읍면동KEY", "RowType", "투표수"}.issubset(turnout_df.columns):
        return pd.DataFrame(columns=output_columns)

    base = turnout_df.loc[
        turnout_df["RowType"].astype("string").eq("읍면동")
        & turnout_df["읍면동KEY"].notna()
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)
    base["투표수"] = pd.to_numeric(base["투표수"], errors="coerce")
    return (
        base.groupby("읍면동KEY", as_index=False, observed=True)["투표수"]
        .sum(min_count=1)
        .rename(columns={"투표수": "읍면동실제_투표수"})
        .loc[:, output_columns]
    )


def _build_total_turnout_summary_metrics(turnout_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["전체실제_투표수"]
    if turnout_df.empty or not {"RowType", "투표수"}.issubset(turnout_df.columns):
        return pd.DataFrame(columns=output_columns)

    base = turnout_df.loc[turnout_df["RowType"].astype("string").eq("합계")].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)
    total_turnout = pd.to_numeric(base["투표수"], errors="coerce").sum(min_count=1)
    return pd.DataFrame([{"전체실제_투표수": total_turnout}], columns=output_columns)


def _build_confirmed_polling_metrics(confirmed_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["투표소KEY", "확정선거인수"]
    if confirmed_df.empty or not {"투표소KEY", "확정선거인수", "RowType"}.issubset(confirmed_df.columns):
        return pd.DataFrame(columns=output_columns)

    base = confirmed_df.loc[
        confirmed_df["RowType"].astype("string").eq("투표소")
        & confirmed_df["투표소KEY"].notna()
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)
    base["확정선거인수"] = pd.to_numeric(base["확정선거인수"], errors="coerce")
    return (
        base.groupby("투표소KEY", as_index=False, observed=True)["확정선거인수"]
        .sum(min_count=1)
        .loc[:, output_columns]
    )


def _build_local_early_turnout_metrics(turnout_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["읍면동KEY", "관내사전_투표수"]
    if turnout_df.empty or not {"읍면동KEY", "RowType", "투표수"}.issubset(turnout_df.columns):
        return pd.DataFrame(columns=output_columns)

    base = turnout_df.loc[
        turnout_df["RowType"].astype("string").eq("관내사전투표")
        & turnout_df["읍면동KEY"].notna()
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=output_columns)
    base["투표수"] = pd.to_numeric(base["투표수"], errors="coerce")
    return (
        base.groupby("읍면동KEY", as_index=False, observed=True)["투표수"]
        .sum(min_count=1)
        .rename(columns={"투표수": "관내사전_투표수"})
        .loc[:, output_columns]
    )


def _build_emd_model_turnout_metrics(turnout_df: pd.DataFrame, confirmed_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "읍면동KEY",
        "읍면동_확정선거인수",
        "읍면동_선거일선거인수",
        "읍면동_선거일투표수",
        "읍면동_선거일투표율",
        "읍면동_관내사전투표수",
        "읍면동_관내사전투표율",
    ]
    if (
        turnout_df.empty
        or confirmed_df.empty
        or not {"읍면동KEY", "RowType", "선거인수", "투표수"}.issubset(turnout_df.columns)
        or not {"읍면동KEY", "RowType", "확정선거인수"}.issubset(confirmed_df.columns)
    ):
        return pd.DataFrame(columns=output_columns)

    confirmed = confirmed_df.loc[
        confirmed_df["RowType"].astype("string").eq("읍면동")
        & confirmed_df["읍면동KEY"].notna()
    ].copy()
    if confirmed.empty:
        return pd.DataFrame(columns=output_columns)
    confirmed["확정선거인수"] = pd.to_numeric(confirmed["확정선거인수"], errors="coerce")
    confirmed = (
        confirmed.groupby("읍면동KEY", as_index=False, observed=True)["확정선거인수"]
        .sum(min_count=1)
        .rename(columns={"확정선거인수": "읍면동_확정선거인수"})
    )

    day = turnout_df.loc[
        turnout_df["RowType"].astype("string").eq("선거일투표")
        & turnout_df["읍면동KEY"].notna()
    ].copy()
    day["선거인수"] = pd.to_numeric(day["선거인수"], errors="coerce")
    day["투표수"] = pd.to_numeric(day["투표수"], errors="coerce")
    day = (
        day.groupby("읍면동KEY", as_index=False, observed=True)[["선거인수", "투표수"]]
        .sum(min_count=1)
        .rename(columns={"선거인수": "읍면동_선거일선거인수", "투표수": "읍면동_선거일투표수"})
    )

    local_early = _build_local_early_turnout_metrics(turnout_df)
    local_early = local_early.rename(columns={"관내사전_투표수": "읍면동_관내사전투표수"})
    result = confirmed.merge(day, on="읍면동KEY", how="left", copy=False)
    result = result.merge(local_early, on="읍면동KEY", how="left", copy=False)
    result["읍면동_선거일투표율"] = _safe_divide(result["읍면동_선거일투표수"], result["읍면동_확정선거인수"])
    result["읍면동_관내사전투표율"] = _safe_divide(result["읍면동_관내사전투표수"], result["읍면동_확정선거인수"])
    return result.loc[:, output_columns]


def _add_turnout_unit_metrics(metrics: pd.DataFrame, confirmed_df: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    confirmed = _build_confirmed_polling_metrics(confirmed_df)
    if confirmed.empty:
        result["확정선거인수"] = pd.NA
    else:
        result = result.merge(confirmed, on="투표소KEY", how="left", copy=False)

    result["개표결과선거인수"] = pd.to_numeric(result.get("선거인수"), errors="coerce")
    result["선거일투표수"] = pd.to_numeric(result.get("투표수"), errors="coerce")
    result["확정선거인수"] = pd.to_numeric(result.get("확정선거인수"), errors="coerce")
    result["사전투표수"] = (result["확정선거인수"] - result["개표결과선거인수"]).clip(lower=0)
    result["선거일투표율"] = _safe_divide(result["선거일투표수"], result["확정선거인수"])
    result["사전투표율"] = _safe_divide(result["사전투표수"], result["확정선거인수"])
    result["사전투표비중_투표구"] = _safe_divide(
        result["사전투표율"],
        result["선거일투표율"].fillna(0) + result["사전투표율"].fillna(0),
    )
    return result


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _estimation_model_specs() -> list[dict[str, object]]:
    return [
        {"model": "A", "target_label": "이재명", "target": "early_lee", "features": ["day_lee"], "role": "후보별"},
        {"model": "B", "target_label": "이재명", "target": "early_lee", "features": ["day_lee", "turnout_ratio"], "role": "후보별 최종"},
        {
            "model": "C",
            "target_label": "이재명",
            "target": "early_lee",
            "features": ["day_lee", "early_turnout", "electionday_turnout"],
            "role": "후보별 보조",
        },
        {"model": "A", "target_label": "김문수", "target": "early_kim", "features": ["day_kim"], "role": "후보별"},
        {"model": "B", "target_label": "김문수", "target": "early_kim", "features": ["day_kim", "turnout_ratio"], "role": "후보별 최종"},
        {
            "model": "C",
            "target_label": "김문수",
            "target": "early_kim",
            "features": ["day_kim", "early_turnout", "electionday_turnout"],
            "role": "후보별 보조",
        },
        {"model": "D", "target_label": "양강 마진", "target": "early_margin", "features": ["day_margin", "turnout_ratio"], "role": "마진 병행"},
    ]


def _model_feature_label(feature: str) -> str:
    labels = {
        "day_lee": "선거일 이재명 득표율",
        "day_kim": "선거일 김문수 득표율",
        "turnout_ratio": "사전투표율/선거일투표율",
        "early_turnout": "사전투표율",
        "electionday_turnout": "선거일투표율",
        "day_margin": "선거일 양강 마진",
    }
    return labels.get(feature, feature)


def _build_estimation_training_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "읍면동",
        "day_lee",
        "day_kim",
        "early_lee",
        "early_kim",
        "early_turnout",
        "electionday_turnout",
        "turnout_ratio",
        "day_margin",
        "early_margin",
    ]
    if metrics.empty or "읍면동KEY_D" not in metrics.columns:
        return pd.DataFrame(columns=columns)

    base = metrics.drop_duplicates(subset=["읍면동KEY_D"]).copy()
    frame = pd.DataFrame(index=base.index)
    frame["읍면동"] = base["읍면동명_F"].astype("string") if "읍면동명_F" in base.columns else base["읍면동KEY_D"].astype("string")
    frame["day_lee"] = _numeric_series(base, "선거일읍면동_이재명득표율")
    frame["day_kim"] = _numeric_series(base, "선거일읍면동_김문수득표율")
    frame["early_lee"] = _numeric_series(base, "관내사전_이재명득표율")
    frame["early_kim"] = _numeric_series(base, "관내사전_김문수득표율")
    frame["early_turnout"] = _numeric_series(base, "읍면동_관내사전투표율")
    frame["electionday_turnout"] = _numeric_series(base, "읍면동_선거일투표율")
    frame["turnout_ratio"] = _safe_divide(frame["early_turnout"], frame["electionday_turnout"])
    frame["day_margin"] = frame["day_lee"] - frame["day_kim"]
    frame["early_margin"] = frame["early_lee"] - frame["early_kim"]
    return frame.loc[:, columns]


def _build_estimation_scoring_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=metrics.index)
    frame["day_lee"] = _numeric_series(metrics, "선거일_이재명득표율")
    frame["day_kim"] = _numeric_series(metrics, "선거일_김문수득표율")
    frame["early_turnout"] = _numeric_series(metrics, "사전투표율")
    frame["electionday_turnout"] = _numeric_series(metrics, "선거일투표율")
    frame["turnout_ratio"] = _safe_divide(frame["early_turnout"], frame["electionday_turnout"])
    frame["day_margin"] = frame["day_lee"] - frame["day_kim"]
    return frame


def _fit_linear_model(frame: pd.DataFrame, target: str, features: list[str]) -> dict[str, object]:
    data = frame.loc[:, [target, *features]].apply(pd.to_numeric, errors="coerce").dropna()
    empty_model: dict[str, object] = {
        "target": target,
        "features": features,
        "n": int(len(data)),
        "intercept": float("nan"),
        "coefs": {feature: float("nan") for feature in features},
        "r2": float("nan"),
        "mae": float("nan"),
        "rmse": float("nan"),
        "valid": False,
    }
    if len(data) <= len(features):
        return empty_model

    x = data.loc[:, features].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(data)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst if sst else float("nan")
    return {
        "target": target,
        "features": features,
        "n": int(len(data)),
        "intercept": float(coefficients[0]),
        "coefs": {feature: float(value) for feature, value in zip(features, coefficients[1:])},
        "r2": r2,
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "valid": True,
    }


def _predict_linear_model(model: dict[str, object], frame: pd.DataFrame) -> pd.Series:
    features = list(model.get("features", []))
    if not model.get("valid") or not features:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")

    feature_frame = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    prediction = pd.Series(float(model["intercept"]), index=frame.index, dtype="float64")
    for feature in features:
        prediction = prediction + feature_frame[feature] * float(model["coefs"][feature])
    prediction = prediction.where(feature_frame.notna().all(axis=1))
    return prediction


def _loocv_prediction_frame(frame: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    data = frame.loc[:, ["읍면동", target, *features]].copy()
    numeric = data.loc[:, [target, *features]].apply(pd.to_numeric, errors="coerce")
    data.loc[:, [target, *features]] = numeric
    data = data.dropna(subset=[target, *features])
    rows: list[dict[str, object]] = []
    if len(data) <= len(features) + 1:
        return pd.DataFrame(columns=["읍면동", "실제", "예측", "잔차"])

    for index in data.index:
        train = data.drop(index)
        test = data.loc[[index]]
        model = _fit_linear_model(train, target, features)
        predicted = _predict_linear_model(model, test).iloc[0]
        actual = float(test[target].iloc[0])
        rows.append(
            {
                "읍면동": test["읍면동"].iloc[0],
                "실제": actual,
                "예측": predicted,
                "잔차": actual - predicted if pd.notna(predicted) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _add_estimated_combined_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    output_columns = [
        "사전추정_유효투표수",
        "추정합산_유효투표수",
        "관외보정_관내비중",
        "관외보정_관외비중",
        "관외보정_비선거일유효투표수",
        "관외보정_관내사전유효투표수",
        "관외보정_관외기타유효투표수",
        "선거일_양강마진",
        "사전추정_양강마진",
        "사전마진프리미엄",
        "마진모델_사전마진",
        "마진모델_사전마진프리미엄",
        *[
            column
            for candidate_name in CANDIDATE_PAIR
            for column in [
                f"관내모형_{candidate_name}득표율",
                f"관외실제_{candidate_name}득표수",
                f"관외실제_{candidate_name}득표율",
                f"사전추정_{candidate_name}득표수",
                f"사전추정_{candidate_name}득표율",
                f"사전프리미엄_{candidate_name}",
                f"추정합산_{candidate_name}득표수",
                f"추정합산_{candidate_name}득표율",
            ]
        ],
        "추정모델설명",
    ]
    for column in output_columns:
        if column not in result.columns:
            result[column] = pd.NA
    if result.empty:
        return result

    training = _build_estimation_training_frame(result)
    scoring = _build_estimation_scoring_frame(result)
    required = {"사전투표수", "선거일_유효투표수", "선거일투표수", "관내사전_유효투표수", "관내사전_투표수"}
    if not required.issubset(result.columns):
        return result

    early_valid_rate = _safe_divide(result["관내사전_유효투표수"], result["관내사전_투표수"])
    day_valid_rate = _safe_divide(result["선거일_유효투표수"], result["선거일투표수"])
    result["사전추정_유효투표수"] = (
        pd.to_numeric(result["사전투표수"], errors="coerce").fillna(0)
        * early_valid_rate.fillna(day_valid_rate).fillna(1.0)
    )

    unique_emd = result.drop_duplicates(subset=["읍면동KEY_D"]) if "읍면동KEY_D" in result.columns else result.iloc[0:0]
    total_actual_valid_values = pd.to_numeric(result.get("전체실제_유효투표수"), errors="coerce").dropna()
    total_actual_valid = float(total_actual_valid_values.iloc[0]) if not total_actual_valid_values.empty else float("nan")
    day_valid_total = pd.to_numeric(result["선거일_유효투표수"], errors="coerce").sum(min_count=1)
    local_early_valid_total = pd.to_numeric(unique_emd.get("관내사전_유효투표수"), errors="coerce").sum(min_count=1)
    non_day_valid_total = total_actual_valid - day_valid_total if pd.notna(total_actual_valid) and pd.notna(day_valid_total) else float("nan")
    outside_valid_total = non_day_valid_total - local_early_valid_total if pd.notna(non_day_valid_total) and pd.notna(local_early_valid_total) else float("nan")
    outside_share = outside_valid_total / non_day_valid_total if pd.notna(outside_valid_total) and non_day_valid_total and non_day_valid_total > 0 else 0.0
    outside_share = min(max(float(outside_share), 0.0), 1.0)
    local_share = 1.0 - outside_share
    result["관외보정_관내비중"] = local_share
    result["관외보정_관외비중"] = outside_share
    result["관외보정_비선거일유효투표수"] = non_day_valid_total
    result["관외보정_관내사전유효투표수"] = local_early_valid_total
    result["관외보정_관외기타유효투표수"] = outside_valid_total

    candidate_keys = {CANDIDATE_PAIR[0]: "lee", CANDIDATE_PAIR[1]: "kim"}
    for candidate_name, candidate_key in candidate_keys.items():
        model = _fit_linear_model(training, f"early_{candidate_key}", [f"day_{candidate_key}", "turnout_ratio"])
        local_model_rate = _predict_linear_model(model, scoring).clip(lower=0, upper=1)
        day_rate = _numeric_series(result, f"선거일_{candidate_name}득표율")
        day_votes = _numeric_series(result, f"선거일_{candidate_name}득표수").fillna(0)
        total_actual_candidate_values = pd.to_numeric(result.get(f"전체실제_{candidate_name}득표수"), errors="coerce").dropna()
        total_actual_candidate = float(total_actual_candidate_values.iloc[0]) if not total_actual_candidate_values.empty else float("nan")
        day_candidate_total = pd.to_numeric(result.get(f"선거일_{candidate_name}득표수"), errors="coerce").sum(min_count=1)
        local_early_candidate_total = pd.to_numeric(unique_emd.get(f"관내사전_{candidate_name}득표수"), errors="coerce").sum(min_count=1)
        outside_candidate_total = (
            total_actual_candidate - day_candidate_total - local_early_candidate_total
            if pd.notna(total_actual_candidate) and pd.notna(day_candidate_total) and pd.notna(local_early_candidate_total)
            else float("nan")
        )
        outside_rate = outside_candidate_total / outside_valid_total if pd.notna(outside_candidate_total) and pd.notna(outside_valid_total) and outside_valid_total else float("nan")
        predicted_rate = (local_share * local_model_rate + outside_share * outside_rate).clip(lower=0, upper=1)
        local_model_rate_col = f"관내모형_{candidate_name}득표율"
        outside_votes_actual_col = f"관외실제_{candidate_name}득표수"
        outside_rate_col = f"관외실제_{candidate_name}득표율"
        early_votes_col = f"사전추정_{candidate_name}득표수"
        early_rate_col = f"사전추정_{candidate_name}득표율"
        premium_col = f"사전프리미엄_{candidate_name}"
        combined_votes_col = f"추정합산_{candidate_name}득표수"
        combined_rate_col = f"추정합산_{candidate_name}득표율"

        result[local_model_rate_col] = local_model_rate
        result[outside_votes_actual_col] = outside_candidate_total
        result[outside_rate_col] = outside_rate
        result[early_rate_col] = predicted_rate
        result[premium_col] = predicted_rate - day_rate
        result[early_votes_col] = result["사전추정_유효투표수"] * predicted_rate
        result[combined_votes_col] = day_votes + pd.to_numeric(result[early_votes_col], errors="coerce").fillna(0)

    result["추정합산_유효투표수"] = (
        pd.to_numeric(result["선거일_유효투표수"], errors="coerce").fillna(0)
        + pd.to_numeric(result["사전추정_유효투표수"], errors="coerce").fillna(0)
    )
    for candidate_name in CANDIDATE_PAIR:
        result[f"추정합산_{candidate_name}득표율"] = _safe_divide(
            result[f"추정합산_{candidate_name}득표수"],
            result["추정합산_유효투표수"],
        )

    result["선거일_양강마진"] = scoring["day_margin"]
    result["사전추정_양강마진"] = (
        pd.to_numeric(result[f"사전추정_{CANDIDATE_PAIR[0]}득표율"], errors="coerce")
        - pd.to_numeric(result[f"사전추정_{CANDIDATE_PAIR[1]}득표율"], errors="coerce")
    )
    result["사전마진프리미엄"] = result["사전추정_양강마진"] - result["선거일_양강마진"]
    margin_model = _fit_linear_model(training, "early_margin", ["day_margin", "turnout_ratio"])
    result["마진모델_사전마진"] = _predict_linear_model(margin_model, scoring).clip(lower=-1, upper=1)
    result["마진모델_사전마진프리미엄"] = result["마진모델_사전마진"] - result["선거일_양강마진"]

    result["추정모델설명"] = [
        (
            "읍면동 관내사전투표의 후보별 득표율 흐름을 기준으로, 해당 투표구의 선거일 득표율과 "
            "사전/선거일 투표율 구조를 반영한 뒤, 전체 비선거일 유효투표 중 관외/기타 비중 "
            f"{outside_share:.1%}를 모든 투표구에 공통 적용해 관외 실제 득표율로 보정했습니다."
        )
        for _ in range(len(result))
    ]
    return result


def _prepare_polling_metrics(
    turnout_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    polling_df: pd.DataFrame,
    confirmed_df: pd.DataFrame,
) -> pd.DataFrame:
    metrics = calc_polling_station_metrics(turnout_df, votes_df, polling_df)
    metrics = _add_turnout_unit_metrics(metrics, confirmed_df)
    metrics = _add_candidate_metrics(metrics, votes_df)
    if metrics.empty:
        return metrics
    election_day_pair = _build_candidate_pair_metrics(votes_df, "투표소KEY", "투표소", "선거일")
    if not election_day_pair.empty:
        metrics = metrics.merge(election_day_pair, on="투표소KEY", how="left", copy=False)
    emd_election_day_pair = _build_candidate_pair_metrics(votes_df, "읍면동KEY", "투표소", "선거일읍면동")
    if not emd_election_day_pair.empty and "읍면동KEY_D" in metrics.columns:
        metrics = metrics.merge(
            emd_election_day_pair.rename(columns={"읍면동KEY": "__선거일읍면동_읍면동KEY"}),
            left_on="읍면동KEY_D",
            right_on="__선거일읍면동_읍면동KEY",
            how="left",
            copy=False,
        ).drop(columns="__선거일읍면동_읍면동KEY")
    local_early_pair = _build_candidate_pair_metrics(votes_df, "읍면동KEY", "관내사전투표", "관내사전")
    if not local_early_pair.empty and "읍면동KEY_D" in metrics.columns:
        metrics = metrics.merge(
            local_early_pair.rename(columns={"읍면동KEY": "__관내사전_읍면동KEY"}),
            left_on="읍면동KEY_D",
            right_on="__관내사전_읍면동KEY",
            how="left",
            copy=False,
        ).drop(columns="__관내사전_읍면동KEY")
    emd_actual_pair = _build_candidate_pair_metrics(votes_df, "읍면동KEY", "읍면동", "읍면동실제")
    if not emd_actual_pair.empty and "읍면동KEY_D" in metrics.columns:
        metrics = metrics.merge(
            emd_actual_pair.rename(columns={"읍면동KEY": "__읍면동실제_읍면동KEY"}),
            left_on="읍면동KEY_D",
            right_on="__읍면동실제_읍면동KEY",
            how="left",
            copy=False,
        ).drop(columns="__읍면동실제_읍면동KEY")
    total_actual_pair = _build_total_candidate_pair_metrics(votes_df, "합계", "전체실제")
    if not total_actual_pair.empty:
        for column in total_actual_pair.columns:
            metrics[column] = total_actual_pair.iloc[0][column]
    total_turnout = _build_total_turnout_summary_metrics(turnout_df)
    if not total_turnout.empty:
        for column in total_turnout.columns:
            metrics[column] = total_turnout.iloc[0][column]
    emd_turnout = _build_emd_turnout_summary_metrics(turnout_df)
    if not emd_turnout.empty and "읍면동KEY_D" in metrics.columns:
        metrics = metrics.merge(
            emd_turnout.rename(columns={"읍면동KEY": "__읍면동실제투표수_읍면동KEY"}),
            left_on="읍면동KEY_D",
            right_on="__읍면동실제투표수_읍면동KEY",
            how="left",
            copy=False,
        ).drop(columns="__읍면동실제투표수_읍면동KEY")
    local_early_turnout = _build_local_early_turnout_metrics(turnout_df)
    if not local_early_turnout.empty and "읍면동KEY_D" in metrics.columns:
        metrics = metrics.merge(
            local_early_turnout.rename(columns={"읍면동KEY": "__관내사전투표수_읍면동KEY"}),
            left_on="읍면동KEY_D",
            right_on="__관내사전투표수_읍면동KEY",
            how="left",
            copy=False,
        ).drop(columns="__관내사전투표수_읍면동KEY")
    emd_model_turnout = _build_emd_model_turnout_metrics(turnout_df, confirmed_df)
    if not emd_model_turnout.empty and "읍면동KEY_D" in metrics.columns:
        metrics = metrics.merge(
            emd_model_turnout.rename(columns={"읍면동KEY": "__모델투표율_읍면동KEY"}),
            left_on="읍면동KEY_D",
            right_on="__모델투표율_읍면동KEY",
            how="left",
            copy=False,
        ).drop(columns="__모델투표율_읍면동KEY")
    metrics = _add_estimated_combined_metrics(metrics)
    for column in ["투표소명_F", "읍면동명_F", "투표소KEY"]:
        if column in metrics.columns:
            metrics[column] = metrics[column].astype("string")
    return metrics


def _area_region_frame(area: dict[str, Any], polling_df: pd.DataFrame) -> pd.DataFrame:
    if polling_df.empty:
        return pd.DataFrame(columns=["읍면동KEY_D", "읍면동명_F", "지도지표"])

    emd_order = {name: index for index, name in enumerate(area["emd_names"])}
    region = polling_df.loc[polling_df["읍면동명_F"].astype("string").isin(area["emd_names"])].copy()
    if region.empty:
        return pd.DataFrame(columns=["읍면동KEY_D", "읍면동명_F", "지도지표"])

    region = (
        region.loc[:, ["읍면동KEY_D", "읍면동명_F"]]
        .dropna(subset=["읍면동KEY_D", "읍면동명_F"])
        .drop_duplicates(subset=["읍면동KEY_D"])
        .assign(
            지도지표=1,
            __order=lambda frame: frame["읍면동명_F"].map(emd_order).fillna(99),
        )
        .sort_values(by=["__order", "읍면동명_F"], kind="stable")
        .drop(columns="__order")
        .reset_index(drop=True)
    )
    return region


def _area_point_frame(area: dict[str, Any], polling_metrics: pd.DataFrame, priority_df: pd.DataFrame) -> pd.DataFrame:
    area_priority = priority_df.loc[priority_df["권역ID"].eq(area["id"])].copy()
    if polling_metrics.empty or area_priority.empty:
        return pd.DataFrame()

    metrics = polling_metrics.copy()
    metrics["투표소명_F"] = metrics["투표소명_F"].astype("string")
    area_priority["투표소명_F"] = area_priority["투표소명_F"].astype("string")
    points = metrics.merge(
        area_priority,
        on=["투표소명_F", "읍면동명_F"],
        how="inner",
        copy=False,
    )
    if points.empty:
        return points

    jurisdiction_df = _load_polling_jurisdiction_frame()
    if not jurisdiction_df.empty:
        points = points.merge(jurisdiction_df, on="투표소명_F", how="left", copy=False)
    if "관할구역" not in points.columns:
        points["관할구역"] = "-"
    points["관할구역"] = points["관할구역"].fillna("-").replace("", "-")
    points["색상"] = points["우선순위"].map(lambda rank: PRIORITY_STYLES[int(rank)]["color"])
    points["짧은표시명"] = points["투표소명_F"].map(_short_polling_label)
    points["이재명득표수_표시"] = points["이재명득표수"].map(_format_number) if "이재명득표수" in points.columns else "-"
    points["이재명득표율_표시"] = points["이재명득표율"].map(_format_percent) if "이재명득표율" in points.columns else "-"
    points["사전투표비중_표시"] = points["사전투표 비중"].map(_format_percent) if "사전투표 비중" in points.columns else "-"
    points["투표수_표시"] = points["투표수"].map(_format_number) if "투표수" in points.columns else "-"
    for source_column in ["확정선거인수", "선거일투표수", "사전투표수"]:
        points[f"{source_column}_표시"] = points[source_column].map(_format_number) if source_column in points.columns else "-"
    for source_column in ["선거일투표율", "사전투표율", "사전투표비중_투표구"]:
        points[f"{source_column}_표시"] = points[source_column].map(_format_percent) if source_column in points.columns else "-"
    if "추정모델설명" not in points.columns:
        points["추정모델설명"] = "후보별 모델 B 기반 사전투표 득표율 추정"
    for prefix in ["선거일", "관내사전", "사전추정", "추정합산"]:
        if f"{prefix}_유효투표수" not in points.columns:
            points[f"{prefix}_유효투표수"] = pd.NA
        for candidate_name in CANDIDATE_PAIR:
            for metric in ["득표수", "득표율"]:
                column = f"{prefix}_{candidate_name}{metric}"
                if column not in points.columns:
                    points[column] = pd.NA
    for source_column in [
        f"사전프리미엄_{CANDIDATE_PAIR[0]}",
        f"사전프리미엄_{CANDIDATE_PAIR[1]}",
        "사전마진프리미엄",
        "마진모델_사전마진프리미엄",
    ]:
        points[f"{source_column}_표시"] = points[source_column].map(_format_percentage_point) if source_column in points.columns else "-"

    points["마커크기"] = 20
    points = _add_display_coordinates(points)

    return points.sort_values(by=["우선순위", "읍면동명_F", "투표소명_F"], kind="stable").reset_index(drop=True)


def _area_early_point_frame(area: dict[str, Any], polling_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["투표소명_F", "읍면동명_F", "장소명", "주소", "위도", "경도", "표시위도", "표시경도"]
    if polling_df.empty:
        return pd.DataFrame(columns=columns)

    early = polling_df.loc[
        polling_df["읍면동명_F"].astype("string").isin(area["emd_names"])
        & polling_df["투표소명_F"].astype("string").str.endswith("사전투표소", na=False)
    ].copy()
    if early.empty:
        return pd.DataFrame(columns=columns)

    early = _add_display_coordinates(early)
    early = early.dropna(subset=["표시위도", "표시경도"])
    if early.empty:
        return pd.DataFrame(columns=columns)
    return early.loc[:, columns].drop_duplicates(subset=["투표소명_F", "표시위도", "표시경도"]).reset_index(drop=True)


def _short_polling_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    if text == "산북면투표소":
        return "투"
    match = re.search(r"제(\d+)투$", text)
    if match:
        return f"{match.group(1)}투"
    if text.endswith("투표소"):
        return "투표소"
    return text


def _add_display_coordinates(points: pd.DataFrame) -> pd.DataFrame:
    result = points.copy()
    result["표시위도"] = pd.to_numeric(result["위도"], errors="coerce")
    result["표시경도"] = pd.to_numeric(result["경도"], errors="coerce")
    if {"읍면동명_F", "투표소명_F"}.issubset(result.columns):
        emd_series = result["읍면동명_F"].astype("string")
        place_series = result["투표소명_F"].astype("string")
        for (emd_name, polling_suffix), (lat, lon) in POLLING_PLACE_COORDINATE_OVERRIDES.items():
            mask = emd_series.eq(emd_name) & (
                place_series.eq(polling_suffix)
                | place_series.eq(f"{emd_name}{polling_suffix}")
                | place_series.str.endswith(polling_suffix, na=False)
            )
            result.loc[mask, ["표시위도", "표시경도"]] = (lat, lon)
    return result


def _filter_geojson_strict(geojson: dict[str, Any], featureidkey: str | None, locations: pd.Series) -> dict[str, Any]:
    if not featureidkey:
        return {"type": geojson.get("type", "FeatureCollection"), "features": []}
    property_name = featureidkey.split(".", maxsplit=1)[1] if "." in featureidkey else featureidkey
    allowed = set(locations.dropna().astype("string"))
    features = [
        feature
        for feature in geojson.get("features", [])
        if str(feature.get("properties", {}).get(property_name)) in allowed
    ]
    return {"type": geojson.get("type", "FeatureCollection"), "features": features}


def _geojson_map_view(geojson: dict[str, Any], fallback_points: pd.DataFrame) -> dict[str, float]:
    features = geojson.get("features", [])
    if features:
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        if not gdf.empty:
            minx, miny, maxx, maxy = gdf.total_bounds
            span = max(float(maxx - minx), float(maxy - miny))
            if span <= 0.05:
                zoom = 11.1
            elif span <= 0.12:
                zoom = 10.2
            elif span <= 0.25:
                zoom = 9.3
            elif span <= 0.45:
                zoom = 8.7
            else:
                zoom = 8.1
            return {"lat": float((miny + maxy) / 2), "lon": float((minx + maxx) / 2), "zoom": zoom}

    if {"위도", "경도"}.issubset(fallback_points.columns):
        valid = fallback_points.loc[fallback_points["위도"].notna() & fallback_points["경도"].notna()]
        if not valid.empty:
            return {
                "lat": float(valid["위도"].astype(float).median()),
                "lon": float(valid["경도"].astype(float).median()),
                "zoom": 10.0,
            }
    return {"lat": 37.30, "lon": 127.63, "zoom": 9.0}


def _advance_polling_ring_coordinates(
    early_point_df: pd.DataFrame,
    map_view: dict[str, float],
    segments: int = 48,
) -> tuple[list[float | None], list[float | None]]:
    lats: list[float | None] = []
    lons: list[float | None] = []
    if early_point_df.empty:
        return lats, lons

    view_lat = float(map_view.get("lat", 37.30))
    view_zoom = float(map_view.get("zoom", 10.0))
    radius_px = max((ADVANCE_POLLING_MARKER_SIZE_PX - ADVANCE_POLLING_RING_WIDTH) / 2, 1)
    meters_per_pixel = 156543.03392 * max(math.cos(math.radians(view_lat)), 0.2) / (2 ** (view_zoom + 1))
    radius_meters = max(radius_px * meters_per_pixel, 2.0)

    for _, row in early_point_df.iterrows():
        try:
            lat = float(row["표시위도"])
            lon = float(row["표시경도"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue

        meters_per_degree_lat = 111_320.0
        meters_per_degree_lon = max(111_320.0 * math.cos(math.radians(lat)), 1.0)
        for index in range(segments + 1):
            angle = 2 * math.pi * index / segments
            lons.append(lon + (math.cos(angle) * radius_meters / meters_per_degree_lon))
            lats.append(lat + (math.sin(angle) * radius_meters / meters_per_degree_lat))
        lons.append(None)
        lats.append(None)

    return lats, lons


def _boundary_label_frame(
    geojson: dict[str, Any],
    featureidkey: str | None,
    region_df: pd.DataFrame,
    avoid_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features = geojson.get("features", [])
    if not features or not featureidkey:
        return pd.DataFrame(columns=["읍면동KEY_D", "읍면동명_F", "lat", "lon"])

    property_name = featureidkey.split(".", maxsplit=1)[1] if "." in featureidkey else featureidkey
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if gdf.empty or property_name not in gdf.columns:
        return pd.DataFrame(columns=["읍면동KEY_D", "읍면동명_F", "lat", "lon"])

    projected = gdf.to_crs(epsg=3857)
    avoid_geometries: list[Point] = []
    if avoid_points is not None and {"표시위도", "표시경도"}.issubset(avoid_points.columns):
        valid_points = avoid_points.loc[avoid_points["표시위도"].notna() & avoid_points["표시경도"].notna()].copy()
        if not valid_points.empty:
            avoid_gdf = gpd.GeoDataFrame(
                valid_points,
                geometry=gpd.points_from_xy(valid_points["표시경도"], valid_points["표시위도"]),
                crs="EPSG:4326",
            ).to_crs(epsg=3857)
            avoid_geometries = list(avoid_gdf.geometry)

    selected_label_points: list[Point] = []
    label_rows: list[dict[str, object]] = []
    for _, row in projected.iterrows():
        geometry = row.geometry
        representative = geometry.representative_point()
        candidates = [representative]
        centroid = geometry.centroid
        if geometry.covers(centroid):
            candidates.append(centroid)

        minx, miny, maxx, maxy = geometry.bounds
        for x_step in range(1, 7):
            for y_step in range(1, 7):
                candidate = Point(
                    minx + (maxx - minx) * x_step / 7,
                    miny + (maxy - miny) * y_step / 7,
                )
                if geometry.covers(candidate):
                    candidates.append(candidate)

        def _candidate_score(candidate: Point) -> float:
            point_distance = min((candidate.distance(point) for point in avoid_geometries), default=80_000.0)
            label_distance = min((candidate.distance(point) for point in selected_label_points), default=80_000.0)
            center_penalty = candidate.distance(representative) * 0.06
            return min(point_distance, label_distance * 0.9) - center_penalty

        best = max(candidates, key=_candidate_score)
        selected_label_points.append(best)
        best_wgs = gpd.GeoSeries([best], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        label_rows.append(
            {
                "읍면동KEY_D": str(row[property_name]),
                "lat": float(best_wgs.y),
                "lon": float(best_wgs.x),
            }
        )

    label_points = pd.DataFrame(label_rows)
    return label_points.merge(
        region_df.loc[:, ["읍면동KEY_D", "읍면동명_F"]].astype("string"),
        on="읍면동KEY_D",
        how="inner",
        copy=False,
    )


def _add_boundary_label_annotations(fig: go.Figure, geojson: dict[str, Any], label_df: pd.DataFrame) -> None:
    if label_df.empty or not geojson.get("features"):
        return

    gdf = gpd.GeoDataFrame.from_features(geojson.get("features", []), crs="EPSG:4326")
    if gdf.empty:
        return
    minx, miny, maxx, maxy = gdf.total_bounds
    x_span = max(float(maxx - minx), 0.001)
    y_span = max(float(maxy - miny), 0.001)
    x_pad = max(x_span * 0.18, 0.006)
    y_pad = max(y_span * 0.18, 0.006)

    positions: list[dict[str, object]] = []
    for row in label_df.itertuples(index=False):
        x = (float(row.lon) - (minx - x_pad)) / (x_span + 2 * x_pad)
        y = (float(row.lat) - (miny - y_pad)) / (y_span + 2 * y_pad)
        positions.append(
            {
                "label": str(row.읍면동명_F),
                "x": min(max(x, 0.06), 0.94),
                "y": min(max(y, 0.07), 0.93),
            }
        )

    for _ in range(50):
        moved = False
        for left_idx in range(len(positions)):
            for right_idx in range(left_idx + 1, len(positions)):
                left = positions[left_idx]
                right = positions[right_idx]
                dx = float(right["x"]) - float(left["x"])
                dy = float(right["y"]) - float(left["y"])
                distance = math.hypot(dx, dy)
                min_sep = 0.085
                if distance >= min_sep:
                    continue
                if distance == 0:
                    dx, dy, distance = min_sep, 0.0, min_sep
                push = (min_sep - distance) / 2
                ux = dx / distance
                uy = dy / distance
                left["x"] = min(max(float(left["x"]) - ux * push, 0.06), 0.94)
                right["x"] = min(max(float(right["x"]) + ux * push, 0.06), 0.94)
                left["y"] = min(max(float(left["y"]) - uy * push, 0.07), 0.93)
                right["y"] = min(max(float(right["y"]) + uy * push, 0.07), 0.93)
                moved = True
        if not moved:
            break

    for item in positions:
        fig.add_annotation(
            x=float(item["x"]),
            y=float(item["y"]),
            xref="paper",
            yref="paper",
            text=f"<b>{item['label']}</b>",
            showarrow=False,
            font=dict(size=14, color="#123A38"),
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(18,58,56,0.35)",
            borderwidth=1,
            borderpad=3,
        )


def _build_custom_map(
    area: dict[str, Any],
    region_df: pd.DataFrame,
    point_df: pd.DataFrame,
    early_point_df: pd.DataFrame,
    geometry_context: dict[str, Any],
) -> go.Figure:
    if region_df.empty and point_df.empty and early_point_df.empty:
        return empty_map_figure("표시할 권역 데이터가 없습니다.")

    featureidkey = geometry_context.get("featureidkey")
    geojson = (
        _filter_geojson_strict(geometry_context["geojson"], featureidkey, region_df["읍면동KEY_D"])
        if geometry_context.get("available") and region_df["읍면동KEY_D"].notna().any()
        else {"type": "FeatureCollection", "features": []}
    )
    fallback_points = pd.concat([point_df, early_point_df], ignore_index=True) if not early_point_df.empty else point_df
    map_view = _geojson_map_view(geojson, fallback_points)
    fig = go.Figure()
    label_df = pd.DataFrame()

    if geojson.get("features"):
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=region_df["읍면동KEY_D"].astype("string"),
                z=[1] * len(region_df),
                featureidkey=featureidkey,
                colorscale=[[0.0, BOUNDARY_FILL_COLOR], [1.0, BOUNDARY_FILL_COLOR]],
                showscale=False,
                marker_line_color=BOUNDARY_LINE_COLOR,
                marker_line_width=2,
                marker_opacity=1.0,
                showlegend=False,
                customdata=region_df[["읍면동명_F"]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
                name="읍면동 경계",
            )
        )

    rank_layers: list[tuple[int, dict[str, str], pd.DataFrame, list[str]]] = []
    for rank in sorted(point_df["우선순위"].dropna().astype(int).unique().tolist()) if not point_df.empty else []:
        rank_points = point_df.loc[point_df["우선순위"].astype(int).eq(rank)].copy()
        style = PRIORITY_STYLES[rank]
        custom_cols = [
            "투표소명_F",
            "우선순위라벨",
            "성격",
            "장소명",
            "주소",
            "관할구역",
            "읍면동명_F",
            "확정선거인수_표시",
            "선거일투표수_표시",
            "선거일투표율_표시",
            "사전투표수_표시",
            "사전투표율_표시",
            "사전투표비중_투표구_표시",
            "선거일_이재명득표수",
            "선거일_이재명득표율",
            "선거일_김문수득표수",
            "선거일_김문수득표율",
            "선거일_유효투표수",
            "관내사전_이재명득표수",
            "관내사전_이재명득표율",
            "관내사전_김문수득표수",
            "관내사전_김문수득표율",
            "관내사전_유효투표수",
            "추정합산_이재명득표수",
            "추정합산_이재명득표율",
            "추정합산_김문수득표수",
            "추정합산_김문수득표율",
            "추정합산_유효투표수",
            "추정모델설명",
        ]
        for column in custom_cols:
            if column not in rank_points.columns:
                rank_points[column] = "-"
        rank_layers.append((rank, style, rank_points, custom_cols))

    for _, _, rank_points, _ in rank_layers:
        fig.add_trace(
            go.Scattermapbox(
                lat=rank_points["표시위도"],
                lon=rank_points["표시경도"],
                mode="markers",
                marker=dict(
                    size=rank_points["마커크기"] + 7,
                    color="#FFFFFF",
                    opacity=0.92,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for _, style, rank_points, custom_cols in rank_layers:
        fig.add_trace(
            go.Scattermapbox(
                lat=rank_points["표시위도"],
                lon=rank_points["표시경도"],
                mode="markers+text",
                name=f"{style['label']} - {rank_points['성격'].iloc[0]}",
                marker=dict(
                    size=rank_points["마커크기"],
                    color=style["color"],
                    opacity=0.92,
                ),
                text=rank_points["짧은표시명"],
                textposition="middle center",
                textfont=dict(size=11, color="#FFFFFF"),
                customdata=rank_points[custom_cols].to_numpy(),
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
        )

    if not early_point_df.empty:
        ring_lats, ring_lons = _advance_polling_ring_coordinates(early_point_df, map_view)
        fig.add_trace(
            go.Scattermapbox(
                lat=early_point_df["표시위도"],
                lon=early_point_df["표시경도"],
                mode="markers",
                name=ADVANCE_POLLING_DATA_TRACE_NAME,
                below="",
                marker=dict(size=1, color="rgba(0,0,0,0)", opacity=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[float(early_point_df["표시위도"].iloc[0]), float(early_point_df["표시위도"].iloc[0])],
                lon=[float(early_point_df["표시경도"].iloc[0]), float(early_point_df["표시경도"].iloc[0]) + 1e-10],
                mode="lines",
                name=ADVANCE_POLLING_LEGEND_NAME,
                line=dict(color=ADVANCE_POLLING_RING_COLOR, width=ADVANCE_POLLING_RING_WIDTH),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=ring_lats,
                lon=ring_lons,
                mode="lines",
                name=ADVANCE_POLLING_EXPORT_TRACE_NAME,
                line=dict(color=ADVANCE_POLLING_RING_COLOR, width=ADVANCE_POLLING_RING_WIDTH),
                hoverinfo="skip",
                showlegend=False,
                visible=False,
            )
        )

    for _, _, rank_points, custom_cols in rank_layers:
        fig.add_trace(
            go.Scattermapbox(
                lat=rank_points["표시위도"],
                lon=rank_points["표시경도"],
                mode="markers",
                marker=dict(
                    size=rank_points["마커크기"] + 26,
                    color="rgba(0,0,0,0.01)",
                    opacity=0.01,
                ),
                customdata=rank_points[custom_cols].to_numpy(),
                hovertemplate="%{customdata[0]}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"권역별 우선순위 투표소: {area['title']}",
            x=0.01,
            y=0.985,
            xanchor="left",
            yanchor="top",
            font=dict(size=18),
        ),
        height=680 if len(point_df) >= 12 else 620,
        margin=dict(l=10, r=10, t=92, b=10),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.055,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.86)",
            font=dict(size=11),
        ),
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "source": [VWORLD_BASE_TILE_URL],
                "sourceattribution": "VWorld",
                "opacity": 1.0,
            }
        ],
        mapbox_center={"lat": map_view["lat"], "lon": map_view["lon"]},
        mapbox_zoom=map_view["zoom"],
    )
    return fig


def _render_hover_chart_map(fig: go.Figure, key: str) -> None:
    height = int(fig.layout.height or 680)
    figure_json = fig.to_json()
    lee_photo = json.dumps(LEE_PHOTO_URL)
    kim_photo = json.dumps(KIM_PHOTO_URL)
    html = f"""
<div id="{key}_wrap" class="custom-map-wrap">
  <button id="{key}_fullscreen" class="map-fullscreen-button" type="button" aria-label="지도 전체화면">전체화면</button>
  <div id="{key}_plot" class="custom-map-plot"></div>
  <div id="{key}_early_stars" class="early-star-layer"></div>
  <div id="{key}_tooltip" class="vote-tooltip"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(() => {{
  const figure = {figure_json};
  const leePhoto = {lee_photo};
  const kimPhoto = {kim_photo};
  const wrap = document.getElementById("{key}_wrap");
  const plot = document.getElementById("{key}_plot");
  const earlyStarLayer = document.getElementById("{key}_early_stars");
  const tooltip = document.getElementById("{key}_tooltip");
  const fullscreenButton = document.getElementById("{key}_fullscreen");
  const baseHeight = {height};
  const earlyPollingDataTraceName = "{ADVANCE_POLLING_DATA_TRACE_NAME}";
  const earlyPollingLegendTraceName = "{ADVANCE_POLLING_LEGEND_NAME}";
  const earlyPollingExportTraceName = "{ADVANCE_POLLING_EXPORT_TRACE_NAME}";
  const earlyPollingSourceId = "{key}_early_polling_source";
  const earlyPollingLayerId = "{key}_early_polling_layer";
  const earlyPollingSegments = 48;
  const earlyPollingMarkerSize = {ADVANCE_POLLING_MARKER_SIZE_PX};
  const earlyPollingStrokeWidth = {ADVANCE_POLLING_RING_WIDTH};
  const earlyPollingStrokeColor = "{ADVANCE_POLLING_RING_COLOR}";
  let expandedInFrame = false;
  let earlyPollingRenderTimer = null;
  const layout = Object.assign({{}}, figure.layout || {{}});
  layout.autosize = true;
  delete layout.width;

  const config = {{
    responsive: true,
    scrollZoom: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d"],
    modeBarButtonsToAdd: [{{
      name: "downloadImageWithAdvancePolling",
      title: "Download plot as a png",
      icon: Plotly.Icons.camera,
      click: downloadPlotImage
    }}]
  }};

  const escapeHtml = (value) => String(value ?? "-")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

  const toNumber = (value) => {{
    if (value === null || value === undefined || value === "" || value === "-") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }};

  const formatVotes = (value) => {{
    const parsed = toNumber(value);
    return parsed === null ? "-" : Math.round(parsed).toLocaleString("ko-KR") + "표";
  }};

  const formatPct = (value) => {{
    const parsed = toNumber(value);
    return parsed === null ? "-" : (parsed * 100).toFixed(1) + "%";
  }};

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const resizePlot = () => {{
    window.setTimeout(() => Plotly.Plots.resize(plot), 80);
  }};

  const setFrameHeight = (nextHeight) => {{
    try {{
      window.parent.postMessage({{
        isStreamlitMessage: true,
        type: "streamlit:setFrameHeight",
        height: nextHeight
      }}, "*");
    }} catch (error) {{}}
  }};

  const setExpandedInFrame = (nextExpanded) => {{
    expandedInFrame = nextExpanded;
    wrap.classList.toggle("is-expanded", expandedInFrame);
    fullscreenButton.textContent = expandedInFrame ? "닫기" : "전체화면";
    setFrameHeight(expandedInFrame ? Math.max(window.innerHeight - 28, 720) : baseHeight);
    resizePlot();
    scheduleEarlyStarsRetries();
  }};

  const getEarlyPollingTrace = () => (plot._fullData || [])
    .find((trace) => trace && trace.name === earlyPollingDataTraceName && trace.lon && trace.lat);

  const earlyPollingSourceData = () => {{
    const earlyTrace = getEarlyPollingTrace();
    if (!earlyTrace) return null;
    const features = [];
    for (let i = 0; i < earlyTrace.lon.length; i += 1) {{
      const lon = Number(earlyTrace.lon[i]);
      const lat = Number(earlyTrace.lat[i]);
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
      features.push({{
        type: "Feature",
        geometry: {{ type: "Point", coordinates: [lon, lat] }},
        properties: {{}}
      }});
    }}
    return features.length ? {{ type: "FeatureCollection", features }} : null;
  }};

  const buildEarlyRingCoordinates = () => {{
    const subplot = plot._fullLayout && plot._fullLayout.mapbox && plot._fullLayout.mapbox._subplot;
    const map = subplot && subplot.map;
    const earlyTrace = getEarlyPollingTrace();
    if (!map || !earlyTrace) return null;

    const zoom = typeof map.getZoom === "function" ? map.getZoom() : 10;
    const center = typeof map.getCenter === "function" ? map.getCenter() : {{ lat: 37.3 }};
    const centerLat = Number(center.lat);
    const viewLat = Number.isFinite(centerLat) ? centerLat : 37.3;
    const radiusPx = Math.max((earlyPollingMarkerSize - earlyPollingStrokeWidth) / 2, 1);
    const metersPerPixel = 156543.03392 * Math.max(Math.cos(viewLat * Math.PI / 180), 0.2) / Math.pow(2, zoom + 1);
    const radiusMeters = Math.max(radiusPx * metersPerPixel, 2);
    const ringLats = [];
    const ringLons = [];
    for (let i = 0; i < earlyTrace.lon.length; i += 1) {{
      const lon = Number(earlyTrace.lon[i]);
      const lat = Number(earlyTrace.lat[i]);
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
      const metersPerDegreeLat = 111320;
      const metersPerDegreeLon = Math.max(111320 * Math.cos(lat * Math.PI / 180), 1);
      for (let index = 0; index <= earlyPollingSegments; index += 1) {{
        const angle = 2 * Math.PI * index / earlyPollingSegments;
        ringLons.push(lon + (Math.cos(angle) * radiusMeters / metersPerDegreeLon));
        ringLats.push(lat + (Math.sin(angle) * radiusMeters / metersPerDegreeLat));
      }}
      ringLons.push(null);
      ringLats.push(null);
    }}
    return ringLats.length ? {{ ringLats, ringLons }} : null;
  }};

  async function downloadPlotImage(gd) {{
    const targetPlot = gd || plot;
    const exportTraceIndex = (targetPlot.data || []).findIndex((trace) => trace && trace.name === earlyPollingExportTraceName);
    const imageOptions = {{
      format: "png",
      filename: "newplot",
      width: targetPlot.clientWidth || 1200,
      height: targetPlot.clientHeight || baseHeight,
      scale: 1
    }};
    if (exportTraceIndex < 0) {{
      return Plotly.downloadImage(targetPlot, imageOptions);
    }}
    const ringCoordinates = buildEarlyRingCoordinates();
    try {{
      if (ringCoordinates) {{
        await Plotly.restyle(
          targetPlot,
          {{ lat: [ringCoordinates.ringLats], lon: [ringCoordinates.ringLons], visible: [true] }},
          [exportTraceIndex]
        );
      }}
      await Plotly.downloadImage(targetPlot, imageOptions);
    }} finally {{
      await Plotly.restyle(targetPlot, {{ visible: [false] }}, [exportTraceIndex]);
    }}
  }}

  const renderEarlyStars = () => {{
    earlyPollingRenderTimer = null;
    earlyStarLayer.innerHTML = "";
    const subplot = plot._fullLayout && plot._fullLayout.mapbox && plot._fullLayout.mapbox._subplot;
    const map = subplot && subplot.map;
    const sourceData = earlyPollingSourceData();
    if (!map || !sourceData) return;

    const applyLayer = () => {{
      if (!map.isStyleLoaded || !map.isStyleLoaded()) return;
      try {{
        const source = map.getSource(earlyPollingSourceId);
        if (source && typeof source.setData === "function") {{
          source.setData(sourceData);
        }} else if (!source) {{
          map.addSource(earlyPollingSourceId, {{ type: "geojson", data: sourceData }});
        }}
        if (!map.getLayer(earlyPollingLayerId)) {{
          map.addLayer({{
            id: earlyPollingLayerId,
            type: "circle",
            source: earlyPollingSourceId,
            paint: {{
              "circle-radius": Math.max((earlyPollingMarkerSize - earlyPollingStrokeWidth) / 2, 1),
              "circle-color": "rgba(255,255,255,0)",
              "circle-opacity": 0,
              "circle-stroke-color": earlyPollingStrokeColor,
              "circle-stroke-width": earlyPollingStrokeWidth,
              "circle-stroke-opacity": 1
            }}
          }});
        }} else if (typeof map.moveLayer === "function") {{
          map.moveLayer(earlyPollingLayerId);
        }}
      }} catch (error) {{}}
    }};

    if (map.isStyleLoaded && map.isStyleLoaded()) {{
      applyLayer();
    }} else if (typeof map.once === "function") {{
      map.once("load", applyLayer);
      map.once("idle", applyLayer);
    }}
  }};

  const scheduleEarlyStars = (delay = 90) => {{
    if (earlyPollingRenderTimer !== null) return;
    earlyPollingRenderTimer = window.setTimeout(renderEarlyStars, delay);
  }};

  const scheduleEarlyStarsRetries = () => {{
    [0, 120, 350, 900, 1800, 3200].forEach((delay) => {{
      window.setTimeout(() => scheduleEarlyStars(0), delay);
    }});
  }};

  const bindEarlyStarMapEvents = () => {{
    const subplot = plot._fullLayout && plot._fullLayout.mapbox && plot._fullLayout.mapbox._subplot;
    const map = subplot && subplot.map;
    if (!map || map.__earlyStarBound) return;
    map.__earlyStarBound = true;
    if (typeof map.on === "function") {{
      map.on("styledata", () => scheduleEarlyStars(120));
      map.on("idle", () => scheduleEarlyStars(120));
    }}
  }};

  const enableWheelZoom = () => {{
    const subplot = plot._fullLayout && plot._fullLayout.mapbox && plot._fullLayout.mapbox._subplot;
    const map = subplot && subplot.map;
    if (map && map.scrollZoom && typeof map.scrollZoom.enable === "function") {{
      map.scrollZoom.enable();
    }}
  }};

  const miniBar = (title, leeVotes, leePct, kimVotes, kimPct, validVotes, note = "") => {{
    const leeWidth = clamp((toNumber(leePct) || 0) * 100, 0, 100);
    const kimWidth = clamp((toNumber(kimPct) || 0) * 100, 0, 100);
    const noteText = note ? `${{escapeHtml(note)}} · ` : "";
    return `
      <div class="vote-row">
        <div class="vote-row-title">${{escapeHtml(title)}} <span>${{noteText}}유효 ${{formatVotes(validVotes)}}</span></div>
        <div class="vote-row-body">
          <div class="candidate candidate-left">
            <img src="${{leePhoto}}" alt="이재명" />
            <div><b>이재명</b><strong>${{formatPct(leePct)}}</strong><em>${{formatVotes(leeVotes)}}</em></div>
          </div>
          <div class="bar-shell">
            <div class="bar-fill lee" style="width:${{leeWidth}}%"></div>
            <div class="bar-fill kim" style="width:${{kimWidth}}%"></div>
          </div>
          <div class="candidate candidate-right">
            <div><b>김문수</b><strong>${{formatPct(kimPct)}}</strong><em>${{formatVotes(kimVotes)}}</em></div>
            <img src="${{kimPhoto}}" alt="김문수" />
          </div>
        </div>
      </div>
    `;
  }};

  const renderTooltip = (data) => {{
    return `
      <div class="tooltip-head">
        <div>
          <b>${{escapeHtml(data[0])}}</b>
          <span>${{escapeHtml(data[6])}} · ${{escapeHtml(data[1])}}</span>
        </div>
      </div>
      <div class="tooltip-meta">
        <div><b>성격</b> ${{escapeHtml(data[2])}}</div>
        <div><b>장소</b> ${{escapeHtml(data[3])}}</div>
        <div><b>주소</b> ${{escapeHtml(data[4])}}</div>
        <div><b>관할구역</b> ${{escapeHtml(data[5])}}</div>
      </div>
      <div class="unit-panel">
        <div class="unit-title">투표구 단위</div>
        <div class="unit-grid">
          <div><b>확정선거인수</b><span>${{escapeHtml(data[7])}}명</span></div>
          <div><b>선거일 투표</b><span>${{escapeHtml(data[8])}}표 / ${{escapeHtml(data[9])}}</span></div>
          <div><b>비선거일 투표</b><span>${{escapeHtml(data[10])}}표 / ${{escapeHtml(data[11])}}</span></div>
          <div><b>비선거일 비중</b><span>${{escapeHtml(data[12])}}</span></div>
        </div>
      </div>
      <div class="mini-title">이재명 vs 김문수</div>
      ${{miniBar("해당 투표구 선거일투표", data[13], data[14], data[15], data[16], data[17])}}
      ${{miniBar("소속 읍면동 관내사전투표", data[18], data[19], data[20], data[21], data[22])}}
      ${{miniBar("해당 투표구 선거일+사전 투표결과(모델 추정)", data[23], data[24], data[25], data[26], data[27], "모델")}}
      <div class="model-note">${{escapeHtml(data[28])}}</div>
    `;
  }};

  const hideTooltip = () => {{
    tooltipPinnedByManual = false;
    tooltip.style.display = "none";
  }};

  let tooltipPinnedByManual = false;

  const positionTooltip = (mouseEvent) => {{
    const rect = wrap.getBoundingClientRect();
    const rawX = mouseEvent ? mouseEvent.clientX - rect.left + 18 : 24;
    const rawY = mouseEvent ? mouseEvent.clientY - rect.top + 18 : 24;
    const maxX = rect.width - tooltip.offsetWidth - 10;
    const maxY = rect.height - tooltip.offsetHeight - 10;
    tooltip.style.left = clamp(rawX, 10, Math.max(10, maxX)) + "px";
    tooltip.style.top = clamp(rawY, 10, Math.max(10, maxY)) + "px";
  }};

  const showTooltip = (data, mouseEvent, source = "plotly") => {{
    tooltipPinnedByManual = source === "manual";
    tooltip.innerHTML = renderTooltip(data);
    tooltip.style.display = "block";
    positionTooltip(mouseEvent);
  }};

  const markerSizeAt = (markerSize, index) => {{
    const value = Array.isArray(markerSize) || ArrayBuffer.isView(markerSize)
      ? markerSize[index]
      : markerSize;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 30;
  }};

  const traceHitPoints = () => {{
    const traces = plot._fullData || [];
    const points = [];
    traces.forEach((trace) => {{
      if (!trace || !trace.customdata || !trace.lon || !trace.lat) return;
      const markerSize = trace.marker ? trace.marker.size : 30;
      for (let i = 0; i < trace.customdata.length; i += 1) {{
        const customdata = trace.customdata[i];
        if (!Array.isArray(customdata) || customdata.length < 29) continue;
        const lon = Number(trace.lon[i]);
        const lat = Number(trace.lat[i]);
        if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
        points.push({{
          lon,
          lat,
          customdata,
          threshold: Math.max(24, markerSizeAt(markerSize, i) / 2 + 16)
        }});
      }}
    }});
    return points;
  }};

  const handleManualHover = (mouseEvent) => {{
    const subplot = plot._fullLayout && plot._fullLayout.mapbox && plot._fullLayout.mapbox._subplot;
    const map = subplot && subplot.map;
    const mapElement = plot.querySelector(".mapboxgl-map");
    if (!map || !mapElement) return;

    const rect = mapElement.getBoundingClientRect();
    const mouseX = mouseEvent.clientX - rect.left;
    const mouseY = mouseEvent.clientY - rect.top;
    if (mouseX < 0 || mouseY < 0 || mouseX > rect.width || mouseY > rect.height) {{
      hideTooltip();
      return;
    }}

    let best = null;
    traceHitPoints().forEach((point) => {{
      const projected = map.project([point.lon, point.lat]);
      const distance = Math.hypot(projected.x - mouseX, projected.y - mouseY);
      if (distance <= point.threshold && (!best || distance < best.distance)) {{
        best = {{ ...point, distance }};
      }}
    }});

    if (best) {{
      showTooltip(best.customdata, mouseEvent, "manual");
    }} else {{
      hideTooltip();
    }}
  }};

  Plotly.newPlot(plot, figure.data, layout, config).then(() => {{
    resizePlot();
    enableWheelZoom();
    bindEarlyStarMapEvents();
    scheduleEarlyStarsRetries();
    fullscreenButton.addEventListener("click", async () => {{
      if (document.fullscreenEnabled) {{
        try {{
          if (document.fullscreenElement === wrap) {{
            await document.exitFullscreen();
          }} else {{
            await wrap.requestFullscreen();
          }}
          return;
        }} catch (error) {{}}
      }}
      setExpandedInFrame(!expandedInFrame);
    }});
    document.addEventListener("fullscreenchange", () => {{
      const isFullscreen = document.fullscreenElement === wrap;
      wrap.classList.toggle("is-fullscreen", isFullscreen);
      fullscreenButton.textContent = isFullscreen || expandedInFrame ? "닫기" : "전체화면";
      resizePlot();
      scheduleEarlyStarsRetries();
    }});
    window.addEventListener("resize", () => {{
      if (expandedInFrame) {{
        setFrameHeight(Math.max(window.innerHeight - 28, 720));
      }}
      resizePlot();
      scheduleEarlyStars(120);
    }});
    plot.on("plotly_hover", (eventData) => {{
      const point = (eventData.points || []).find((item) => Array.isArray(item.customdata) && item.customdata.length >= 29);
      if (!point) {{
        return;
      }}
      showTooltip(point.customdata, eventData.event || window.event);
    }});
    plot.addEventListener("mousemove", handleManualHover);
    plot.addEventListener("mouseleave", hideTooltip);
    plot.on("plotly_unhover", () => {{
      window.setTimeout(() => {{
        if (!tooltipPinnedByManual) hideTooltip();
      }}, 80);
    }});
    plot.on("plotly_relayout", () => {{
      hideTooltip();
    }});
  }});
}})();
</script>
<style>
  .custom-map-wrap {{
    position: relative;
    width: 100%;
    height: {height}px;
    overflow: hidden;
    background: #ffffff;
  }}
  .custom-map-wrap.is-fullscreen {{
    width: 100vw;
    height: 100vh;
  }}
  .custom-map-wrap.is-expanded {{
    height: calc(100vh - 10px);
    min-height: 720px;
  }}
  .custom-map-plot {{
    width: 100%;
    height: 100%;
  }}
  .early-star-layer {{
    position: absolute;
    inset: 0;
    z-index: 16;
    pointer-events: none;
  }}
  .early-circle {{
    position: absolute;
    width: 21px;
    height: 21px;
    transform: translate(-50%, -50%);
    border: 2px solid rgba(17, 24, 39, 0.58);
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.03);
    box-shadow:
      0 0 0 1px rgba(255, 255, 255, 0.85),
      0 1px 3px rgba(15, 23, 42, 0.24);
  }}
  .map-fullscreen-button {{
    position: absolute;
    top: 44px;
    right: 12px;
    z-index: 30;
    height: 30px;
    padding: 0 12px;
    border: 1px solid rgba(15, 23, 42, 0.2);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.94);
    color: #0f172a;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.14);
    font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
    font-size: 12px;
    font-weight: 800;
    cursor: pointer;
  }}
  .map-fullscreen-button:hover {{
    background: #f8fafc;
  }}
  .vote-tooltip {{
    position: absolute;
    display: none;
    width: 430px;
    max-height: calc(100% - 20px);
    z-index: 20;
    overflow-y: auto;
    padding: 12px;
    border: 1px solid rgba(15, 23, 42, 0.18);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.26);
    color: #111827;
    font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
    pointer-events: none;
  }}
  .tooltip-head b {{
    display: block;
    font-size: 16px;
    line-height: 1.25;
  }}
  .tooltip-head span {{
    display: block;
    margin-top: 2px;
    color: #64748b;
    font-size: 12px;
  }}
  .tooltip-meta {{
    display: grid;
    gap: 2px;
    margin: 8px 0 10px;
    color: #334155;
    font-size: 12px;
    line-height: 1.35;
  }}
  .tooltip-meta b {{
    color: #0f172a;
  }}
  .unit-panel {{
    margin: 8px 0 10px;
    padding: 8px;
    border: 1px solid rgba(148, 163, 184, 0.36);
    border-radius: 8px;
    background: #ffffff;
  }}
  .unit-title {{
    margin-bottom: 6px;
    color: #0f172a;
    font-size: 12px;
    font-weight: 800;
  }}
  .unit-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 5px 10px;
    color: #334155;
    font-size: 11px;
    line-height: 1.3;
  }}
  .unit-grid div {{
    min-width: 0;
  }}
  .unit-grid b,
  .unit-grid span {{
    display: block;
  }}
  .unit-grid b {{
    color: #64748b;
    font-weight: 700;
  }}
  .unit-grid span {{
    color: #0f172a;
    font-weight: 800;
  }}
  .mini-title {{
    margin: 4px 0 7px;
    color: #0f172a;
    font-size: 13px;
    font-weight: 800;
  }}
  .vote-row {{
    padding: 8px 8px 9px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 8px;
    background: #f8fafc;
  }}
  .vote-row + .vote-row {{
    margin-top: 8px;
  }}
  .vote-row-title {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 6px;
    color: #0f172a;
    font-size: 12px;
    font-weight: 800;
  }}
  .vote-row-title span {{
    color: #64748b;
    font-weight: 600;
  }}
  .model-note {{
    margin-top: 7px;
    color: #64748b;
    font-size: 10px;
    line-height: 1.3;
  }}
  .vote-row-body {{
    display: grid;
    grid-template-columns: 92px minmax(120px, 1fr) 92px;
    align-items: center;
    gap: 8px;
  }}
  .candidate {{
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }}
  .candidate-right {{
    justify-content: flex-end;
    text-align: right;
  }}
  .candidate img {{
    width: 34px;
    height: 34px;
    object-fit: cover;
    border-radius: 50%;
    border: 1px solid rgba(15, 23, 42, 0.14);
    background: #e2e8f0;
  }}
  .candidate b,
  .candidate strong,
  .candidate em {{
    display: block;
    white-space: nowrap;
  }}
  .candidate b {{
    font-size: 11px;
  }}
  .candidate strong {{
    font-size: 16px;
    line-height: 1.05;
  }}
  .candidate em {{
    color: #64748b;
    font-size: 10px;
    font-style: normal;
  }}
  .bar-shell {{
    position: relative;
    height: 28px;
    overflow: hidden;
    border-radius: 4px;
    background: #b8b8b8;
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
  }}
  .bar-fill {{
    position: absolute;
    top: 0;
    bottom: 0;
  }}
  .bar-fill.lee {{
    left: 0;
    background: #2f57b8;
  }}
  .bar-fill.kim {{
    right: 0;
    background: #c94f4f;
  }}
</style>
"""
    components.html(html, height=height, scrolling=False)


def _missing_priority_names(area: dict[str, Any], point_df: pd.DataFrame) -> list[str]:
    wanted = {
        f"{emd_name}{suffix}"
        for group in area["priority_groups"]
        for emd_name, suffixes in group["places"]
        for suffix in suffixes
    }
    found = set(point_df["투표소명_F"].dropna().astype("string")) if not point_df.empty else set()
    return sorted(wanted - found)


def _estimate_model_performance_frame(polling_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["모델", "대상", "역할", "입력변수", "학습 n", "R²", "표본내 MAE", "표본내 RMSE", "LOOCV MAE", "LOOCV RMSE"]
    training = _build_estimation_training_frame(polling_metrics)
    rows: list[dict[str, object]] = []
    for spec in _estimation_model_specs():
        features = list(spec["features"])
        model = _fit_linear_model(training, str(spec["target"]), features)
        loocv = _loocv_prediction_frame(training, str(spec["target"]), features)
        loocv_residual = pd.to_numeric(loocv.get("잔차"), errors="coerce").dropna()
        rows.append(
            {
                "모델": spec["model"],
                "대상": spec["target_label"],
                "역할": spec["role"],
                "입력변수": ", ".join(_model_feature_label(feature) for feature in features),
                "학습 n": model["n"],
                "R²": f"{float(model['r2']):.3f}" if pd.notna(model["r2"]) else "-",
                "표본내 MAE": _format_percent(model["mae"]),
                "표본내 RMSE": _format_percent(model["rmse"]),
                "LOOCV MAE": _format_percent(loocv_residual.abs().mean()) if not loocv_residual.empty else "-",
                "LOOCV RMSE": _format_percent(math.sqrt(float((loocv_residual**2).mean()))) if not loocv_residual.empty else "-",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _estimate_model_coefficients_frame(polling_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["모델", "대상", "항목", "계수"]
    training = _build_estimation_training_frame(polling_metrics)
    rows: list[dict[str, object]] = []
    for spec in _estimation_model_specs():
        features = list(spec["features"])
        model = _fit_linear_model(training, str(spec["target"]), features)
        rows.append(
            {
                "모델": spec["model"],
                "대상": spec["target_label"],
                "항목": "절편",
                "계수": f"{float(model['intercept']):+.5f}" if pd.notna(model["intercept"]) else "-",
            }
        )
        for feature in features:
            value = model["coefs"].get(feature, float("nan")) if isinstance(model["coefs"], dict) else float("nan")
            rows.append(
                {
                    "모델": spec["model"],
                    "대상": spec["target_label"],
                    "항목": _model_feature_label(feature),
                    "계수": f"{float(value):+.5f}" if pd.notna(value) else "-",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _estimate_model_validation_frame(polling_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["모델", "대상", "읍면동", "실제", "예측", "잔차"]
    training = _build_estimation_training_frame(polling_metrics)
    rows: list[dict[str, object]] = []
    for spec in _estimation_model_specs():
        if spec["model"] not in {"B", "D"}:
            continue
        loocv = _loocv_prediction_frame(training, str(spec["target"]), list(spec["features"]))
        for row in loocv.itertuples(index=False):
            rows.append(
                {
                    "모델": spec["model"],
                    "대상": spec["target_label"],
                    "읍면동": row.읍면동,
                    "실제": row.실제,
                    "예측": row.예측,
                    "잔차": row.잔차,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _display_validation_frame(validation: pd.DataFrame) -> pd.DataFrame:
    display = validation.copy()
    if display.empty:
        return display
    display["실제"] = display["실제"].map(_format_percent)
    display["예측"] = display["예측"].map(_format_percent)
    display["잔차"] = display["잔차"].map(_format_percentage_point)
    return display


def _validation_scatter_figure(validation: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if validation.empty:
        return fig
    for target_label, group in validation.groupby("대상", observed=True):
        fig.add_trace(
            go.Scatter(
                x=group["실제"],
                y=group["예측"],
                mode="markers",
                name=str(target_label),
                text=group["읍면동"],
                hovertemplate="%{text}<br>실제 %{x:.1%}<br>예측 %{y:.1%}<extra></extra>",
            )
        )
    values = pd.concat([pd.to_numeric(validation["실제"], errors="coerce"), pd.to_numeric(validation["예측"], errors="coerce")]).dropna()
    if not values.empty:
        lower = float(values.min()) - 0.02
        upper = float(values.max()) + 0.02
        fig.add_shape(type="line", x0=lower, y0=lower, x1=upper, y1=upper, line=dict(color="#64748b", dash="dash"))
        fig.update_xaxes(range=[lower, upper], tickformat=".0%")
        fig.update_yaxes(range=[lower, upper], tickformat=".0%")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="실제", yaxis_title="LOOCV 예측")
    return fig


def _validation_residual_figure(validation: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if validation.empty:
        return fig
    for target_label, group in validation.groupby("대상", observed=True):
        fig.add_trace(
            go.Scatter(
                x=group["읍면동"],
                y=group["잔차"],
                mode="markers",
                name=str(target_label),
                hovertemplate="%{x}<br>잔차 %{y:+.1%}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="#64748b", line_dash="dash")
    fig.update_yaxes(tickformat="+.0%")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=70), xaxis_title="", yaxis_title="실제 - 예측")
    return fig


def _estimate_total_error_frame(polling_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["구분", "모델 추정 총량", "실제 총량", "차이", "차이율", "모델 득표율", "실제 득표율", "득표율 차이"]
    if polling_metrics.empty:
        return pd.DataFrame(columns=columns)

    model_rows = polling_metrics.loc[pd.to_numeric(polling_metrics.get("추정합산_유효투표수"), errors="coerce").notna()].copy()
    if model_rows.empty:
        return pd.DataFrame(columns=columns)

    model_valid = pd.to_numeric(model_rows.get("추정합산_유효투표수"), errors="coerce").sum()
    actual_valid_values = pd.to_numeric(model_rows.get("전체실제_유효투표수"), errors="coerce").dropna()
    actual_valid = float(actual_valid_values.iloc[0]) if not actual_valid_values.empty else float("nan")
    rows: list[dict[str, object]] = []

    def _append_row(label: str, estimated: float, actual: float, model_rate: float | None = None, actual_rate: float | None = None) -> None:
        diff = estimated - actual if pd.notna(actual) else float("nan")
        rows.append(
            {
                "구분": label,
                "모델 추정 총량": _format_number(estimated),
                "실제 총량": _format_number(actual) if pd.notna(actual) else "-",
                "차이": _format_number(diff) if pd.notna(diff) else "-",
                "차이율": _format_percent(diff / actual) if pd.notna(diff) and actual else "-",
                "모델 득표율": _format_percent(model_rate) if model_rate is not None and pd.notna(model_rate) else "-",
                "실제 득표율": _format_percent(actual_rate) if actual_rate is not None and pd.notna(actual_rate) else "-",
                "득표율 차이": _format_percentage_point(model_rate - actual_rate)
                if model_rate is not None and actual_rate is not None and pd.notna(model_rate) and pd.notna(actual_rate)
                else "-",
            }
        )

    _append_row("유효투표수", float(model_valid), actual_valid)
    for candidate_name in CANDIDATE_PAIR:
        estimated = pd.to_numeric(model_rows.get(f"추정합산_{candidate_name}득표수"), errors="coerce").sum()
        actual_values = pd.to_numeric(model_rows.get(f"전체실제_{candidate_name}득표수"), errors="coerce").dropna()
        actual = float(actual_values.iloc[0]) if not actual_values.empty else float("nan")
        model_rate = estimated / model_valid if model_valid else float("nan")
        actual_rate = actual / actual_valid if pd.notna(actual) and actual_valid else float("nan")
        _append_row(candidate_name, float(estimated), actual, model_rate, actual_rate)
    return pd.DataFrame(rows, columns=columns)


def _outside_adjustment_summary_frame(polling_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["항목", "값", "비중/득표율"]
    if polling_metrics.empty:
        return pd.DataFrame(columns=columns)
    model_rows = polling_metrics.loc[pd.to_numeric(polling_metrics.get("추정합산_유효투표수"), errors="coerce").notna()].copy()
    if model_rows.empty:
        return pd.DataFrame(columns=columns)

    def _first_number(column: str) -> float:
        values = pd.to_numeric(model_rows.get(column), errors="coerce").dropna()
        return float(values.iloc[0]) if not values.empty else float("nan")

    non_day_valid = _first_number("관외보정_비선거일유효투표수")
    local_valid = _first_number("관외보정_관내사전유효투표수")
    outside_valid = _first_number("관외보정_관외기타유효투표수")
    outside_share = _first_number("관외보정_관외비중")
    rows = [
        {"항목": "전체 비선거일 유효투표수", "값": _format_number(non_day_valid), "비중/득표율": "100.0%"},
        {"항목": "관내사전 유효투표수", "값": _format_number(local_valid), "비중/득표율": _format_percent(1 - outside_share) if pd.notna(outside_share) else "-"},
        {"항목": "관외/기타 유효투표수", "값": _format_number(outside_valid), "비중/득표율": _format_percent(outside_share)},
    ]
    for candidate_name in CANDIDATE_PAIR:
        rows.append(
            {
                "항목": f"관외/기타 {candidate_name} 실제 득표율",
                "값": _format_number(_first_number(f"관외실제_{candidate_name}득표수")),
                "비중/득표율": _format_percent(_first_number(f"관외실제_{candidate_name}득표율")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _render_estimation_methodology(polling_metrics: pd.DataFrame) -> None:
    with st.expander("사전투표 모델 및 총량 오차 검증", expanded=False):
        st.markdown(
            """
            **화면 해석**

            - 첫 번째 막대는 해당 투표구의 선거일 실제 개표결과입니다.
            - 두 번째 막대는 해당 투표구가 속한 읍면동의 관내사전투표 실제 개표결과입니다.
            - 세 번째 막대는 해당 투표구의 `선거일 실제 결과 + 사전투표 모델 추정치`를 합친 결과입니다.

            **추정 방식**

            - 사전투표 후보별 득표율은 읍면동 관내사전투표 자료를 기준으로 추정했습니다.
            - 투표구별로는 선거일 후보 득표율, 선거일투표율, 사전투표율 구조를 반영했습니다.
            - 다만 관내사전투표만 기준으로 삼으면 이재명 쪽으로 과대 추정될 수 있어 관외/기타 보정을 추가했습니다.
            - 전체 비선거일 유효투표 중 관외/기타 유효투표 비중을 계산하고, 이 비중을 모든 투표구에 공통 적용했습니다.
            - 최종 비선거일 후보 득표율은 `관내비중 × 관내모형 추정 득표율 + 관외비중 × 관외/기타 실제 득표율`입니다.
            - 투표구별 실제 관외투표 비중은 알 수 없으므로 공통 비중을 적용한 보정입니다.
            - 전체 합계에 억지로 맞추는 후보별 사후 보정은 하지 않았습니다.

            **오차 검증**

            - 아래 표는 모든 투표구의 모델 추정치를 합산한 총량과 실제 여주시 전체 개표결과를 비교한 것입니다.
            - 즉, 개별 투표구 단위가 아니라 전체 합계 기준으로 어느 정도 차이가 나는지 보는 표입니다.
            """
        )
        st.caption("관외/기타 보정 기준")
        st.dataframe(_outside_adjustment_summary_frame(polling_metrics), use_container_width=True, hide_index=True)
        st.caption("총량 기준 오차")
        st.dataframe(_estimate_total_error_frame(polling_metrics), use_container_width=True, hide_index=True)


def _render_area(
    index: int,
    area: dict[str, Any],
    polling_df: pd.DataFrame,
    polling_metrics: pd.DataFrame,
    priority_df: pd.DataFrame,
    geometry_context: dict[str, Any],
) -> pd.DataFrame:
    region_df = _area_region_frame(area, polling_df)
    point_df = _area_point_frame(area, polling_metrics, priority_df)
    early_point_df = _area_early_point_frame(area, polling_df)
    st.subheader(f"{index}. {area['title']}")
    st.caption(f"{area['description']} | 표시 단위: 해당 읍면동 경계 + 우선순위 투표소")

    missing_names = _missing_priority_names(area, point_df)
    if missing_names:
        st.warning("우선순위 목록 중 좌표/투표소 매칭이 안 된 항목: " + ", ".join(missing_names))

    _render_hover_chart_map(
        _build_custom_map(area, region_df, point_df, early_point_df, geometry_context),
        key=f"custom_gis_{area['id']}",
    )
    st.dataframe(_priority_summary_frame(area), use_container_width=True, hide_index=True)
    return point_df


AREA_GENERATION_ORDER = [
    ("core_win", "핵심 승리권역"),
    ("expanded_weak", "확장형 열세권역"),
    ("loss_control", "손실관리지역"),
    ("low_efficiency", "저효율 관리지역"),
    ("insufficient_data", "자료부족"),
]
AREA_GENERATION_LABELS = dict(AREA_GENERATION_ORDER)
AREA_PRIORITY_PROFILES = {
    1: "권역 내 선거일 득표수 높음 / 사전투표율 높음",
    2: "권역 내 선거일 득표수 높음 / 사전투표율 낮음",
    3: "권역 내 선거일 득표수 낮음 / 사전투표율 높음",
    4: "권역 내 선거일 득표수 낮음 / 사전투표율 낮음",
}


def _target_region_options(app_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    polling = app_data.get("polling", pd.DataFrame())
    votes = app_data.get("votes", pd.DataFrame())
    if isinstance(polling, pd.DataFrame) and not polling.empty and {"선거시점", "시도명_F", "구시군명_F"}.issubset(polling.columns):
        polling_current = polling.loc[polling["선거시점"].astype("string").str.startswith(CURRENT_PERIOD, na=False)]
        frames.append(
            polling_current.loc[:, ["시도명_F", "구시군명_F"]].rename(
                columns={"시도명_F": "시도명", "구시군명_F": "구시군명"}
            )
        )
    if isinstance(votes, pd.DataFrame) and not votes.empty and {"선거KEY", "시도명", "구시군명"}.issubset(votes.columns):
        votes_current = votes.loc[votes["선거KEY"].astype("string").eq(CURRENT_ELECTION_KEY)]
        frames.append(votes_current.loc[:, ["시도명", "구시군명"]])

    if not frames:
        return pd.DataFrame(columns=["시도명", "구시군명", "key", "label"])
    options = pd.concat(frames, ignore_index=True).dropna(subset=["시도명", "구시군명"]).drop_duplicates()
    options["시도명"] = options["시도명"].astype("string").str.strip()
    options["구시군명"] = options["구시군명"].astype("string").str.strip()
    options = options.loc[options["시도명"].ne("") & options["구시군명"].ne("")]
    options = options.sort_values(["시도명", "구시군명"], kind="stable").reset_index(drop=True)
    options["key"] = options["시도명"] + "||" + options["구시군명"]
    options["label"] = options["시도명"] + " " + options["구시군명"]
    return options


def _select_repro_target_region(options: pd.DataFrame) -> tuple[str, str]:
    if options.empty:
        raise ValueError("202506-P 기준으로 선택 가능한 구시군 데이터를 찾지 못했습니다.")
    option_keys = options["key"].astype("string").tolist()
    key_to_label = dict(zip(options["key"].astype("string"), options["label"].astype("string")))
    preferred_key = f"{TARGET_SIDO}||{TARGET_GUSIGUN}"
    if preferred_key not in option_keys:
        preferred_key = str(option_keys[0])
    state_key = "repro_custom_gis_target_region"
    if st.session_state.get(state_key) not in option_keys:
        st.session_state[state_key] = preferred_key
    selected_key = st.selectbox(
        "분석 구시군",
        options=option_keys,
        index=option_keys.index(st.session_state[state_key]),
        key=state_key,
        format_func=lambda value: key_to_label.get(str(value), str(value)),
    )
    sido, gusigun = str(selected_key).split("||", maxsplit=1)
    return sido, gusigun


def _build_historical_alias_map(alias_df: pd.DataFrame, target_sido: str, target_gusigun: str) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    if not alias_df.empty and {"시도명_F", "구시군명_F", "읍면동명_F", "시도명_D", "구시군명_D", "읍면동명_D"}.issubset(alias_df.columns):
        base = alias_df.loc[
            (
                alias_df["시도명_D"].astype("string").eq(target_sido)
                & alias_df["구시군명_D"].astype("string").eq(target_gusigun)
            )
            | (
                alias_df["시도명_F"].astype("string").eq(target_sido)
                & alias_df["구시군명_F"].astype("string").eq(target_gusigun)
            )
        ].copy()
        for _, row in base.iterrows():
            source = row.get("읍면동명_F")
            target = row.get("읍면동명_D")
            if pd.notna(source) and pd.notna(target) and str(source).strip() and str(target).strip() and str(source) != str(target):
                rename_map[str(source)] = str(target)
    if target_sido == "경기도" and target_gusigun == "여주시":
        rename_map.update({"능서면": "세종대왕면"})
    return rename_map


def _configure_repro_context(target_sido: str, target_gusigun: str, alias_df: pd.DataFrame) -> None:
    global TARGET_SIDO, TARGET_GUSIGUN, POLLING_JURISDICTION_FILE, HISTORICAL_EMD_NAME_RENAMES, POLLING_PLACE_COORDINATE_OVERRIDES
    TARGET_SIDO = target_sido
    TARGET_GUSIGUN = target_gusigun
    POLLING_JURISDICTION_FILE = Path.home() / "Desktop" / f"{target_gusigun} 투표구 관할구역.xlsx"
    HISTORICAL_EMD_NAME_RENAMES = _build_historical_alias_map(alias_df, target_sido, target_gusigun)
    POLLING_PLACE_COORDINATE_OVERRIDES = (
        dict(DEFAULT_POLLING_PLACE_COORDINATE_OVERRIDES)
        if target_sido == "경기도" and target_gusigun == "여주시"
        else {}
    )


def _build_strategy_seed_frame(votes_df: pd.DataFrame, polling_df: pd.DataFrame) -> pd.DataFrame:
    vote_rates = _historical_vote_rate_frame(votes_df)
    if not polling_df.empty and "읍면동명_F" in polling_df.columns:
        emd_names = polling_df["읍면동명_F"].dropna().astype("string").drop_duplicates().tolist()
    elif not vote_rates.empty:
        emd_names = vote_rates["읍면동명"].dropna().astype("string").drop_duplicates().tolist()
    else:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for order, emd_name in enumerate(emd_names):
        emd_base = vote_rates.loc[vote_rates["읍면동명"].astype("string").eq(str(emd_name))].copy()
        row = _historical_summary_row(str(emd_name), emd_base, {"읍면동명": str(emd_name), "__order": order})
        rows.append(row)
    seed = pd.DataFrame(rows)
    if seed.empty:
        return seed

    avg_gap = pd.to_numeric(seed["평균 격차"], errors="coerce")
    recent_flow = pd.to_numeric(seed["최근 흐름"], errors="coerce")
    negative = seed.loc[avg_gap.lt(0)].copy()
    negative_flows = pd.to_numeric(negative["최근 흐름"], errors="coerce").dropna()
    recovery_cut = max(0.0, float(negative_flows.median())) if not negative_flows.empty else 0.0
    negative_gaps = pd.to_numeric(negative["평균 격차"], errors="coerce").dropna()
    severe_loss_cut = float(negative_gaps.quantile(0.30)) if len(negative_gaps) >= 3 else -0.20

    def _classify(row: pd.Series) -> str:
        average = row.get("평균 격차")
        flow = row.get("최근 흐름")
        if pd.isna(average):
            return "insufficient_data"
        if float(average) >= 0:
            return "core_win"
        if pd.notna(flow) and float(flow) > 0 and float(flow) >= recovery_cut:
            return "expanded_weak"
        if float(average) <= severe_loss_cut:
            return "low_efficiency"
        return "loss_control"

    seed["권역ID"] = seed.apply(_classify, axis=1)
    seed["권역"] = seed["권역ID"].map(AREA_GENERATION_LABELS)
    seed["권역분류근거"] = seed.apply(
        lambda row: (
            f"평균 격차 {_format_percentage_point(row.get('평균 격차'))} / "
            f"최근 흐름 {_format_percentage_point(row.get('최근 흐름'))}"
        ),
        axis=1,
    )
    return seed.sort_values("__order", kind="stable").reset_index(drop=True)


def _polling_suffix(emd_name: object, polling_name: object) -> str:
    emd_text = "" if pd.isna(emd_name) else str(emd_name)
    place_text = "" if pd.isna(polling_name) else str(polling_name)
    if emd_text and place_text.startswith(emd_text):
        suffix = place_text[len(emd_text):]
        return suffix or place_text
    return place_text


def _build_generated_priority_groups(area_emd_names: list[str], polling_metrics: pd.DataFrame) -> list[dict[str, object]]:
    if polling_metrics.empty or not {"읍면동명_F", "투표소명_F"}.issubset(polling_metrics.columns):
        return []
    base = polling_metrics.loc[
        polling_metrics["읍면동명_F"].astype("string").isin(area_emd_names)
        & ~polling_metrics["투표소명_F"].astype("string").str.endswith("사전투표소", na=False)
    ].copy()
    if base.empty:
        return []

    target_candidate = CANDIDATE_PAIR[0]
    day_col = f"선거일_{target_candidate}득표수"
    if day_col not in base.columns or pd.to_numeric(base[day_col], errors="coerce").dropna().empty:
        day_col = f"{target_candidate}득표수" if f"{target_candidate}득표수" in base.columns else "투표수"
    early_col = "사전투표율" if "사전투표율" in base.columns else "사전투표 비중"
    base["__day_votes"] = pd.to_numeric(base[day_col], errors="coerce")
    base["__early_rate"] = pd.to_numeric(base[early_col], errors="coerce") if early_col in base.columns else np.nan
    day_cut = float(base["__day_votes"].median(skipna=True)) if base["__day_votes"].notna().any() else 0.0
    early_cut = float(base["__early_rate"].median(skipna=True)) if base["__early_rate"].notna().any() else 0.0

    def _rank(row: pd.Series) -> int:
        high_day = pd.notna(row["__day_votes"]) and float(row["__day_votes"]) >= day_cut
        high_early = pd.notna(row["__early_rate"]) and float(row["__early_rate"]) >= early_cut
        if high_day and high_early:
            return 1
        if high_day and not high_early:
            return 2
        if not high_day and high_early:
            return 3
        return 4

    base["__rank"] = base.apply(_rank, axis=1)
    emd_order = {name: index for index, name in enumerate(area_emd_names)}
    base["__emd_order"] = base["읍면동명_F"].map(emd_order).fillna(99)
    base = base.sort_values(["__rank", "__emd_order", "투표소명_F"], kind="stable")

    groups: list[dict[str, object]] = []
    for rank in [1, 2, 3, 4]:
        rank_base = base.loc[base["__rank"].eq(rank)].copy()
        if rank_base.empty:
            continue
        places: list[tuple[str, list[str]]] = []
        for emd_name, emd_group in rank_base.groupby("읍면동명_F", observed=True, sort=False):
            suffixes = [
                _polling_suffix(emd_name, polling_name)
                for polling_name in emd_group["투표소명_F"].dropna().astype("string").drop_duplicates().tolist()
            ]
            if suffixes:
                places.append((str(emd_name), suffixes))
        groups.append(
            {
                "rank": rank,
                "profile": AREA_PRIORITY_PROFILES[rank],
                "places": places,
            }
        )
    return groups


def _build_generated_strategy_areas(
    votes_df: pd.DataFrame,
    polling_df: pd.DataFrame,
    polling_metrics: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    seed = _build_strategy_seed_frame(votes_df, polling_df)
    if seed.empty:
        return [], seed

    areas: list[dict[str, Any]] = []
    for area_id, title in AREA_GENERATION_ORDER:
        area_seed = seed.loc[seed["권역ID"].astype("string").eq(area_id)].sort_values("__order", kind="stable")
        if area_seed.empty:
            continue
        emd_names = area_seed["읍면동명"].dropna().astype("string").tolist()
        areas.append(
            {
                "id": area_id,
                "title": title,
                "description": ", ".join(emd_names),
                "emd_names": emd_names,
                "priority_groups": _build_generated_priority_groups(emd_names, polling_metrics),
            }
        )
    return areas, seed


def main() -> None:
    global CUSTOM_GIS_AREAS

    st.title("GIS분석(재현)")

    try:
        with st.spinner("복제 GIS 데이터를 불러오는 중입니다..."):
            app_data = _load_bundle()
    except FileNotFoundError:
        st.error("먼저 python scripts/build_parquet.py 를 실행해 주세요.")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.caption("cache parquet 스키마가 바뀌었을 수 있습니다. python scripts/build_parquet.py 를 다시 실행해 주세요.")
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.caption("cache parquet를 다시 생성할 필요가 있는지 확인해 주세요.")
        st.stop()

    try:
        target_sido, target_gusigun = _select_repro_target_region(_target_region_options(app_data))
        _configure_repro_context(target_sido, target_gusigun, app_data["dong_alias"])
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    st.caption(f"{CURRENT_ELECTION_LABEL} / {TARGET_SIDO} {TARGET_GUSIGUN}")
    st.caption("현재 GIS분석 페이지를 복제한 구조에서 권역 구분과 투표구 우선순위를 선택 구시군 데이터로 재산출합니다.")

    try:
        with st.spinner("권역과 투표구 우선순위를 재산출하는 중입니다..."):
            current_confirmed = _current_election_frame(app_data["confirmed"])
            current_turnout = _current_election_frame(app_data["turnout"])
            current_votes = _current_election_frame(app_data["votes"])
            current_polling = _current_polling_frame(app_data["polling"])
            polling_metrics = _prepare_polling_metrics(current_turnout, current_votes, current_polling, current_confirmed)
            CUSTOM_GIS_AREAS, strategy_seed = _build_generated_strategy_areas(app_data["votes"], current_polling, polling_metrics)
            priority_df = _build_priority_frame()
            geometry_context = load_geometry_context("읍면동")
    except Exception as exc:
        st.error(f"재현 GIS 산출 중 오류가 발생했습니다: {exc}")
        st.stop()

    if current_confirmed.empty or current_turnout.empty or current_votes.empty or current_polling.empty:
        st.warning(f"{CURRENT_ELECTION_LABEL} / {TARGET_SIDO} {TARGET_GUSIGUN} 기준 GIS 데이터를 찾지 못했습니다.")
        if current_confirmed.empty:
            st.code(
                "python scripts/fetch_confirmed_electorate.py --sgid 20250603\n"
                "python scripts/build_parquet.py",
                language="powershell",
            )
        st.stop()

    if not CUSTOM_GIS_AREAS:
        st.warning("선택 구시군의 읍면동 권역을 산출할 수 없습니다.")
        st.stop()

    if not geometry_context.get("available"):
        st.warning(geometry_context.get("message", "읍면동 경계 파일을 찾지 못했습니다. 투표소 포인트만 표시됩니다."))
    else:
        st.caption(geometry_context["message"])

    if not strategy_seed.empty:
        with st.expander("재현 권역 산출 근거", expanded=False):
            display_columns = [
                "권역",
                "구분",
                "18도지사 민주",
                "18도지사 국힘",
                "24지역구 민주",
                "24지역구 국힘",
                "25대선 민주",
                "25대선 국힘",
                "평균 격차",
                "최근 흐름 요약",
                "권역분류근거",
            ]
            display_columns = [column for column in display_columns if column in strategy_seed.columns]
            st.dataframe(
                _style_historical_summary(strategy_seed.loc[:, display_columns]),
                use_container_width=True,
                hide_index=True,
            )

    _render_historical_strategy_summary(app_data["votes"])
    _render_estimation_methodology(polling_metrics)

    all_points: list[pd.DataFrame] = []
    for index, area in enumerate(CUSTOM_GIS_AREAS, start=1):
        point_df = _render_area(index, area, current_polling, polling_metrics, priority_df, geometry_context)
        if not point_df.empty:
            all_points.append(point_df)
        if index < len(CUSTOM_GIS_AREAS):
            st.divider()

    if all_points:
        export_df = pd.concat(all_points, ignore_index=True)
        export_columns = [
            "권역",
            "우선순위라벨",
            "성격",
            "읍면동명_F",
            "투표소명_F",
            "장소명",
            "주소",
            "위도",
            "경도",
            "표시위도",
            "표시경도",
            "확정선거인수",
            "선거일투표수",
            "선거일투표율",
            "사전투표수",
            "사전투표율",
            "사전투표비중_투표구",
            "이재명득표수",
            "이재명득표율",
            "사전투표 비중",
            "투표수",
            "관외보정_관내비중",
            "관외보정_관외비중",
            "관외보정_비선거일유효투표수",
            "관외보정_관내사전유효투표수",
            "관외보정_관외기타유효투표수",
            "관내모형_이재명득표율",
            "관외실제_이재명득표수",
            "관외실제_이재명득표율",
            "관내모형_김문수득표율",
            "관외실제_김문수득표수",
            "관외실제_김문수득표율",
            "사전추정_유효투표수",
            "사전추정_이재명득표수",
            "사전추정_이재명득표율",
            "사전추정_김문수득표수",
            "사전추정_김문수득표율",
            "사전프리미엄_이재명",
            "사전프리미엄_김문수",
            "사전마진프리미엄",
            "마진모델_사전마진프리미엄",
            "추정합산_이재명득표수",
            "추정합산_이재명득표율",
            "추정합산_김문수득표수",
            "추정합산_김문수득표율",
            "추정합산_유효투표수",
            "추정모델설명",
            "1위후보",
            "1위정당",
        ]
        export_columns = [column for column in export_columns if column in export_df.columns]
        st.download_button(
            "재현 GIS 우선순위 투표소 CSV 다운로드",
            dataframe_to_csv_bytes(export_df.loc[:, export_columns]),
            file_name=f"repro_custom_gis_priority_{CURRENT_ELECTION_KEY}_{TARGET_SIDO}_{TARGET_GUSIGUN}.csv",
            mime="text/csv",
            use_container_width=True,
        )


main()
