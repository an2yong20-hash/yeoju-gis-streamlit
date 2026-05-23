from __future__ import annotations

from typing import Any

import streamlit as st

from src.charts import lollipop_chart
from src.filters import apply_common_filters, build_breadcrumb_text, render_sidebar_filters
from src.loaders import get_cache_file_signatures, load_cached_dims, load_fact_turnout_enriched, load_fact_votes_enriched, load_global_filter_options
from src.maps import BASEMAP_STYLE_OPTIONS, build_polling_point_map
from src.metrics import calc_polling_station_metrics, dataframe_to_csv_bytes, format_int

POLLING_DIM_COLUMNS = {
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
        "층수",
        "법정동",
        "위도",
        "경도",
        "읍면동KEY_D",
    ],
}

TURNOUT_COLUMNS = (
    "선거KEY",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "RowType",
    "투표소KEY",
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
)

VOTES_COLUMNS = (
    "선거KEY",
    "정당KEY",
    "선거구명",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "구분",
    "RowType",
    "투표소KEY",
    "정당명",
    "후보명",
    "후보라벨",
    "유효투표수",
    "득표수",
)


def _load_bundle() -> dict[str, object]:
    cache_signature = get_cache_file_signatures()
    dims = load_cached_dims(cache_signature, dim_names=("DimPollingPlace",), columns_map=POLLING_DIM_COLUMNS)
    return {
        "polling": dims["DimPollingPlace"],
        "turnout": load_fact_turnout_enriched(cache_signature, TURNOUT_COLUMNS),
        "votes": load_fact_votes_enriched(cache_signature, VOTES_COLUMNS),
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


def _resolve_current_election_key(filtered_turnout, filtered_votes, filter_options: dict[str, object]) -> str:
    vote_keys = filtered_votes["선거KEY"].dropna().astype("string").tolist() if "선거KEY" in filtered_votes.columns else []
    turnout_keys = filtered_turnout["선거KEY"].dropna().astype("string").tolist() if "선거KEY" in filtered_turnout.columns else []
    option_keys = sorted(set([*vote_keys, *turnout_keys]), reverse=True)
    if not option_keys:
        raise ValueError("현재 필터 조건에 맞는 선거를 찾을 수 없습니다.")
    if len(option_keys) == 1:
        return option_keys[0]

    key_to_label = filter_options.get("election_key_to_label", {})
    current_value = str(st.session_state.get("polling_current_election_key", option_keys[0]))
    if current_value not in option_keys:
        current_value = option_keys[0]
        st.session_state["polling_current_election_key"] = current_value

    selected_key = st.selectbox(
        "현재 선거",
        options=option_keys,
        index=option_keys.index(current_value),
        key="polling_current_election_key",
        format_func=lambda value: str(key_to_label.get(str(value), value)),
    )
    st.caption("여러 선거가 같은 시점에 포함되어 있어 현재 분석 선거를 직접 선택하도록 표시했습니다.")
    return str(selected_key)


def main() -> None:
    st.title("투표소")

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
    filtered_polling = apply_common_filters(app_data["polling"], selected)
    filtered_turnout = apply_common_filters(app_data["turnout"], selected)
    filtered_votes = apply_common_filters(app_data["votes"], selected)

    if filtered_polling.empty:
        st.warning("현재 필터 조건에 맞는 투표소 데이터가 없습니다.")
        st.stop()

    current_election_key = _resolve_current_election_key(filtered_turnout, filtered_votes, filter_options)
    current_period = current_election_key[:6]
    current_polling = filtered_polling.loc[filtered_polling["투표소KEY"].astype("string").str[:6] == str(current_period)].copy()
    current_turnout = filtered_turnout.loc[filtered_turnout["선거KEY"].astype("string") == current_election_key].copy()
    current_votes = filtered_votes.loc[filtered_votes["선거KEY"].astype("string") == current_election_key].copy()
    if current_polling.empty:
        st.warning(f"{current_period} 시점에는 투표소 위치 데이터가 없어 투표소 지도를 표시할 수 없습니다.")
        st.stop()
    polling_metrics = calc_polling_station_metrics(current_turnout, current_votes, current_polling)

    st.caption("현재 필터: " + build_breadcrumb_text(selected, filter_options))

    size_option = st.selectbox(
        "점 크기",
        ["투표수", "유효투표수", "1위득표수", "2위득표수", "득표수격차", "득표율격차", "경쟁도지수"],
        key="polling_size_option",
    )
    color_option = st.selectbox("점 색상", ["1위정당", "1위후보"], key="polling_color_option")
    basemap_labels = list(BASEMAP_STYLE_OPTIONS.keys())
    basemap_option = st.selectbox(
        "베이스맵",
        basemap_labels,
        index=basemap_labels.index("배경 없음") if "배경 없음" in basemap_labels else 0,
        key="polling_basemap_option",
    )
    if "투표소KEY" not in current_votes.columns or not current_votes["투표소KEY"].notna().any():
        st.caption("이 선거는 투표소 단위 득표 데이터가 없어 득표 기반 크기/색상은 자동으로 기본값으로 대체됩니다.")

    effective_size_option = size_option if size_option in polling_metrics.columns and polling_metrics[size_option].notna().any() else "투표수"
    effective_color_option = color_option if color_option in polling_metrics.columns and polling_metrics[color_option].notna().any() else None
    has_polling_metric_values = "투표수" in polling_metrics.columns and polling_metrics["투표수"].notna().any()
    if not has_polling_metric_values:
        st.info("선택한 선거는 투표소별 집계와 위치 매핑이 없어 포인트 위치만 표시됩니다.")

    col1, col2, col3 = st.columns(3)
    col1.metric("투표소 수", format_int(polling_metrics["투표소KEY"].nunique()))
    col2.metric("좌표 보유 수", format_int(polling_metrics.loc[polling_metrics["위도"].notna() & polling_metrics["경도"].notna(), "투표소KEY"].nunique()))
    col3.metric("좌표 누락 수", format_int(polling_metrics.loc[polling_metrics["위도"].isna() | polling_metrics["경도"].isna(), "투표소KEY"].nunique()))

    st.plotly_chart(
        build_polling_point_map(
            polling_metrics,
            title=f"{current_period} 투표소 포인트 지도",
            size_col=effective_size_option,
            color_col=effective_color_option or color_option,
            basemap_style=basemap_option,
        ),
        use_container_width=True,
    )
    st.caption("배경 지명 라벨은 베이스맵 제공자에 따라 한글/영문 혼용으로 보일 수 있습니다.")

    ranking_source = polling_metrics.sort_values(by=effective_size_option, ascending=False, kind="stable").head(30).copy()
    ranking_source["투표소라벨"] = ranking_source["투표소명_F"].fillna(ranking_source["투표소KEY"])
    selection = st.plotly_chart(
        lollipop_chart(
            ranking_source,
            category_col="투표소라벨",
            value_col=effective_size_option,
            title=f"{effective_size_option} 기준 상위 투표소",
            percent=False,
            color_col=effective_color_option,
        ),
        use_container_width=True,
        key="polling_rank_chart",
        on_select="rerun",
    )
    selected_station = _extract_selection_value(selection, ("y",))

    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["상세 테이블", "경쟁도 상위", "좌표 누락"])
    with detail_tab1:
        if selected_station:
            detail_df = ranking_source.loc[ranking_source["투표소라벨"].astype("string") == selected_station]
            st.caption(f"선택된 투표소 drill-down: {selected_station}")
        else:
            detail_df = polling_metrics.sort_values(by="투표수", ascending=False, kind="stable").head(200)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    with detail_tab2:
        competition_df = polling_metrics.sort_values(by="경쟁도지수", ascending=False, kind="stable").head(200)
        st.dataframe(competition_df, use_container_width=True, hide_index=True)
    with detail_tab3:
        missing_coords = polling_metrics.loc[polling_metrics["위도"].isna() | polling_metrics["경도"].isna()].copy()
        st.dataframe(missing_coords, use_container_width=True, hide_index=True)

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "투표소 집계 CSV 다운로드",
            dataframe_to_csv_bytes(polling_metrics),
            file_name="polling_station_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "좌표 누락 CSV 다운로드",
            dataframe_to_csv_bytes(polling_metrics.loc[polling_metrics["위도"].isna() | polling_metrics["경도"].isna()]),
            file_name="polling_missing_coordinates.csv",
            mime="text/csv",
            use_container_width=True,
        )


main()
