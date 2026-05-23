from __future__ import annotations

import streamlit as st

from src.charts import bump_chart, entity_color_map, heatmap_chart, line_metric_chart, slope_chart, stacked_bar_chart
from src.filters import apply_common_filters, build_breadcrumb_text, clear_rowtype_filter, render_sidebar_filters, resolve_region_level
from src.loaders import get_cache_file_signatures, load_fact_turnout_enriched, load_fact_votes_enriched, load_global_filter_options
from src.metrics import (
    build_competitiveness_sentence,
    build_delta_sentence,
    calc_entity_trend,
    calc_entity_trend_by_region,
    calc_rowtype_turnout_trend,
    calc_rowtype_vote_trend,
    calc_top2_gap_by_region,
    calc_top2_gap_trend,
    calc_turnout_timeseries,
    calc_turnout_trend_by_level,
    dataframe_to_csv_bytes,
    format_delta_pp,
    format_percent,
    get_latest_and_previous,
)

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
    "RowType",
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
)

VOTES_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "정당KEY",
    "정당명",
    "정당구분",
    "구분2",
    "성향",
    "후보명",
    "후보라벨",
    "선거구명",
    "시도명",
    "구시군명",
    "구시군KEY",
    "일반구명",
    "읍면동명",
    "읍면동KEY",
    "구분",
    "RowType",
    "유효투표수",
    "득표수",
)
ENTITY_TYPE_OPTIONS = ["정당", "후보", "구분", "구분2", "성향"]
ENTITY_LABEL_COLUMNS = {
    "정당": "정당명",
    "후보": "후보명",
    "구분": "정당구분",
    "구분2": "구분2",
    "성향": "성향",
}


def _load_bundle() -> dict[str, object]:
    cache_signature = get_cache_file_signatures()
    return {
        "turnout": load_fact_turnout_enriched(cache_signature, TURNOUT_COLUMNS),
        "votes": load_fact_votes_enriched(cache_signature, VOTES_COLUMNS),
        "filter_options": load_global_filter_options(cache_signature),
    }


def _label_col(entity_type: str) -> str:
    return ENTITY_LABEL_COLUMNS[entity_type]


def main() -> None:
    st.title("추이분석")
    st.info("추이분석은 이제 `투표율`과 `득표` 페이지 내부 탭에서 제공합니다.")
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/2_투표율.py", label="투표율로 이동", use_container_width=True)
    with col2:
        st.page_link("pages/3_득표.py", label="득표로 이동", use_container_width=True)
    st.stop()

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
    analysis_selected = clear_rowtype_filter(selected)
    filtered_turnout = apply_common_filters(app_data["turnout"], analysis_selected)
    filtered_votes = apply_common_filters(app_data["votes"], analysis_selected)

    if filtered_turnout.empty and filtered_votes.empty:
        st.warning("현재 필터 조건에 맞는 추이 데이터가 없습니다.")
        st.stop()

    st.caption("현재 필터: " + build_breadcrumb_text(analysis_selected, filter_options))

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        region_level = st.selectbox("지역 추이 레벨", ["시도", "구시군", "읍면동"], index=["시도", "구시군", "읍면동"].index(resolve_region_level(selected, "구시군")), key="trend_region_level")
    with control_col2:
        entity_type = st.selectbox("추적 기준", ENTITY_TYPE_OPTIONS, index=ENTITY_TYPE_OPTIONS.index("정당"), key="trend_entity_type")
    with control_col3:
        focus_metric = st.selectbox("추이 지표", ["투표율", "유효투표율", "무효투표율"], key="trend_focus_metric")

    st.caption("구분/구분2/성향 추이는 DimParty 정당 분류 기준으로 집계됩니다.")

    turnout_trend = calc_turnout_timeseries(filtered_turnout)
    region_trend = calc_turnout_trend_by_level(filtered_turnout, level=region_level)
    rowtype_turnout_trend = calc_rowtype_turnout_trend(filtered_turnout, family="투표방식")
    rowtype_vote_trend = calc_rowtype_vote_trend(filtered_votes)
    entity_trend = calc_entity_trend(filtered_votes, entity_type=entity_type)
    label_col = _label_col(entity_type)
    entity_options = entity_trend[label_col].dropna().astype("string").unique().tolist()
    selected_entities = st.multiselect(f"{entity_type} 추적 대상", options=entity_options, default=entity_options[: min(6, len(entity_options))], key="trend_selected_entities")
    if selected_entities:
        entity_trend = entity_trend.loc[entity_trend[label_col].astype("string").isin(selected_entities)].copy()

    entity_region_trend = calc_entity_trend_by_region(filtered_votes, entity_type=entity_type, level=region_level)
    if selected_entities:
        entity_region_trend = entity_region_trend.loc[entity_region_trend[label_col].astype("string").isin(selected_entities)].copy()

    entity_rank = entity_trend.sort_values(by=["선거KEY", "득표수"], ascending=[True, False], kind="stable").copy()
    entity_rank["순위"] = entity_rank.groupby("선거KEY", observed=True)["득표수"].rank(method="dense", ascending=False).astype("Int64")
    gap_trend = calc_top2_gap_trend(filtered_votes)
    current_competition = calc_top2_gap_by_region(
        filtered_votes.loc[
            filtered_votes["선거KEY"].astype("string")
            == str(turnout_trend.sort_values(by=["선거시점", "선거KEY"], ascending=[False, False]).head(1)["선거KEY"].iloc[0])
        ].copy()
        if not turnout_trend.empty
        else filtered_votes,
        level=region_level,
    )

    latest_turnout = get_latest_and_previous(turnout_trend, focus_metric)
    primary_entity = selected_entities[0] if selected_entities else None
    primary_entity_trend = entity_trend.loc[entity_trend[label_col].astype("string") == str(primary_entity)].copy() if primary_entity else entity_trend.iloc[0:0]
    latest_entity = get_latest_and_previous(primary_entity_trend, "득표율_계산")

    st.info(build_delta_sentence("선택 범위", focus_metric, latest_turnout["delta"]))
    if primary_entity:
        st.caption(build_delta_sentence(primary_entity, "득표율", latest_entity["delta"]))
    if not gap_trend.empty:
        st.caption(build_competitiveness_sentence(gap_trend.sort_values(by=["선거시점", "선거KEY"], ascending=[False, False]).head(1), "최근 선거"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최근 투표율", format_percent(latest_turnout["current"]), delta=format_delta_pp(latest_turnout["delta"]))
    col2.metric("최근 추적 기준", "-" if primary_entity is None else primary_entity)
    col3.metric("최근 득표율", "-" if primary_entity is None else format_percent(latest_entity["current"]), delta="" if primary_entity is None else format_delta_pp(latest_entity["delta"]))
    col4.metric("최근 경쟁도", "-" if gap_trend.empty else format_percent(gap_trend.sort_values(by=["선거시점", "선거KEY"], ascending=[False, False]).iloc[0]["경쟁도지수"]))

    tab1, tab2, tab3, tab4 = st.tabs(["투표율 추이", "대상 추이", "RowType 추이", "경쟁도"])

    with tab1:
        st.plotly_chart(
            line_metric_chart(
                turnout_trend,
                x_col="선거라벨" if "선거라벨" in turnout_trend.columns else "선거KEY",
                y_col=focus_metric,
                title=f"선거시점별 {focus_metric} 추이",
                percent=True,
            ),
            use_container_width=True,
        )
        compare_regions = region_trend["지역"].drop_duplicates().tolist()[:10]
        region_plot_df = region_trend.loc[region_trend["지역"].isin(compare_regions)].copy()
        st.plotly_chart(
            line_metric_chart(
                region_plot_df,
                x_col="선거라벨" if "선거라벨" in region_plot_df.columns else "선거KEY",
                y_col=focus_metric,
                color_col="지역",
                title=f"{region_level} 다중 비교: {focus_metric}",
                percent=True,
            ),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            line_metric_chart(
                entity_trend,
                x_col="선거라벨" if "선거라벨" in entity_trend.columns else "선거KEY",
                y_col="득표율_계산",
                color_col=label_col,
                title=f"{entity_type} 묶음 득표율 추이",
                percent=True,
                color_map=entity_color_map(entity_trend, label_col),
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            slope_chart(
                entity_trend,
                entity_col=label_col,
                x_col="선거라벨" if "선거라벨" in entity_trend.columns else "선거KEY",
                y_col="득표율_계산",
                title=f"{entity_type} 묶음 득표율 slope chart",
                percent=True,
                color_map=entity_color_map(entity_trend, label_col),
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            bump_chart(
                entity_rank,
                entity_col=label_col,
                x_col="선거라벨" if "선거라벨" in entity_rank.columns else "선거KEY",
                rank_col="순위",
                title=f"{entity_type} 묶음 순위 변화 bump chart",
                color_map=entity_color_map(entity_rank, label_col),
            ),
            use_container_width=True,
        )
        if not entity_region_trend.empty:
            region_heat_source = entity_region_trend.loc[entity_region_trend[label_col].astype("string") == str(primary_entity)].copy() if primary_entity else entity_region_trend.head(0)
            if not region_heat_source.empty:
                st.plotly_chart(
                    heatmap_chart(
                        region_heat_source,
                        x_col="선거라벨" if "선거라벨" in region_heat_source.columns else "선거KEY",
                        y_col="지역",
                        value_col="득표율_계산",
                        title=f"{primary_entity}의 지역별 득표율 heatmap",
                        percent=True,
                    ),
                    use_container_width=True,
                )

    with tab3:
        st.caption("RowType 추이는 `관내사전투표 / 선거일투표(읍면동-관내사전투표 또는 투표소합계) / 관외사전투표` 기준으로 정규화합니다.")
        st.plotly_chart(
            stacked_bar_chart(
                rowtype_turnout_trend,
                x_col="선거라벨" if "선거라벨" in rowtype_turnout_trend.columns else "선거KEY",
                y_col="투표구성비",
                color_col="RowType",
                title="사전투표/선거일투표 구성비 변화",
                percent=True,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            stacked_bar_chart(
                rowtype_vote_trend,
                x_col="선거라벨" if "선거라벨" in rowtype_vote_trend.columns else "선거KEY",
                y_col="득표구성비",
                color_col="RowType",
                title="정규화된 RowType별 득표 구성비 변화",
                percent=True,
            ),
            use_container_width=True,
        )

    with tab4:
        st.plotly_chart(
            line_metric_chart(
                gap_trend,
                x_col="선거라벨" if "선거라벨" in gap_trend.columns else "선거KEY",
                y_col="득표율격차",
                title="1위-2위 득표율 격차 추이",
                percent=True,
            ),
            use_container_width=True,
        )
        if not current_competition.empty:
            current_competition = current_competition.sort_values(by="경쟁도지수", ascending=False, kind="stable").head(30)
            current_competition["지표"] = "경쟁도"
            st.plotly_chart(
                heatmap_chart(current_competition, x_col="지표", y_col="지역", value_col="경쟁도지수", title=f"{region_level}별 경쟁도 heatmap", percent=True),
                use_container_width=True,
            )
            st.dataframe(current_competition, use_container_width=True, hide_index=True)

    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        st.download_button("투표율 추이 CSV", dataframe_to_csv_bytes(turnout_trend), file_name="trend_turnout.csv", mime="text/csv", use_container_width=True)
    with download_col2:
        st.download_button("대상 추이 CSV", dataframe_to_csv_bytes(entity_trend), file_name="trend_entity.csv", mime="text/csv", use_container_width=True)
    with download_col3:
        st.download_button("경쟁도 CSV", dataframe_to_csv_bytes(gap_trend), file_name="trend_competitiveness.csv", mime="text/csv", use_container_width=True)


main()
