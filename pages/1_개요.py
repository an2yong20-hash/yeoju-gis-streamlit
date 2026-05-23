from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import bar_candidate_votes, bar_party_share, line_metric_chart, party_color_map
from src.filters import apply_common_filters, build_breadcrumb_text, clear_rowtype_filter, render_sidebar_filters
from src.loaders import (
    get_cache_file_signatures,
    load_cached_dims,
    load_fact_turnout_enriched,
    load_fact_votes_enriched,
    load_global_filter_options,
)
from src.metrics import (
    build_competitiveness_sentence,
    build_delta_sentence,
    calc_entity_share,
    calc_top2_gap_trend,
    calc_turnout_summary,
    calc_turnout_summary_by_election,
    calc_turnout_timeseries,
    calc_votes_summary,
    calc_votes_summary_by_election,
    competitiveness_help_text,
    dataframe_to_csv_bytes,
    format_delta_pp,
    format_int,
    format_percent,
    get_latest_and_previous,
)

OVERVIEW_DIM_COLUMNS = {
    "DimElection": ["선거KEY", "선거시점", "선거명", "선거종류"],
    "DimDong": ["읍면동KEY", "시도명", "구시군명", "일반구명", "읍면동명"],
    "DimPollingPlace": ["투표소KEY", "선거시점", "시도명_F", "구시군명_F", "일반구명_F", "읍면동명_F"],
}

TURNOUT_OVERVIEW_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "RowType",
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
)

VOTES_OVERVIEW_COLUMNS = (
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
    "일반구명",
    "읍면동명",
    "구분",
    "RowType",
    "유효투표수",
    "득표수",
)


def _load_bundle() -> dict[str, object]:
    cache_signature = get_cache_file_signatures()
    return {
        "dims": load_cached_dims(cache_signature, dim_names=tuple(OVERVIEW_DIM_COLUMNS.keys()), columns_map=OVERVIEW_DIM_COLUMNS),
        "turnout": load_fact_turnout_enriched(cache_signature, TURNOUT_OVERVIEW_COLUMNS),
        "votes": load_fact_votes_enriched(cache_signature, VOTES_OVERVIEW_COLUMNS),
        "filter_options": load_global_filter_options(cache_signature),
    }


def _build_election_summary_table(turnout_df: pd.DataFrame, votes_df: pd.DataFrame, dim_election_df: pd.DataFrame) -> pd.DataFrame:
    turnout_summary = calc_turnout_summary_by_election(turnout_df)
    vote_summary = calc_votes_summary_by_election(votes_df)

    election_key_frames = [df.loc[:, ["선거KEY"]].drop_duplicates() for df in (turnout_df, votes_df) if "선거KEY" in df.columns]
    if not election_key_frames:
        return pd.DataFrame()

    election_keys = pd.concat(election_key_frames, ignore_index=True).drop_duplicates(subset=["선거KEY"])
    election_meta = election_keys.merge(
        dim_election_df.loc[:, ["선거KEY", "선거시점", "선거명", "선거종류"]].drop_duplicates(subset=["선거KEY"]),
        on="선거KEY",
        how="left",
        copy=False,
    )
    summary = election_meta.merge(
        turnout_summary.loc[:, ["선거KEY", "선거인수", "투표수", "유효투표수", "무효투표수", "기권수", "투표율", "유효투표율", "무효투표율"]],
        on="선거KEY",
        how="left",
        copy=False,
    )
    summary = summary.merge(
        vote_summary.loc[:, ["선거KEY", "득표수", "후보수", "정당수"]],
        on="선거KEY",
        how="left",
        copy=False,
    )
    return summary.sort_values(by=["선거시점", "선거KEY"], ascending=[False, False], kind="stable").reset_index(drop=True)


def main() -> None:
    st.title("개요")

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

    dims = app_data["dims"]
    turnout = app_data["turnout"]
    votes = app_data["votes"]
    filter_options = app_data["filter_options"]
    selected = render_sidebar_filters(filter_options, include_party=True, include_candidate=True)
    analysis_selected = clear_rowtype_filter(selected)

    filtered_turnout = apply_common_filters(turnout, analysis_selected)
    filtered_votes = apply_common_filters(votes, analysis_selected)
    filtered_dong = apply_common_filters(dims["DimDong"], selected)
    filtered_polling = apply_common_filters(dims["DimPollingPlace"], selected)

    turnout_summary = calc_turnout_summary(filtered_turnout)
    votes_summary = calc_votes_summary(filtered_votes)
    turnout_trend = calc_turnout_timeseries(filtered_turnout)
    latest_turnout = get_latest_and_previous(turnout_trend, "투표율")
    top2_trend = calc_top2_gap_trend(filtered_votes)
    election_summary = _build_election_summary_table(filtered_turnout, filtered_votes, dims["DimElection"])
    current_party_share = calc_entity_share(filtered_votes, entity_type="정당")
    current_party_share = current_party_share.sort_values(by=["선거시점", "득표수"], ascending=[False, False], kind="stable").head(12)

    st.caption("현재 필터: " + build_breadcrumb_text(analysis_selected, filter_options))
    st.info(build_delta_sentence("선택 범위", "투표율", latest_turnout["delta"]))
    if not top2_trend.empty:
        st.caption(build_competitiveness_sentence(top2_trend.sort_values(by=["선거시점", "선거KEY"], ascending=[False, False]).head(1), "선택 범위"))
        st.caption(competitiveness_help_text())

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("총 선거 수", format_int(filtered_votes["선거KEY"].nunique()))
    col2.metric("총 읍면동 수", format_int(filtered_dong["읍면동KEY"].nunique()))
    col3.metric("총 투표소 수", format_int(filtered_polling["투표소KEY"].nunique()))
    col4.metric("총 투표수", format_int(turnout_summary["투표수"]))
    col5.metric("총 유효투표수", format_int(votes_summary["유효투표수"]))
    col6.metric("최근 투표율", format_percent(latest_turnout["current"]), delta=format_delta_pp(latest_turnout["delta"]))

    chart_col1, chart_col2 = st.columns([1.15, 0.85])
    with chart_col1:
        st.plotly_chart(
            line_metric_chart(
                turnout_trend,
                x_col="선거라벨" if "선거라벨" in turnout_trend.columns else "선거KEY",
                y_col="투표율",
                title="선거시점별 투표율 추이",
                percent=True,
            ),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(
            bar_party_share(
                current_party_share,
                title="현재 필터 기준 주요 정당 득표율",
            ),
            use_container_width=True,
        )

    st.subheader("선거별 요약")
    display_summary = election_summary.copy()
    for column in ["선거인수", "투표수", "유효투표수", "무효투표수", "기권수", "득표수", "후보수", "정당수"]:
        if column in display_summary.columns:
            display_summary[column] = display_summary[column].map(format_int)
    for column in ["투표율", "유효투표율", "무효투표율"]:
        if column in display_summary.columns:
            display_summary[column] = display_summary[column].map(format_percent)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "선거별 요약 CSV 다운로드",
        dataframe_to_csv_bytes(election_summary),
        file_name="overview_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        current_candidate_ranking = filtered_votes.groupby(["후보명", "정당명"], as_index=False, observed=True)["득표수"].sum(min_count=1)
        current_candidate_ranking = current_candidate_ranking.sort_values(by="득표수", ascending=False, kind="stable").head(12)
        st.plotly_chart(
            bar_candidate_votes(
                current_candidate_ranking,
                x_col="후보명",
                y_col="득표수",
                color_col="정당명",
                title="현재 필터 기준 후보 득표수",
            ),
            use_container_width=True,
        )
    with detail_col2:
        raw_preview = filtered_votes.head(200).copy()
        st.subheader("원자료 미리보기")
        st.dataframe(raw_preview, use_container_width=True, hide_index=True)
        st.download_button(
            "현재 필터 원자료 CSV 다운로드",
            dataframe_to_csv_bytes(filtered_votes),
            file_name="overview_filtered_votes.csv",
            mime="text/csv",
            use_container_width=True,
        )


main()
