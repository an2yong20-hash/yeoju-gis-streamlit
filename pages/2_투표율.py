from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.charts import grouped_bar_chart, heatmap_chart, line_metric_chart, lollipop_chart, stacked_bar_chart
from src.filters import (
    apply_common_filters,
    build_breadcrumb_text,
    build_region_comparison_selection,
    get_selected_adjacent_gusigun_names,
    render_sidebar_filters,
    resolve_region_level,
)
from src.loaders import get_cache_file_signatures, load_fact_turnout_enriched, load_global_filter_options
from src.maps import build_region_choropleth, get_adjacent_gusigun_keys, load_geometry_context
from src.metrics import (
    build_delta_sentence,
    calc_rowtype_turnout_trend,
    calc_turnout_summary_by_election,
    calc_turnout_trend_by_level,
    dataframe_to_csv_bytes,
    format_delta_pp,
    format_int,
    format_percent,
    get_latest_and_previous,
    safe_divide,
)

TURNOUT_PAGE_COLUMNS = (
    "선거KEY",
    "선거시점",
    "선거명",
    "선거종류",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "구시군KEY",
    "읍면동KEY",
    "투표소KEY",
    "RowType",
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
)
TURNOUT_NUMERIC_COLUMNS = ("선거인수", "투표수", "유효투표수", "무효투표수", "기권수")
PERIOD_EVENT_NAME_MAP = {
    "P": "대통령선거",
    "N": "국회의원선거",
    "L": "지방선거",
}
REGION_KEY_COLUMNS = ["시도명", "구시군KEY", "구시군명", "읍면동KEY", "일반구명", "읍면동명", "투표소KEY", "지역"]


def _load_bundle() -> dict[str, object]:
    cache_signature = get_cache_file_signatures()
    return {
        "fact_turnout": load_fact_turnout_enriched(cache_signature, TURNOUT_PAGE_COLUMNS),
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


def _add_turnout_rates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["투표율"] = safe_divide(result["투표수"], result["선거인수"])
    result["유효투표율"] = safe_divide(result["유효투표수"], result["투표수"])
    result["무효투표율"] = safe_divide(result["무효투표수"], result["투표수"])
    result["기권율"] = safe_divide(result["기권수"], result["선거인수"])
    return result


def _aggregate_turnout_period_frame(df: pd.DataFrame, group_cols: list[str], numeric_cols: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df

    passthrough_cols = [column for column in REGION_KEY_COLUMNS if column in df.columns and column not in group_cols]
    if "선거KEY" not in group_cols and "선거KEY" in df.columns:
        passthrough_cols.append("선거KEY")
    rows: list[dict[str, object]] = []

    for key_values, group in df.groupby(group_cols, observed=True, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {column: value for column, value in zip(group_cols, key_values)}
        representative = group.sort_values(by=["선거KEY"], kind="stable").iloc[0] if "선거KEY" in group.columns else group.iloc[0]
        for column in passthrough_cols:
            row[column] = representative[column]
        for column in numeric_cols:
            if column in group.columns:
                row[column] = group[column].max()
        event_name = _event_name_from_keys(group["선거KEY"]) if "선거KEY" in group.columns else "선거"
        row["선거이벤트"] = event_name
        row["선거축라벨"] = _build_period_label(row.get("선거시점"), event_name)
        row["선거라벨"] = row["선거축라벨"]
        rows.append(row)

    result = pd.DataFrame(rows)
    if all(column in result.columns for column in TURNOUT_NUMERIC_COLUMNS):
        result = _add_turnout_rates(result)
    sort_columns = [column for column in [*group_cols, "선거KEY"] if column in result.columns]
    if sort_columns:
        result = result.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)
    return result


def _build_turnout_period_trend_frame(turnout_df: pd.DataFrame) -> pd.DataFrame:
    by_election = calc_turnout_summary_by_election(turnout_df)
    if by_election.empty:
        return by_election
    result = _aggregate_turnout_period_frame(by_election, ["선거시점"], TURNOUT_NUMERIC_COLUMNS)
    return result.sort_values(by=["선거시점"], ascending=True, kind="stable").reset_index(drop=True)


def _build_turnout_region_period_trend_frame(turnout_df: pd.DataFrame, level: str) -> pd.DataFrame:
    by_election = calc_turnout_trend_by_level(turnout_df, level=level)
    if by_election.empty:
        return by_election
    region_cols = [column for column in REGION_KEY_COLUMNS if column in by_election.columns]
    result = _aggregate_turnout_period_frame(by_election, ["선거시점", *region_cols], TURNOUT_NUMERIC_COLUMNS)
    return result.sort_values(by=["선거시점", "지역"], ascending=[True, True], kind="stable").reset_index(drop=True)


def _build_turnout_rowtype_period_trend_frame(turnout_df: pd.DataFrame) -> pd.DataFrame:
    by_election = calc_rowtype_turnout_trend(turnout_df)
    if by_election.empty:
        return by_election
    numeric_cols = tuple(column for column in ["투표수"] if column in by_election.columns)
    result = _aggregate_turnout_period_frame(by_election, ["선거시점", "RowType"], numeric_cols)
    if "투표수" in result.columns:
        period_totals = result.groupby("선거시점", observed=True)["투표수"].transform("sum")
        result["투표구성비"] = safe_divide(result["투표수"], period_totals)
    return result.sort_values(by=["선거시점", "RowType"], ascending=[True, True], kind="stable").reset_index(drop=True)


def _build_region_current_frame(region_trend_df: pd.DataFrame, latest_period: object, sort_metric: str) -> pd.DataFrame:
    if region_trend_df.empty:
        return region_trend_df
    result = region_trend_df.loc[region_trend_df["선거시점"] == latest_period].copy()
    return result.sort_values(by=[sort_metric, "지역"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _resolve_latest_period(frame: pd.DataFrame, fallback: object) -> object:
    if frame.empty or "선거시점" not in frame.columns:
        return fallback
    period_series = frame["선거시점"].dropna()
    if period_series.empty:
        return fallback
    return period_series.max()


def _infer_sido_scope_for_gis(filtered_turnout: pd.DataFrame, selected: dict[str, list[str]]) -> list[str]:
    if selected.get("시도명") or filtered_turnout.empty or "시도명" not in filtered_turnout.columns:
        return []
    return filtered_turnout["시도명"].dropna().astype("string").unique().tolist()


def _resolve_focus_gusigun_key(df: pd.DataFrame) -> str | None:
    if df.empty or "구시군KEY" not in df.columns:
        return None
    keys = [
        str(value)
        for value in df["구시군KEY"].dropna().astype("string").unique().tolist()
        if "합계" not in str(value)
    ]
    return keys[0] if len(keys) == 1 else None


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


def main() -> None:
    st.title("투표율")

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
    filtered_turnout = apply_common_filters(app_data["fact_turnout"], selected)

    if filtered_turnout.empty:
        st.warning("현재 필터 조건에 맞는 투표율 데이터가 없습니다.")
        st.stop()

    st.caption("현재 필터: " + build_breadcrumb_text(selected, filter_options))
    st.caption("투표율 추이는 같은 선거시점의 하위 선거를 하나로 묶은 `대통령선거 / 국회의원선거 / 지방선거` 기준으로 표시합니다.")

    trend_df = _build_turnout_period_trend_frame(filtered_turnout)
    latest = get_latest_and_previous(trend_df, "투표율")
    default_level = resolve_region_level(selected, default="구시군")

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        region_level = st.selectbox("비교 레벨", ["시도", "구시군", "읍면동"], index=["시도", "구시군", "읍면동"].index(default_level), key="turnout_level")
    with control_col2:
        focus_metric = st.selectbox("핵심 지표", ["투표율", "유효투표율", "무효투표율", "기권율"], key="turnout_focus_metric")
    st.caption("RowType 구성비는 `관내사전투표 / 선거일투표(읍면동-관내사전투표) / 관외사전투표(재외·거소·부재자 포함)` 기준으로 계산합니다.")

    latest_row = trend_df.sort_values(by=["선거시점"], ascending=[False], kind="stable").iloc[0]
    current_summary = latest_row
    comparison_selected = build_region_comparison_selection(selected, region_level)
    comparison_turnout = apply_common_filters(app_data["fact_turnout"], comparison_selected)
    inferred_sido_scope = _infer_sido_scope_for_gis(filtered_turnout, selected)
    if region_level == "구시군" and inferred_sido_scope:
        comparison_turnout = comparison_turnout.loc[
            comparison_turnout["시도명"].astype("string").isin(inferred_sido_scope)
        ].copy()
    focus_emd_gusigun_key = _resolve_focus_gusigun_key(filtered_turnout if region_level == "읍면동" else comparison_turnout.iloc[0:0].copy())
    if region_level == "읍면동" and focus_emd_gusigun_key and "구시군KEY" in comparison_turnout.columns:
        comparison_turnout = comparison_turnout.loc[
            comparison_turnout["구시군KEY"].astype("string") == str(focus_emd_gusigun_key)
        ].copy()
    focus_gusigun_key = _resolve_focus_gusigun_key(filtered_turnout if region_level == "구시군" else comparison_turnout.iloc[0:0].copy())
    comparison_gusigun_key = _resolve_focus_gusigun_key(comparison_turnout if region_level == "읍면동" else comparison_turnout.iloc[0:0].copy())
    region_trend_df = _build_turnout_region_period_trend_frame(comparison_turnout, region_level)
    region_latest_period = _resolve_latest_period(region_trend_df, latest_row["선거시점"])
    region_df = _build_region_current_frame(region_trend_df, region_latest_period, focus_metric)
    rowtype_trend_df = _build_turnout_rowtype_period_trend_frame(filtered_turnout)
    rowtype_df = rowtype_trend_df.loc[rowtype_trend_df["선거시점"] == latest_row["선거시점"]].copy()

    st.info(build_delta_sentence("선택 범위", focus_metric, latest["delta"] if focus_metric == "투표율" else get_latest_and_previous(trend_df, focus_metric)["delta"]))

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("최신 선거시점", str(latest_row["선거축라벨"]))
    col2.metric("투표수", format_int(current_summary["투표수"]))
    col3.metric("투표율", format_percent(current_summary["투표율"]), delta=format_delta_pp(latest["delta"]))
    col4.metric("유효투표율", format_percent(current_summary["유효투표율"]))
    col5.metric("무효투표율", format_percent(current_summary["무효투표율"]))
    col6.metric("기권율", format_percent(current_summary["기권율"]))

    tab1, tab2, tab3 = st.tabs(["추이분석", "지역 비교", "RowType 분석"])

    with tab1:
        st.caption("투표율 추이분석은 이 페이지 내부에서 관리하며, 같은 선거시점의 하위 선거는 하나로 묶습니다.")
        trend_chart = line_metric_chart(
            trend_df,
            x_col="선거축라벨" if "선거축라벨" in trend_df.columns else "선거KEY",
            y_col=focus_metric,
            title=f"선거시점별 {focus_metric} 추이",
            percent=True,
            show_labels=True,
        )
        st.plotly_chart(trend_chart, use_container_width=True)
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

    with tab2:
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
                value_col=focus_metric,
                title=f"{region_level}별 {focus_metric} 랭킹",
                percent=True,
            ),
            use_container_width=True,
            key="turnout_region_rank_chart",
            on_select="rerun",
        )
        selected_region = _extract_selection_value(selection, ("y",))

        if region_level == "구시군" and "구시군KEY" in region_df.columns:
            geometry_context = load_geometry_context("구시군")
            st.plotly_chart(
                build_region_choropleth(
                    region_df,
                    level="구시군",
                    key_col="구시군KEY",
                    value_col=focus_metric,
                    title=f"구시군별 {focus_metric} GIS 히트맵",
                    geometry_context=geometry_context,
                    percent=True,
                    label_col="구시군명" if "구시군명" in region_df.columns else "지역",
                ),
                use_container_width=True,
            )
        elif region_level == "읍면동" and "읍면동KEY" in region_df.columns:
            geometry_context = load_geometry_context("읍면동")
            st.plotly_chart(
                build_region_choropleth(
                    region_df,
                    level="읍면동",
                    key_col="읍면동KEY",
                    value_col=focus_metric,
                    title=f"읍면동별 {focus_metric} GIS 히트맵",
                    geometry_context=geometry_context,
                    percent=True,
                    label_col="읍면동명" if "읍면동명" in region_df.columns else "지역",
                ),
                use_container_width=True,
            )
        else:
            heatmap_source = region_df.sort_values(by=focus_metric, ascending=False, kind="stable").head(25).copy()
            heatmap_source["지표"] = focus_metric
            st.plotly_chart(
                heatmap_chart(
                    heatmap_source,
                    x_col="지표",
                    y_col="지역",
                    value_col=focus_metric,
                    title=f"{region_level}별 {focus_metric} heatmap",
                    percent=True,
                ),
                use_container_width=True,
            )

        end_label_regions, end_label_text_map = _resolve_chart_end_labels(selected, region_df, preferred_level=region_level)
        compare_fig = line_metric_chart(
            region_trend_df,
            end_label_entities=end_label_regions or None,
            end_label_text_map=end_label_text_map or None,
            x_col="선거축라벨" if "선거축라벨" in region_trend_df.columns else "선거KEY",
            y_col=focus_metric,
            color_col="지역",
            title=f"{region_level} 다중 비교: {focus_metric}",
            percent=True,
            legend_order=region_df["지역"].dropna().astype("string").tolist(),
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
                    x_col="선거축라벨" if "선거축라벨" in region_detail.columns else "선거KEY",
                    y_col=focus_metric,
                    title=f"{selected_region}의 선거시점별 {focus_metric}",
                    percent=True,
                ),
                use_container_width=True,
            )
            st.dataframe(region_detail, use_container_width=True, hide_index=True)

    with tab3:
        st.caption("합계와 오류투표는 구성비에서 제외합니다.")
        st.plotly_chart(
            grouped_bar_chart(
                rowtype_df,
                x_col="RowType",
                y_col="투표구성비",
                color_col="RowType",
                title="현재 선거 정규화된 투표방식 구성비",
                percent=True,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            stacked_bar_chart(
                rowtype_trend_df,
                x_col="선거축라벨" if "선거축라벨" in rowtype_trend_df.columns else "선거KEY",
                y_col="투표구성비",
                color_col="RowType",
                title="정규화된 투표방식 구성비 추이",
                percent=True,
            ),
            use_container_width=True,
        )
        st.dataframe(rowtype_trend_df, use_container_width=True, hide_index=True)

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "투표율 추이분석 CSV 다운로드",
            dataframe_to_csv_bytes(trend_df),
            file_name="turnout_trend.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "현재 선거 지역 비교 CSV 다운로드",
            dataframe_to_csv_bytes(region_df),
            file_name="turnout_region_compare.csv",
            mime="text/csv",
            use_container_width=True,
        )


main()
