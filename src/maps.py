from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import BASE_DIR, CACHE_DIR
from src.charts import entity_color_map

DEFAULT_CENTER = {"lat": 36.35, "lon": 127.8, "zoom": 6.3}
BASEMAP_STYLE_OPTIONS = {
    "배경 없음": "white-bg",
    "한글 친화": "open-street-map",
    "기본": "carto-positron",
    "다크": "carto-darkmatter",
}
GEOMETRY_PATHS = [
    BASE_DIR / "assets" / "geo",
    BASE_DIR / "assets" / "geojson",
    BASE_DIR / "geometry",
    BASE_DIR / "geojson",
    BASE_DIR / "DB" / "행정동경계",
    BASE_DIR / "DB" / "행정동경계" / "BND_ADM_DONG_PG",
]
GEOMETRY_CATALOG = {
    "읍면동": {
        "filenames": ["emd.geojson", "dong.geojson", "읍면동.geojson", "BND_ADM_DONG_PG.shp"],
        "featureid_candidates": ["properties.읍면동KEY", "properties.EMD_KEY", "properties.emd_key"],
    },
    "구시군": {
        "filenames": ["sig.geojson", "gusigun.geojson", "구시군.geojson"],
        "featureid_candidates": ["properties.구시군KEY", "properties.SIG_KEY", "properties.sig_key"],
    },
}
DONG_CODEBOOK_FILENAMES = ["센서스 공간정보 지역 코드.xlsx"]
DONG_DIM_COLUMNS = ["읍면동KEY", "구시군KEY", "시도명", "구시군명", "일반구명", "읍면동명"]
GEOMETRY_CACHE_DIR = CACHE_DIR / "geometry"
SIMPLIFY_TOLERANCE = {
    "읍면동": 0.00025,
    "구시군": 0.0007,
}
GIS_THREE_TONE_COLOR_SCALE = [
    [0.0, "#D94C3D"],
    [0.5, "#F8FAFC"],
    [1.0, "#1F5AA6"],
]
GIS_SIGNED_GAP_COLOR_SCALE = [
    [0.0, "#1F5AA6"],
    [0.5, "#F8FAFC"],
    [1.0, "#D94C3D"],
]


def _parse_hex_color(color: str) -> tuple[int, int, int] | None:
    value = str(color).strip()
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return None
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def _blend_rgb(start: tuple[int, int, int], end: tuple[int, int, int], amount: float) -> str:
    amount = max(0.0, min(1.0, float(amount)))
    rgb = tuple(int(round(start[idx] + (end[idx] - start[idx]) * amount)) for idx in range(3))
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def build_single_hue_colorscale(color: str | None) -> list[list[object]] | None:
    parsed = _parse_hex_color(str(color)) if color else None
    if parsed is None:
        return None
    white = (248, 250, 252)
    return [
        [0.0, _blend_rgb(white, parsed, 0.18)],
        [0.45, _blend_rgb(white, parsed, 0.48)],
        [1.0, f"#{parsed[0]:02X}{parsed[1]:02X}{parsed[2]:02X}"],
    ]


def build_diverging_colorscale(negative_color: str | None, positive_color: str | None) -> list[list[object]] | None:
    negative = _parse_hex_color(str(negative_color)) if negative_color else None
    positive = _parse_hex_color(str(positive_color)) if positive_color else None
    if negative is None or positive is None:
        return None
    white = (248, 250, 252)
    return [
        [0.0, _blend_rgb(white, negative, 0.92)],
        [0.5, "#F8FAFC"],
        [1.0, _blend_rgb(white, positive, 0.92)],
    ]


def resolve_basemap_style(option: str | None) -> str:
    if option in BASEMAP_STYLE_OPTIONS.values():
        return str(option)
    return BASEMAP_STYLE_OPTIONS.get(str(option), BASEMAP_STYLE_OPTIONS["배경 없음"])


def empty_map_figure(message: str = "표시할 지도 데이터가 없습니다.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def discover_geometry_path(level: str) -> Path | None:
    spec = GEOMETRY_CATALOG.get(level)
    if spec is None:
        return None

    for directory in GEOMETRY_PATHS:
        for filename in spec["filenames"]:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


def discover_dong_codebook_path() -> Path | None:
    for directory in GEOMETRY_PATHS:
        for filename in DONG_CODEBOOK_FILENAMES:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


def _geometry_cache_path(level: str) -> Path:
    suffix = "emd" if level == "읍면동" else "sig"
    return GEOMETRY_CACHE_DIR / f"{suffix}_boundaries.geojson"


def _geometry_meta_path(level: str) -> Path:
    suffix = "emd" if level == "읍면동" else "sig"
    return GEOMETRY_CACHE_DIR / f"{suffix}_boundaries_meta.json"


def _geometry_adjacency_path() -> Path:
    return GEOMETRY_CACHE_DIR / "sig_adjacency.json"


def _ensure_geometry_cache_dir() -> None:
    GEOMETRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_cache_fresh(target: Path, sources: list[Path]) -> bool:
    if not target.exists():
        return False
    target_mtime = target.stat().st_mtime
    return all(source.exists() and source.stat().st_mtime <= target_mtime for source in sources)


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, mtime: float) -> dict[str, Any]:
    del mtime
    with Path(path_str).open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_json_file(path: Path) -> dict[str, Any]:
    return _read_json_cached(str(path), path.stat().st_mtime)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def _feature_property_keys(geojson: dict[str, Any]) -> set[str]:
    features = geojson.get("features", [])
    if not features:
        return set()
    return set(features[0].get("properties", {}).keys())


def resolve_featureidkey(level: str, geojson: dict[str, Any]) -> str | None:
    property_keys = _feature_property_keys(geojson)
    for candidate in GEOMETRY_CATALOG.get(level, {}).get("featureid_candidates", []):
        property_name = candidate.split(".", maxsplit=1)[1] if "." in candidate else candidate
        if property_name in property_keys:
            return candidate
    return None


def _normalize_region_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value).strip().replace(" ", "")


def _normalize_dong_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value).strip().replace(" ", "").replace("·", ".").replace("ㆍ", ".").replace(",", ".")


def _build_dong_candidates(value: Any) -> list[str]:
    normalized = _normalize_dong_name(value)
    if not normalized:
        return []

    candidates = [normalized]
    ga_variant = re.sub(
        r"(\d)가(\d(?:[.,]\d)*)동$",
        lambda match: f"{match.group(1)}가제{match.group(2)}동",
        normalized,
    )
    if ga_variant not in candidates:
        candidates.append(ga_variant)

    seq_variant = re.sub(
        r"(\d(?:[.,]\d)*)동$",
        lambda match: f"제{match.group(1)}동",
        normalized,
    )
    if seq_variant not in candidates:
        candidates.append(seq_variant)
    return candidates


@st.cache_data(show_spinner=False)
def _load_dong_census_mapping_cached(codebook_path_str: str) -> pd.DataFrame:
    codebook_path = Path(codebook_path_str)
    workbook = pd.ExcelFile(codebook_path)
    sheet_name = workbook.sheet_names[0]

    census = pd.read_excel(codebook_path, sheet_name=sheet_name, header=1, dtype="string")
    if census.shape[1] < 6:
        raise ValueError(f"{codebook_path.name} 형식이 예상과 다릅니다.")

    census = census.iloc[:, :6].copy()
    census.columns = ["시도코드", "시도명", "시군구코드", "시군구명", "읍면동코드", "읍면동명"]
    for column in ["시도코드", "시군구코드", "읍면동코드"]:
        census[column] = census[column].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()

    census = census.loc[
        census["시도코드"].notna()
        & census["시군구코드"].notna()
        & census["읍면동코드"].notna()
        & census["시도명"].notna()
        & census["시군구명"].notna()
        & census["읍면동명"].notna()
    ].copy()
    census["ADM_CD"] = (
        census["시도코드"].str.zfill(2)
        + census["시군구코드"].str.zfill(3)
        + census["읍면동코드"].str.zfill(3)
    )
    census["시군구정규"] = census["시군구명"].map(_normalize_region_name)
    census["읍면동정규"] = census["읍면동명"].map(_normalize_dong_name)

    dim = pd.read_parquet(CACHE_DIR / "dim_dong.parquet", columns=DONG_DIM_COLUMNS).copy()
    dim["시군구정규"] = (dim["구시군명"].fillna("") + dim["일반구명"].fillna("")).map(_normalize_region_name)
    dim["읍면동정규"] = dim["읍면동명"].map(_normalize_dong_name)
    dim_lookup = dim[
        ["읍면동KEY", "구시군KEY", "시도명", "시군구정규", "읍면동정규"]
    ].drop_duplicates().copy()

    exact = census.merge(
        dim_lookup,
        on=["시도명", "시군구정규", "읍면동정규"],
        how="left",
    )
    exact["매핑상태"] = pd.Series(pd.NA, index=exact.index, dtype="string")
    exact.loc[exact["읍면동KEY"].notna(), "매핑상태"] = "exact"

    pending = exact.loc[exact["읍면동KEY"].isna(), ["ADM_CD", "시도명", "시군구정규", "읍면동명"]].copy()
    candidate_rows: list[dict[str, str | None]] = []
    for row in pending.itertuples(index=False):
        for candidate in _build_dong_candidates(row.읍면동명):
            candidate_rows.append(
                {
                    "ADM_CD": row.ADM_CD,
                    "시도명": row.시도명,
                    "시군구정규": row.시군구정규,
                    "후보명": candidate,
                }
            )

    fallback_lookup = pd.DataFrame(candidate_rows)
    if not fallback_lookup.empty:
        fallback_lookup = fallback_lookup.merge(
            dim_lookup.rename(columns={"읍면동정규": "후보명"}),
            on=["시도명", "시군구정규", "후보명"],
            how="left",
        )
        fallback_lookup = fallback_lookup.loc[
            fallback_lookup["읍면동KEY"].notna(),
            ["ADM_CD", "읍면동KEY", "구시군KEY"],
        ].drop_duplicates()
        match_counts = (
            fallback_lookup.groupby("ADM_CD", dropna=False, observed=True)["읍면동KEY"]
            .nunique()
            .reset_index(name="match_count")
        )
        fallback_lookup = fallback_lookup.merge(match_counts, on="ADM_CD", how="left")
        fallback_lookup = fallback_lookup.loc[
            fallback_lookup["match_count"] == 1,
            ["ADM_CD", "읍면동KEY", "구시군KEY"],
        ].drop_duplicates()
    else:
        fallback_lookup = pd.DataFrame(columns=["ADM_CD", "읍면동KEY", "구시군KEY"])

    mapping = exact[
        ["ADM_CD", "시도명", "시군구명", "읍면동명", "읍면동KEY", "구시군KEY", "매핑상태"]
    ].copy()
    mapping = mapping.merge(
        fallback_lookup.rename(
            columns={
                "읍면동KEY": "fallback_읍면동KEY",
                "구시군KEY": "fallback_구시군KEY",
            }
        ),
        on="ADM_CD",
        how="left",
    )
    mapping["읍면동KEY"] = mapping["읍면동KEY"].fillna(mapping["fallback_읍면동KEY"])
    mapping["구시군KEY"] = mapping["구시군KEY"].fillna(mapping["fallback_구시군KEY"])
    mapping.loc[mapping["매핑상태"].isna() & mapping["읍면동KEY"].notna(), "매핑상태"] = "fallback"
    mapping.loc[mapping["매핑상태"].isna(), "매핑상태"] = "unmatched"
    mapping["센서스시트"] = sheet_name
    mapping["매핑우선순위"] = mapping["매핑상태"].map({"exact": 0, "fallback": 1, "unmatched": 2}).fillna(9)
    mapping["구두점우선순위"] = (
        mapping["읍면동KEY"].fillna("").astype("string").str.contains(",", regex=False).astype("int8")
    )
    mapping = (
        mapping.sort_values(
            by=["ADM_CD", "매핑우선순위", "구두점우선순위", "읍면동KEY"],
            kind="stable",
        )
        .drop_duplicates(subset=["ADM_CD"], keep="first")
        .reset_index(drop=True)
    )
    return mapping[
        ["ADM_CD", "시도명", "시군구명", "읍면동명", "읍면동KEY", "구시군KEY", "매핑상태", "센서스시트"]
    ]


def _build_boundary_caches(dong_path: Path, codebook_path: Path) -> dict[str, Any]:
    _ensure_geometry_cache_dir()
    sources = [dong_path, codebook_path, CACHE_DIR / "dim_dong.parquet"]
    emd_cache_path = _geometry_cache_path("읍면동")
    sig_cache_path = _geometry_cache_path("구시군")
    emd_meta_path = _geometry_meta_path("읍면동")
    sig_meta_path = _geometry_meta_path("구시군")

    if all(
        _is_cache_fresh(target, sources)
        for target in [emd_cache_path, sig_cache_path, emd_meta_path, sig_meta_path]
    ):
        return {
            "emd_geojson": _load_json_file(emd_cache_path),
            "sig_geojson": _load_json_file(sig_cache_path),
            "emd_meta": _load_json_file(emd_meta_path),
            "sig_meta": _load_json_file(sig_meta_path),
        }

    gdf = gpd.read_file(dong_path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)

    gdf["ADM_CD"] = gdf["ADM_CD"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    mapping = _load_dong_census_mapping_cached(str(codebook_path))
    merged = gdf.merge(
        mapping[["ADM_CD", "시도명", "시군구명", "읍면동명", "읍면동KEY", "구시군KEY", "매핑상태"]],
        on="ADM_CD",
        how="left",
    )

    total_features = int(len(merged))
    matched_emd = int(merged["읍면동KEY"].notna().sum())
    unmatched_emd = max(total_features - matched_emd, 0)

    emd_gdf = merged.loc[merged["읍면동KEY"].notna()].copy()
    emd_gdf = emd_gdf[
        ["geometry", "ADM_CD", "ADM_NM", "시도명", "시군구명", "읍면동명", "읍면동KEY", "구시군KEY", "매핑상태"]
    ]
    emd_gdf["geometry"] = emd_gdf.geometry.simplify(SIMPLIFY_TOLERANCE["읍면동"], preserve_topology=True)

    sig_source = emd_gdf.loc[emd_gdf["구시군KEY"].notna()].copy()
    sig_gdf = (
        sig_source.dissolve(
            by="구시군KEY",
            as_index=False,
            aggfunc={
                "ADM_CD": "first",
                "ADM_NM": "first",
                "시도명": "first",
                "시군구명": "first",
                "매핑상태": "first",
            },
        )
        if not sig_source.empty
        else gpd.GeoDataFrame(columns=["구시군KEY", "geometry"], geometry="geometry", crs="EPSG:4326")
    )
    if not sig_gdf.empty:
        sig_gdf["geometry"] = sig_gdf.geometry.simplify(SIMPLIFY_TOLERANCE["구시군"], preserve_topology=True)

    emd_geojson = json.loads(emd_gdf.to_json(drop_id=True))
    sig_geojson = json.loads(sig_gdf.to_json(drop_id=True))
    emd_meta = {
        "available": True,
        "path": str(dong_path),
        "featureidkey": "properties.읍면동KEY",
        "matched_features": matched_emd,
        "unmatched_features": unmatched_emd,
        "total_features": total_features,
        "message": (
            f"{dong_path.name} + {codebook_path.name} 매핑 사용 "
            f"({matched_emd:,}/{total_features:,}개 읍면동KEY 연결"
            + (f", 미연결 {unmatched_emd:,}개" if unmatched_emd else "")
            + ")"
        ),
    }
    sig_meta = {
        "available": True,
        "path": str(dong_path),
        "featureidkey": "properties.구시군KEY",
        "matched_features": int(sig_gdf["구시군KEY"].nunique()) if not sig_gdf.empty else 0,
        "unmatched_features": 0,
        "total_features": int(sig_gdf["구시군KEY"].nunique()) if not sig_gdf.empty else 0,
        "message": f"{dong_path.name}를 읍면동 경계에서 집계해 구시군 경계로 사용합니다.",
    }

    _write_json_file(emd_cache_path, emd_geojson)
    _write_json_file(sig_cache_path, sig_geojson)
    _write_json_file(emd_meta_path, emd_meta)
    _write_json_file(sig_meta_path, sig_meta)

    return {
        "emd_geojson": emd_geojson,
        "sig_geojson": sig_geojson,
        "emd_meta": emd_meta,
        "sig_meta": sig_meta,
    }


def _load_cached_boundary_context(level: str) -> dict[str, Any]:
    dong_path = discover_geometry_path("읍면동")
    codebook_path = discover_dong_codebook_path()
    if dong_path is None or codebook_path is None:
        return {
            "available": False,
            "path": None,
            "geojson": None,
            "featureidkey": None,
            "message": "현재 경계 파일이 없어 포인트 지도만 표시합니다.",
        }

    bundle = _build_boundary_caches(dong_path, codebook_path)
    if level == "읍면동":
        meta = bundle["emd_meta"]
        return {
            "available": True,
            "path": Path(meta["path"]),
            "geojson": bundle["emd_geojson"],
            "featureidkey": meta["featureidkey"],
            "message": meta["message"],
            "matched_features": meta["matched_features"],
            "unmatched_features": meta["unmatched_features"],
        }

    meta = bundle["sig_meta"]
    return {
        "available": True,
        "path": Path(meta["path"]),
        "geojson": bundle["sig_geojson"],
        "featureidkey": meta["featureidkey"],
        "message": meta["message"],
        "matched_features": meta["matched_features"],
        "unmatched_features": meta["unmatched_features"],
    }


@st.cache_data(show_spinner=False)
def _build_gusigun_adjacency_cached(
    dong_path_str: str,
    dong_mtime: float,
    codebook_path_str: str,
    codebook_mtime: float,
) -> dict[str, list[str]]:
    del dong_mtime, codebook_mtime
    gdf = gpd.read_file(dong_path_str)
    if gdf.empty or "ADM_CD" not in gdf.columns:
        return {}
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    gdf["ADM_CD"] = gdf["ADM_CD"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    mapping = _load_dong_census_mapping_cached(codebook_path_str)
    merged = gdf.merge(mapping[["ADM_CD", "구시군KEY"]], on="ADM_CD", how="left")
    sig_source = merged.loc[merged["구시군KEY"].notna(), ["구시군KEY", "geometry"]].copy()
    if sig_source.empty:
        return {}

    sig_gdf = sig_source.dissolve(by="구시군KEY", as_index=False)
    sig_gdf["geometry"] = sig_gdf.geometry.buffer(0)
    sig_gdf = sig_gdf.loc[sig_gdf["geometry"].notna() & ~sig_gdf.geometry.is_empty, ["구시군KEY", "geometry"]].reset_index(drop=True)
    if sig_gdf.empty:
        return {}

    sindex = sig_gdf.sindex
    adjacency: dict[str, list[str]] = {}
    for idx, row in sig_gdf.iterrows():
        region_key = str(row["구시군KEY"])
        geometry = row["geometry"]
        if geometry is None or geometry.is_empty:
            adjacency[region_key] = []
            continue

        candidate_index = list(sindex.intersection(geometry.bounds))
        if not candidate_index:
            adjacency[region_key] = []
            continue

        neighbors = sig_gdf.iloc[candidate_index].copy()
        neighbors = neighbors.loc[neighbors["구시군KEY"].astype("string") != region_key]
        if neighbors.empty:
            adjacency[region_key] = []
            continue

        related_mask = neighbors.geometry.touches(geometry)
        if not bool(related_mask.any()):
            related_mask = neighbors.geometry.intersects(geometry)
        adjacency[region_key] = sorted(neighbors.loc[related_mask, "구시군KEY"].astype("string").unique().tolist())

    return adjacency


def get_adjacent_gusigun_keys(target_key: str | None) -> list[str]:
    if not target_key:
        return []

    dong_path = discover_geometry_path("읍면동")
    codebook_path = discover_dong_codebook_path()
    if dong_path is None or codebook_path is None:
        return []

    _ensure_geometry_cache_dir()
    adjacency_path = _geometry_adjacency_path()
    sources = [dong_path, codebook_path, CACHE_DIR / "dim_dong.parquet"]
    if _is_cache_fresh(adjacency_path, sources):
        adjacency = _load_json_file(adjacency_path)
    else:
        adjacency = _build_gusigun_adjacency_cached(
            str(dong_path),
            dong_path.stat().st_mtime,
            str(codebook_path),
            codebook_path.stat().st_mtime,
        )
        _write_json_file(adjacency_path, adjacency)
    return adjacency.get(str(target_key), [])


def load_geometry_context(level: str) -> dict[str, Any]:
    direct_path = discover_geometry_path(level)
    if direct_path is not None and direct_path.suffix.lower() != ".shp":
        geojson = _load_json_file(direct_path)
        featureidkey = resolve_featureidkey(level, geojson)
        if featureidkey is None:
            return {
                "available": False,
                "path": direct_path,
                "geojson": geojson,
                "featureidkey": None,
                "message": f"{direct_path.name}는 읽었지만 KEY 연결 속성을 찾지 못했습니다.",
            }
        return {
            "available": True,
            "path": direct_path,
            "geojson": geojson,
            "featureidkey": featureidkey,
            "message": f"{direct_path.name} 경계 파일을 사용합니다.",
        }

    if level in {"읍면동", "구시군"}:
        return _load_cached_boundary_context(level)

    return {
        "available": False,
        "path": None,
        "geojson": None,
        "featureidkey": None,
        "message": "현재 경계 파일이 없어 포인트 지도만 표시합니다.",
    }


def _compute_map_view(df: pd.DataFrame, lat_col: str = "위도", lon_col: str = "경도") -> dict[str, float]:
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return DEFAULT_CENTER.copy()

    valid = df.loc[df[lat_col].notna() & df[lon_col].notna()]
    if valid.empty:
        return DEFAULT_CENTER.copy()

    return {
        "lat": float(valid[lat_col].astype(float).median()),
        "lon": float(valid[lon_col].astype(float).median()),
        "zoom": 9.0 if len(valid) < 500 else 7.5,
    }


def _filter_geojson_for_locations(geojson: dict[str, Any], featureidkey: str | None, locations: pd.Series) -> dict[str, Any]:
    if not featureidkey:
        return geojson

    property_name = featureidkey.split(".", maxsplit=1)[1] if "." in featureidkey else featureidkey
    allowed = set(locations.dropna().astype("string"))
    if not allowed:
        return geojson

    filtered_features = [
        feature
        for feature in geojson.get("features", [])
        if str(feature.get("properties", {}).get(property_name)) in allowed
    ]
    if not filtered_features:
        return geojson
    return {"type": geojson.get("type", "FeatureCollection"), "features": filtered_features}


def _filter_frame_for_geojson_locations(df: pd.DataFrame, geojson: dict[str, Any], featureidkey: str | None, key_col: str) -> pd.DataFrame:
    if not featureidkey or key_col not in df.columns:
        return df

    property_name = featureidkey.split(".", maxsplit=1)[1] if "." in featureidkey else featureidkey
    visible_locations = {
        str(feature.get("properties", {}).get(property_name))
        for feature in geojson.get("features", [])
        if feature.get("properties", {}).get(property_name) is not None
    }
    if not visible_locations:
        return df

    return df.loc[df[key_col].astype("string").isin(visible_locations)].copy()


def _build_geojson_label_frame(
    geojson: dict[str, Any],
    featureidkey: str | None,
    key_col: str,
    label_df: pd.DataFrame,
    label_col: str,
    value_col: str | None = None,
) -> pd.DataFrame:
    required_columns = [key_col, label_col]
    if value_col and value_col in label_df.columns:
        required_columns.append(value_col)
    if not featureidkey or label_df.empty or key_col not in label_df.columns or label_col not in label_df.columns:
        return pd.DataFrame(columns=[*required_columns, "lat", "lon"])

    property_name = featureidkey.split(".", maxsplit=1)[1] if "." in featureidkey else featureidkey
    features = geojson.get("features", [])
    if not features:
        return pd.DataFrame(columns=[*required_columns, "lat", "lon"])

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if gdf.empty or property_name not in gdf.columns:
        return pd.DataFrame(columns=[*required_columns, "lat", "lon"])

    projected = gdf.to_crs(epsg=3857)
    centroids = projected.geometry.centroid.to_crs(epsg=4326)
    label_points = pd.DataFrame(
        {
            key_col: gdf[property_name].astype("string"),
            "lat": centroids.y,
            "lon": centroids.x,
        }
    )
    labels = (
        label_df.loc[:, required_columns]
        .dropna(subset=[key_col, label_col])
        .drop_duplicates(subset=[key_col])
        .assign(**{key_col: lambda frame: frame[key_col].astype("string")})
    )
    return label_points.merge(labels, on=key_col, how="inner", copy=False)


def _format_map_value_label(value: object, value_col: str, percent: bool = False) -> str:
    if pd.isna(value):
        return ""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if "격차" in value_col and "율" in value_col:
        return f"{numeric_value:+.1%}p"
    if percent or any(keyword in value_col for keyword in ["비중", "구성비", "투표율", "득표율", "점유율", "경쟁도지수"]):
        return f"{numeric_value:.1%}"
    if "평균연령" in value_col:
        return f"{numeric_value:.1f}세"
    if "세대수" in value_col:
        return f"{int(round(numeric_value)):,}세대"
    if "가구수" in value_col:
        return f"{int(round(numeric_value)):,}가구"
    if any(keyword in value_col for keyword in ["인구", "선거인수", "득표수", "투표수", "유효투표수", "무효투표수", "기권수"]):
        return f"{int(round(numeric_value)):,}명"
    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"
    return f"{numeric_value:,.1f}"


def _compute_geojson_center(geojson: dict[str, Any]) -> dict[str, float]:
    features = geojson.get("features", [])
    if not features:
        return DEFAULT_CENTER.copy()

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if gdf.empty:
        return DEFAULT_CENTER.copy()

    minx, miny, maxx, maxy = gdf.total_bounds
    lon = float((minx + maxx) / 2)
    lat = float((miny + maxy) / 2)
    span = max(maxx - minx, maxy - miny)
    if span <= 0.05:
        zoom = 10.5
    elif span <= 0.12:
        zoom = 9.6
    elif span <= 0.3:
        zoom = 8.8
    elif span <= 0.8:
        zoom = 7.8
    elif span <= 1.5:
        zoom = 7.0
    else:
        zoom = DEFAULT_CENTER["zoom"]
    return {"lat": lat, "lon": lon, "zoom": zoom}


def build_polling_point_map(
    df: pd.DataFrame,
    title: str,
    size_col: str = "투표수",
    color_col: str = "1위정당",
    max_points: int = 12000,
    basemap_style: str = "open-street-map",
) -> go.Figure:
    if df.empty:
        return empty_map_figure("표시할 투표소 포인트가 없습니다.")
    if "위도" not in df.columns or "경도" not in df.columns:
        return empty_map_figure("투표소 좌표 컬럼이 없어 지도를 그릴 수 없습니다.")

    points = df.loc[df["위도"].notna() & df["경도"].notna()].copy()
    if points.empty:
        return empty_map_figure("좌표가 있는 투표소가 없습니다.")

    size_arg: str | None = None
    if size_col in points.columns:
        size_values = pd.to_numeric(points[size_col], errors="coerce")
        if size_values.notna().any() and size_values.gt(0).any():
            points = points.assign(__size_metric=size_values)
            size_arg = "__size_metric"

    if len(points) > max_points:
        if size_arg is not None:
            points = points.nlargest(max_points, size_arg, keep="all")
        else:
            points = points.head(max_points).copy()

    hover_cols = [
        column
        for column in [
            "투표소명_F",
            "주소",
            "읍면동명_F",
            "선거인수",
            "투표수",
            "유효투표수",
            "사전투표 비중",
            "선거일투표 비중",
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
            "경쟁도지수",
        ]
        if column in points.columns
    ]
    map_view = _compute_map_view(points)
    color_arg: str | None = None
    if color_col in points.columns and points[color_col].notna().any():
        color_arg = color_col
    color_map = entity_color_map(points, color_arg, party_col="1위정당") if color_arg else None
    fig = px.scatter_mapbox(
        points,
        lat="위도",
        lon="경도",
        color=color_arg,
        size=size_arg,
        hover_name="투표소명_F" if "투표소명_F" in points.columns else None,
        hover_data=hover_cols,
        color_discrete_map=color_map,
        zoom=map_view["zoom"],
        center={"lat": map_view["lat"], "lon": map_view["lon"]},
        height=620,
    )
    fig.update_layout(
        title=title,
        mapbox_style=resolve_basemap_style(basemap_style),
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
    )
    return fig


def build_region_choropleth(
    df: pd.DataFrame,
    level: str,
    key_col: str,
    value_col: str,
    title: str,
    geometry_context: dict[str, Any],
    percent: bool = False,
    basemap_style: str = "white-bg",
    label_col: str | None = None,
    colorscale: list[list[object]] | None = None,
    prefer_geo: bool = True,
    show_value_label: bool = True,
) -> go.Figure:
    if value_col not in df.columns and "지도지표" in df.columns:
        value_col = "지도지표"
    if df.empty or key_col not in df.columns or value_col not in df.columns:
        return empty_map_figure("표시할 지역 집계가 없습니다.")
    if not geometry_context.get("available"):
        return empty_map_figure(geometry_context.get("message", "현재 경계 파일이 없어 색채 지도를 표시할 수 없습니다."))

    featureidkey = geometry_context["featureidkey"]
    geojson = _filter_geojson_for_locations(
        geometry_context["geojson"],
        featureidkey,
        df[key_col].astype("string"),
    )
    df = _filter_frame_for_geojson_locations(df, geojson, featureidkey, key_col)
    if df.empty:
        return empty_map_figure("표시할 경계와 매칭되는 지역 집계가 없습니다.")

    hover_cols = [
        column
        for column in [
            "지역",
            value_col,
            "정당명",
            "후보명",
            "득표수",
            "유효투표수",
            "득표율_계산",
            "기준대상",
            "비교대상",
            "기준정당",
            "비교정당",
            "기준득표수",
            "비교득표수",
            "기준득표율",
            "비교득표율",
            "스윙분류",
            "승리횟수스윙분류",
            "평균격차스윙분류",
            "민주당계 승리횟수",
            "국민의힘계 승리횟수",
            "제3지대 승리횟수",
            "무소속/기타 승리횟수",
            "기타 승리횟수",
            "민주당계 평균득표율",
            "국민의힘계 평균득표율",
            "승리횟수격차",
            "승리균형지수",
            "평균격차",
            "최근격차",
            "최저격차",
            "최고격차",
            "선거수",
            "격차방향",
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
            "경쟁도지수",
        ]
        if column in df.columns
    ]
    signed_gap_metric = value_col in {"부호득표수격차", "부호득표율격차"} or (
        value_col == "지도지표" and ("부호득표수격차" in df.columns or "부호득표율격차" in df.columns)
    )
    effective_colorscale = colorscale or GIS_THREE_TONE_COLOR_SCALE
    zmin = None
    zmax = None
    zmid = None
    if signed_gap_metric:
        effective_colorscale = colorscale or GIS_SIGNED_GAP_COLOR_SCALE
        abs_max = float(pd.Series(df[value_col]).dropna().abs().max()) if not df.empty else 0.0
        if abs_max <= 0:
            abs_max = 1.0
        zmin = -abs_max
        zmax = abs_max
        zmid = 0.0
    else:
        values = pd.to_numeric(df[value_col], errors="coerce").dropna()
        if not values.empty:
            value_min = float(values.min())
            value_max = float(values.max())
            if value_min == value_max:
                padding = max(abs(value_min) * 0.05, 0.01 if percent else 1.0)
                zmin = value_min - padding
                zmax = value_max + padding
            else:
                zmin = value_min
                zmax = value_max

    resolved_basemap_style = resolve_basemap_style(basemap_style)
    use_geo_trace = prefer_geo and resolved_basemap_style == "white-bg"
    if use_geo_trace:
        fig = go.Figure(
            go.Choropleth(
                geojson=geojson,
                locations=df[key_col],
                z=df[value_col],
                featureidkey=featureidkey,
                colorscale=effective_colorscale,
                zmin=zmin,
                zmax=zmax,
                zmid=zmid,
                marker_line_color="#334155",
                marker_line_width=0.7 if level == "읍면동" else 1.0,
                marker_opacity=0.72,
                customdata=df[hover_cols].to_numpy() if hover_cols else None,
                hovertemplate="<br>".join(
                    ["%{customdata[0]}"]
                    + [f"{column}: %{{customdata[{idx}]}}" for idx, column in enumerate(hover_cols[1:], start=1)]
                )
                + "<extra></extra>"
                if hover_cols
                else None,
            )
        )
        fig.update_layout(
            title=title,
            height=620,
            margin=dict(l=20, r=20, t=60, b=20),
            geo=dict(
                fitbounds="locations",
                visible=False,
                projection_type="mercator",
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        if percent and fig.data:
            fig.data[0].colorbar.tickformat = "+.0%" if signed_gap_metric else ".0%"

        if label_col and label_col in df.columns:
            label_points = _build_geojson_label_frame(geojson, featureidkey, key_col, df, label_col, value_col=value_col)
            if not label_points.empty:
                if show_value_label and value_col in label_points.columns:
                    value_source = label_points[value_col]
                    if isinstance(value_source, pd.DataFrame):
                        numeric_indices = [
                            index
                            for index in range(value_source.shape[1])
                            if pd.to_numeric(value_source.iloc[:, index], errors="coerce").notna().any()
                        ]
                        value_source = value_source.iloc[:, numeric_indices[0] if numeric_indices else 0]
                    label_value_col = "득표율격차" if signed_gap_metric and percent else value_col
                    value_labels = value_source.map(lambda value: _format_map_value_label(value, label_value_col, percent=percent))
                else:
                    value_labels = pd.Series([""] * len(label_points), index=label_points.index)
                label_text = label_points[label_col].astype("string")
                label_text = label_text.where(value_labels.eq(""), label_text + "<br>" + value_labels)
                fig.add_trace(
                    go.Scattergeo(
                        lat=label_points["lat"],
                        lon=label_points["lon"],
                        mode="text",
                        text=label_text,
                        textfont=dict(size=13 if level == "읍면동" else 15, color="#0f172a"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        return fig

    map_view = _compute_geojson_center(geojson)
    if label_col and label_col in df.columns:
        map_view["zoom"] = max(float(map_view["zoom"]) - 0.18, 0.0)

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=df[key_col],
            z=df[value_col],
            featureidkey=featureidkey,
            colorscale=effective_colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            marker_line_color="#334155",
            marker_line_width=0.7 if level == "읍면동" else 1.0,
            marker_opacity=0.72,
            customdata=df[hover_cols].to_numpy() if hover_cols else None,
            hovertemplate="<br>".join(
                ["%{customdata[0]}"]
                + [f"{column}: %{{customdata[{idx}]}}" for idx, column in enumerate(hover_cols[1:], start=1)]
            )
            + "<extra></extra>"
            if hover_cols
            else None,
        )
    )
    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox_style=resolved_basemap_style,
        mapbox_center={"lat": map_view["lat"], "lon": map_view["lon"]},
        mapbox_zoom=map_view["zoom"],
    )
    if percent and fig.data:
        fig.data[0].colorbar.tickformat = "+.0%" if signed_gap_metric else ".0%"

    if label_col and label_col in df.columns:
        label_points = _build_geojson_label_frame(geojson, featureidkey, key_col, df, label_col, value_col=value_col)
        if not label_points.empty:
            if show_value_label and value_col in label_points.columns:
                value_source = label_points[value_col]
                if isinstance(value_source, pd.DataFrame):
                    numeric_indices = [
                        index
                        for index in range(value_source.shape[1])
                        if pd.to_numeric(value_source.iloc[:, index], errors="coerce").notna().any()
                    ]
                    value_source = value_source.iloc[:, numeric_indices[0] if numeric_indices else 0]
                label_value_col = "득표율격차" if signed_gap_metric and percent else value_col
                value_labels = value_source.map(lambda value: _format_map_value_label(value, label_value_col, percent=percent))
            else:
                value_labels = pd.Series([""] * len(label_points), index=label_points.index)
            label_text = label_points[label_col].astype("string")
            label_text = label_text.where(value_labels.eq(""), label_text + "<br>" + value_labels)
            fig.add_trace(
                go.Scattermapbox(
                    lat=label_points["lat"],
                    lon=label_points["lon"],
                    mode="text",
                    text=label_text,
                    textfont=dict(size=13 if level == "읍면동" else 15, color="#0f172a"),
                    textposition="middle center",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return fig


def build_composite_region_polling_map(
    region_df: pd.DataFrame,
    polling_df: pd.DataFrame,
    level: str,
    key_col: str,
    value_col: str,
    title: str,
    geometry_context: dict[str, Any],
    percent: bool = False,
    point_size_col: str = "투표수",
    point_color_col: str = "1위정당",
    basemap_style: str = "white-bg",
    max_points: int = 4000,
    label_col: str | None = None,
    colorscale: list[list[object]] | None = None,
    show_value_label: bool = True,
) -> go.Figure:
    if not geometry_context.get("available"):
        return build_polling_point_map(
            polling_df,
            title=f"{title} / 포인트 지도",
            size_col=point_size_col,
            color_col=point_color_col,
            max_points=max_points,
            basemap_style=basemap_style,
        )

    fig = build_region_choropleth(
        region_df,
        level=level,
        key_col=key_col,
        value_col=value_col,
        title=title,
        geometry_context=geometry_context,
        percent=percent,
        basemap_style=basemap_style,
        label_col=label_col,
        colorscale=colorscale,
        prefer_geo=False,
        show_value_label=show_value_label,
    )

    if polling_df.empty or "위도" not in polling_df.columns or "경도" not in polling_df.columns:
        return fig

    points = polling_df.loc[polling_df["위도"].notna() & polling_df["경도"].notna()].copy()
    if points.empty:
        return fig

    if len(points) > max_points and point_size_col in points.columns:
        points = points.nlargest(max_points, point_size_col, keep="all")

    color_map = entity_color_map(points, point_color_col, party_col="1위정당") if point_color_col in points.columns else None
    grouped_points = points.groupby(point_color_col, observed=True) if point_color_col in points.columns else [(None, points)]

    for category, category_df in grouped_points:
        hover_cols = [
            column
            for column in [
                "투표소명_F",
                "읍면동명_F",
                "선거인수",
                "투표수",
                "유효투표수",
                "사전투표 비중",
                "선거일투표 비중",
                "1위후보",
                "1위정당",
                "2위후보",
                "2위정당",
                "득표수격차",
                "득표율격차",
                "경쟁도지수",
            ]
            if column in category_df.columns
        ]
        hover_lines: list[str] = []
        if hover_cols:
            hover_lines.append("%{customdata[0]}")
            for idx, column in enumerate(hover_cols[1:], start=1):
                hover_lines.append(f"{column}: %{{customdata[{idx}]}}")

        if point_size_col in category_df.columns:
            size_base = category_df[point_size_col].fillna(0)
            size_max = max(float(size_base.max()), 1.0)
            marker_size = size_base.clip(lower=1) / size_max * 18 + 4
        else:
            marker_size = 9

        fig.add_trace(
            go.Scattermapbox(
                lat=category_df["위도"],
                lon=category_df["경도"],
                mode="markers",
                name=str(category) if category is not None else "투표소",
                marker=dict(
                    size=marker_size,
                    color=(color_map or {}).get(str(category), "#D94C3D") if category is not None else "#D94C3D",
                    opacity=0.68,
                ),
                customdata=category_df[hover_cols].to_numpy() if hover_cols else None,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>" if hover_lines else None,
            )
        )
    return fig
