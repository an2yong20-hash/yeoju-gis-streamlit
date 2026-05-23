from __future__ import annotations

"""Data loading entry points.

Rules:
- Streamlit app paths must read only cache parquet files.
- Raw CSV/TXT readers are batch-only and are intended for scripts such as
  `scripts/build_parquet.py`.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    BASE_DIR,
    CACHE_DIR,
    DIM_DIR,
    FACT_CONFIRMED_ELECTORATE_PATH,
    FACT_DIR,
    FACT_RESIDENT_COMPOSITION_PATH,
    FACT_VOTES_DIR,
    RAW_ENCODING,
)
from src.qa import check_required_columns
from src.utils import clean_text_columns

FACT_TURNOUT_COLUMNS = [
    "선거KEY",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "투표구명",
    "선거인수",
    "투표수",
    "유효투표수",
    "무효투표수",
    "기권수",
]

FACT_VOTES_RAW_COLUMNS = [
    "선거KEY",
    "정당KEY",
    "선거구명",
    "시도명",
    "구시군명",
    "읍면동명",
    "구분",
    "후보슬롯",
    "정당명",
    "후보명",
    "후보라벨",
    "유효투표수",
    "득표수",
    "득표율",
]

FACT_CONFIRMED_ELECTORATE_RAW_COLUMNS = [
    "선거KEY",
    "API선거ID",
    "시도명_API",
    "구시군명_API",
    "읍면동명_API",
    "투표구명_API",
    "인구수",
    "인구수_재외국민",
    "인구수_외국인",
    "확정선거인수",
    "확정선거인수_재외국민",
    "확정선거인수_외국인",
    "확정선거인수_남",
    "확정선거인수_남_재외국민",
    "확정선거인수_남_외국인",
    "확정선거인수_여",
    "확정선거인수_여_재외국민",
    "확정선거인수_여_외국인",
    "거소투표신고인명부등재자수",
    "거소투표신고인명부등재자수_재외국민",
    "거소투표신고인명부등재자수_남",
    "거소투표신고인명부등재자수_남_재외국민",
    "거소투표신고인명부등재자수_여",
    "거소투표신고인명부등재자수_여_재외국민",
]

FACT_RESIDENT_EXACT_AGE_COLUMNS = [f"{age}세" for age in range(0, 100)] + ["100세이상"]
FACT_RESIDENT_DETAIL_AGE_COLUMNS = ["70대인구", "80대인구", "90대인구", "100세이상인구"]

FACT_RESIDENT_COMPOSITION_RAW_COLUMNS = [
    "선거KEY",
    "API선거ID",
    "선거일",
    "선거연령기준",
    "기준월",
    "기준월라벨",
    "행정기관코드",
    "행정구역명_API",
    "시도명_API",
    "구시군명_API",
    "일반구명_API",
    "읍면동명_API",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "RowType",
    "구시군KEY",
    "읍면동KEY",
    "총인구수",
    "세대수",
    "세대당인구",
    "남자인구수",
    "여자인구수",
    "남여비율",
    "평균연령",
    "남자평균연령",
    "여자평균연령",
    "전월인구수",
    "당월인구수",
    "인구증감",
    "인구증감률",
    "전월남자인구수",
    "전월여자인구수",
    "당월남자인구수",
    "당월여자인구수",
    "남자인구증감",
    "여자인구증감",
    "아동인구",
    "청소년인구",
    "청년인구",
    "고령인구",
    "1인가구수",
    "청년1인가구수_추정",
    "노년1인가구수_추정",
    "10대인구",
    "20대인구",
    "30대인구",
    "40대인구",
    "50대인구",
    "60대인구",
    *FACT_RESIDENT_DETAIL_AGE_COLUMNS,
    "70세이상인구",
    *FACT_RESIDENT_EXACT_AGE_COLUMNS,
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
]

FACT_VOTES_TEXT_COLUMNS = ["구분", "정당명", "후보명", "후보라벨"]
FACT_CONFIRMED_ELECTORATE_TEXT_COLUMNS = ["선거KEY", "API선거ID", "시도명_API", "구시군명_API", "읍면동명_API", "투표구명_API"]
FACT_RESIDENT_COMPOSITION_TEXT_COLUMNS = [
    "선거KEY",
    "API선거ID",
    "선거일",
    "기준월",
    "기준월라벨",
    "행정기관코드",
    "행정구역명_API",
    "시도명_API",
    "구시군명_API",
    "일반구명_API",
    "읍면동명_API",
    "시도명",
    "구시군명",
    "일반구명",
    "읍면동명",
    "RowType",
    "구시군KEY",
    "읍면동KEY",
]
ELECTION_METADATA_COLUMNS = ["선거KEY", "선거시점", "선거명", "선거종류"]
PARTY_METADATA_COLUMNS = ["선거KEY", "정당KEY", "정당명", "구분", "구분2", "성향", "IsIndependent"]
PARTY_METADATA_OUTPUT_ALIASES = {
    "구분": "정당구분",
}
PARTY_METADATA_REQUEST_TO_SOURCE = {
    "정당구분": "구분",
    "구분2": "구분2",
    "성향": "성향",
    "IsIndependent": "IsIndependent",
    "정당명": "정당명",
    "정당KEY": "정당KEY",
    "선거KEY": "선거KEY",
}

CACHED_DIM_FILES = {
    "DimDong": CACHE_DIR / "dim_dong.parquet",
    "DimGusigun": CACHE_DIR / "dim_gusigun.parquet",
    "DimElection": CACHE_DIR / "dim_election.parquet",
    "DimParty": CACHE_DIR / "dim_party.parquet",
    "DimPollingPlace": CACHE_DIR / "dim_polling_place.parquet",
    "DimDongAlias": CACHE_DIR / "dim_dong_alias.parquet",
}

CACHED_FACT_FILES = {
    "FactConfirmedElectorate": CACHE_DIR / "fact_confirmed_electorate.parquet",
    "FactResidentComposition": CACHE_DIR / "fact_resident_composition.parquet",
    "FactTurnout": CACHE_DIR / "fact_turnout.parquet",
    "FactVotes": CACHE_DIR / "fact_votes.parquet",
}

RAW_CACHE_FILES = {
    "FactConfirmedElectorate_Raw": CACHE_DIR / "fact_confirmed_electorate_raw.parquet",
    "FactResidentComposition_Raw": CACHE_DIR / "fact_resident_composition_raw.parquet",
    "FactTurnout_Raw": CACHE_DIR / "fact_turnout_raw.parquet",
    "FactVotes_Raw": CACHE_DIR / "fact_votes_raw.parquet",
}

ALL_CACHE_FILES = {
    **CACHED_DIM_FILES,
    **RAW_CACHE_FILES,
    **CACHED_FACT_FILES,
}

CACHED_REQUIRED_COLUMNS = {
    "DimDong": ["읍면동KEY", "구시군KEY", "시도명", "구시군명", "읍면동명"],
    "DimGusigun": ["구시군KEY", "시도", "구시군"],
    "DimElection": ["선거KEY", "선거시점", "선거명", "선거종류"],
    "DimParty": ["선거KEY", "정당KEY", "정당명"],
    "DimPollingPlace": ["투표소KEY", "선거시점", "시도명_F", "구시군명_F"],
    "DimDongAlias": ["읍면동KEY_F", "읍면동KEY_D"],
    "FactConfirmedElectorate": ["선거KEY", "시도명", "구시군명", "읍면동명", "RowType", "확정선거인수"],
    "FactResidentComposition": ["선거KEY", "기준월", "시도명", "구시군명", "읍면동명", "RowType", "총인구수"],
    "FactTurnout": ["선거KEY", "시도명", "구시군명", "읍면동명", "RowType", "투표수"],
    "FactVotes": ["선거KEY", "시도명", "구시군명", "읍면동명", "RowType", "득표수"],
}

RAW_CACHE_REQUIRED_COLUMNS = {
    "FactConfirmedElectorate_Raw": FACT_CONFIRMED_ELECTORATE_RAW_COLUMNS,
    "FactResidentComposition_Raw": FACT_RESIDENT_COMPOSITION_RAW_COLUMNS,
    "FactTurnout_Raw": FACT_TURNOUT_COLUMNS,
    "FactVotes_Raw": FACT_VOTES_RAW_COLUMNS,
}

CACHE_REQUIRED_COLUMNS = {
    **CACHED_REQUIRED_COLUMNS,
    **RAW_CACHE_REQUIRED_COLUMNS,
}

ELECTION_DIM_COLUMNS = {
    "DimElection": ["선거KEY", "선거시점", "선거명", "선거종류"],
}

HOME_DIM_COLUMNS = {
    "DimDong": ["읍면동KEY", "시도명", "구시군명", "일반구명", "읍면동명"],
    "DimParty": ["선거KEY", "정당KEY", "정당명"],
    "DimPollingPlace": ["투표소KEY", "선거시점", "시도명_F", "구시군명_F", "일반구명_F", "읍면동명_F"],
}

HOME_FACT_COLUMNS = {
    "FactConfirmedElectorate": ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "투표구명", "RowType", "확정선거인수"],
    "FactResidentComposition": [
        "선거KEY",
        "기준월",
        "시도명",
        "구시군명",
        "일반구명",
        "읍면동명",
        "RowType",
        "총인구수",
        "청년인구",
        "1인가구수",
    ],
    "FactTurnout": ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "RowType", "선거인수", "투표수", "유효투표수", "무효투표수", "기권수"],
    "FactVotes": ["선거KEY", "선거구명", "시도명", "구시군명", "일반구명", "읍면동명", "구분", "RowType", "정당명", "유효투표수", "득표수"],
}

TURNOUT_FACT_COLUMNS = {
    "FactTurnout": ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "RowType", "선거인수", "투표수", "유효투표수", "무효투표수", "기권수"],
}

VOTES_FACT_COLUMNS = {
    "FactVotes": ["선거KEY", "정당KEY", "선거구명", "시도명", "구시군명", "일반구명", "읍면동명", "구분", "RowType", "정당명", "후보명", "후보라벨", "유효투표수", "득표수"],
}

CATEGORY_COLUMNS_BY_TABLE = {
    "DimDong": ["읍면동KEY", "구시군KEY", "시도명", "구시군명", "일반구명", "읍면동명"],
    "DimGusigun": ["구시군KEY", "권역", "시도", "구시군"],
    "DimElection": ["선거KEY", "선거종류", "선거명", "단위", "최소단위"],
    "DimParty": ["선거KEY", "정당KEY", "정당명", "구분", "구분2", "성향"],
    "DimPollingPlace": ["투표소KEY", "시도명_F", "구시군명_F", "일반구명_F", "읍면동명_F", "시도명_D", "구시군명_D", "일반구명_D", "읍면동명_D"],
    "DimDongAlias": ["읍면동KEY_F", "읍면동KEY_D", "시도명_F", "구시군명_F", "일반구명_F", "읍면동명_F"],
    "FactConfirmedElectorate": ["선거KEY", "API선거ID", "시도명", "구시군명", "일반구명", "읍면동명", "투표구명", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY"],
    "FactResidentComposition": [
        "선거KEY",
        "API선거ID",
        "기준월",
        "행정기관코드",
        "행정구역명_API",
        "시도명",
        "구시군명",
        "일반구명",
        "읍면동명",
        "RowType",
        "구시군KEY",
        "읍면동KEY",
    ],
    "FactTurnout": ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "투표구명", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY"],
    "FactVotes": ["선거KEY", "정당KEY", "선거구명", "시도명", "구시군명", "구시군명_원본", "일반구명", "읍면동명", "구분", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "정당명", "후보명", "후보라벨"],
    "FactConfirmedElectorate_Enriched": ["선거KEY", "API선거ID", "시도명", "구시군명", "일반구명", "읍면동명", "투표구명", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "선거명", "선거종류"],
    "FactResidentComposition_Enriched": [
        "선거KEY",
        "API선거ID",
        "기준월",
        "행정기관코드",
        "행정구역명_API",
        "시도명",
        "구시군명",
        "일반구명",
        "읍면동명",
        "RowType",
        "구시군KEY",
        "읍면동KEY",
        "선거명",
        "선거종류",
    ],
    "FactTurnout_Enriched": ["선거KEY", "시도명", "구시군명", "일반구명", "읍면동명", "투표구명", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "선거명", "선거종류"],
    "FactVotes_Enriched": ["선거KEY", "정당KEY", "선거구명", "시도명", "구시군명", "구시군명_원본", "일반구명", "읍면동명", "구분", "정당구분", "구분2", "성향", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "정당명", "후보명", "후보라벨", "선거명", "선거종류"],
}

NUMERIC_CASTS_BY_TABLE = {
    "DimElection": {"선거시점": "Int32"},
    "DimParty": {"선거시점": "Int32", "연번": "Int16", "IsIndependent": "Int8"},
    "DimPollingPlace": {"위도": "float32", "경도": "float32"},
    "FactConfirmedElectorate": {
        "선거연령기준": "Int8",
        "인구수": "Int32",
        "인구수_재외국민": "Int32",
        "인구수_외국인": "Int32",
        "확정선거인수": "Int32",
        "확정선거인수_재외국민": "Int32",
        "확정선거인수_외국인": "Int32",
        "확정선거인수_남": "Int32",
        "확정선거인수_남_재외국민": "Int32",
        "확정선거인수_남_외국인": "Int32",
        "확정선거인수_여": "Int32",
        "확정선거인수_여_재외국민": "Int32",
        "확정선거인수_여_외국인": "Int32",
        "거소투표신고인명부등재자수": "Int32",
        "거소투표신고인명부등재자수_재외국민": "Int32",
        "거소투표신고인명부등재자수_남": "Int32",
        "거소투표신고인명부등재자수_남_재외국민": "Int32",
        "거소투표신고인명부등재자수_여": "Int32",
        "거소투표신고인명부등재자수_여_재외국민": "Int32",
    },
    "FactResidentComposition": {
        "선거연령기준": "Int8",
        "총인구수": "Int32",
        "세대수": "Int32",
        "세대당인구": "float32",
        "남자인구수": "Int32",
        "여자인구수": "Int32",
        "남여비율": "float32",
        "평균연령": "float32",
        "남자평균연령": "float32",
        "여자평균연령": "float32",
        "전월인구수": "Int32",
        "당월인구수": "Int32",
        "인구증감": "Int32",
        "인구증감률": "float32",
        "전월남자인구수": "Int32",
        "전월여자인구수": "Int32",
        "당월남자인구수": "Int32",
        "당월여자인구수": "Int32",
        "남자인구증감": "Int32",
        "여자인구증감": "Int32",
        "아동인구": "Int32",
        "청소년인구": "Int32",
        "청년인구": "Int32",
        "고령인구": "Int32",
        "1인가구수": "Int32",
        "청년1인가구수_추정": "Int32",
        "노년1인가구수_추정": "Int32",
        "10대인구": "Int32",
        "20대인구": "Int32",
        "30대인구": "Int32",
        "40대인구": "Int32",
        "50대인구": "Int32",
        "60대인구": "Int32",
        "70대인구": "Int32",
        "80대인구": "Int32",
        "90대인구": "Int32",
        "100세이상인구": "Int32",
        "70세이상인구": "Int32",
        "선거연령청년인구_추정": "Int32",
        "남성구성비": "float32",
        "여성구성비": "float32",
        "선거연령청년구성비_추정": "float32",
        "30대구성비": "float32",
        "40대구성비": "float32",
        "50대구성비": "float32",
        "60대구성비": "float32",
        "70세이상구성비": "float32",
        "아동구성비": "float32",
        "청소년구성비": "float32",
        "청년구성비": "float32",
        "고령구성비": "float32",
        "1인가구비율": "float32",
        "청년1인가구비율_추정": "float32",
        "노년1인가구비율_추정": "float32",
        "1인가구연령통계가용여부": "Int8",
    },
    "FactTurnout": {
        "선거인수": "Int32",
        "투표수": "Int32",
        "유효투표수": "Int32",
        "무효투표수": "Int32",
        "기권수": "Int32",
    },
    "FactVotes": {
        "후보슬롯": "Int16",
        "유효투표수": "Int32",
        "득표수": "Int32",
        "득표율": "float32",
    },
    "FactTurnout_Enriched": {
        "선거시점": "Int32",
        "선거인수": "Int32",
        "투표수": "Int32",
        "유효투표수": "Int32",
        "무효투표수": "Int32",
        "기권수": "Int32",
    },
    "FactVotes_Enriched": {
        "선거시점": "Int32",
        "후보슬롯": "Int16",
        "유효투표수": "Int32",
        "득표수": "Int32",
        "득표율": "float32",
    },
    "FactConfirmedElectorate_Enriched": {
        "선거연령기준": "Int8",
        "선거시점": "Int32",
        "인구수": "Int32",
        "인구수_재외국민": "Int32",
        "인구수_외국인": "Int32",
        "확정선거인수": "Int32",
        "확정선거인수_재외국민": "Int32",
        "확정선거인수_외국인": "Int32",
        "확정선거인수_남": "Int32",
        "확정선거인수_남_재외국민": "Int32",
        "확정선거인수_남_외국인": "Int32",
        "확정선거인수_여": "Int32",
        "확정선거인수_여_재외국민": "Int32",
        "확정선거인수_여_외국인": "Int32",
        "거소투표신고인명부등재자수": "Int32",
        "거소투표신고인명부등재자수_재외국민": "Int32",
        "거소투표신고인명부등재자수_남": "Int32",
        "거소투표신고인명부등재자수_남_재외국민": "Int32",
        "거소투표신고인명부등재자수_여": "Int32",
        "거소투표신고인명부등재자수_여_재외국민": "Int32",
    },
    "FactResidentComposition_Enriched": {
        "선거시점": "Int32",
        "선거연령기준": "Int8",
        "총인구수": "Int32",
        "세대수": "Int32",
        "세대당인구": "float32",
        "남자인구수": "Int32",
        "여자인구수": "Int32",
        "남여비율": "float32",
        "평균연령": "float32",
        "남자평균연령": "float32",
        "여자평균연령": "float32",
        "전월인구수": "Int32",
        "당월인구수": "Int32",
        "인구증감": "Int32",
        "인구증감률": "float32",
        "전월남자인구수": "Int32",
        "전월여자인구수": "Int32",
        "당월남자인구수": "Int32",
        "당월여자인구수": "Int32",
        "남자인구증감": "Int32",
        "여자인구증감": "Int32",
        "아동인구": "Int32",
        "청소년인구": "Int32",
        "청년인구": "Int32",
        "고령인구": "Int32",
        "1인가구수": "Int32",
        "청년1인가구수_추정": "Int32",
        "노년1인가구수_추정": "Int32",
        "10대인구": "Int32",
        "20대인구": "Int32",
        "30대인구": "Int32",
        "40대인구": "Int32",
        "50대인구": "Int32",
        "60대인구": "Int32",
        "70대인구": "Int32",
        "80대인구": "Int32",
        "90대인구": "Int32",
        "100세이상인구": "Int32",
        "70세이상인구": "Int32",
        "선거연령청년인구_추정": "Int32",
        "남성구성비": "float32",
        "여성구성비": "float32",
        "선거연령청년구성비_추정": "float32",
        "30대구성비": "float32",
        "40대구성비": "float32",
        "50대구성비": "float32",
        "60대구성비": "float32",
        "70세이상구성비": "float32",
        "아동구성비": "float32",
        "청소년구성비": "float32",
        "청년구성비": "float32",
        "고령구성비": "float32",
        "1인가구비율": "float32",
        "청년1인가구비율_추정": "float32",
        "노년1인가구비율_추정": "float32",
        "1인가구연령통계가용여부": "Int8",
    },
}

for table_name in ("FactResidentComposition", "FactResidentComposition_Enriched"):
    NUMERIC_CASTS_BY_TABLE[table_name].update({column: "Int32" for column in FACT_RESIDENT_EXACT_AGE_COLUMNS})

CACHE_KEY_COLUMNS = {
    "DimDong": ["읍면동KEY"],
    "DimGusigun": ["구시군KEY"],
    "DimElection": ["선거KEY"],
    "DimParty": ["정당KEY"],
    "DimPollingPlace": ["투표소KEY"],
    "DimDongAlias": ["읍면동KEY_F"],
    "FactConfirmedElectorate": ["선거KEY", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "투표구명"],
    "FactResidentComposition": ["선거KEY", "RowType", "행정기관코드"],
    "FactTurnout": ["선거KEY", "RowType", "구시군KEY", "읍면동KEY", "투표소KEY", "투표구명"],
    "FactVotes": ["선거KEY", "RowType", "정당KEY", "구시군KEY", "읍면동KEY", "투표소KEY", "후보명", "구분"],
}


def _resolve_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = BASE_DIR / resolved_path
    return resolved_path


def _normalize_columns(columns: Iterable[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    return tuple(dict.fromkeys(columns))


def _cache_missing_message(path: Path) -> str:
    return f"캐시 파일이 없습니다: {path}\n먼저 python scripts/build_parquet.py 를 실행하세요."


def _cache_schema_message(name: str, missing_columns: list[str]) -> str:
    return (
        f"{name} cache parquet 스키마가 예상과 다릅니다. "
        "python scripts/build_parquet.py 를 다시 실행하세요. "
        f"누락 컬럼: {missing_columns}"
    )


def _validate_cache_schema(df: pd.DataFrame, name: str, required_columns: Iterable[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(_cache_schema_message(name, missing_columns))


def _optimize_loaded_frame(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    result = df.copy()

    for column, dtype in NUMERIC_CASTS_BY_TABLE.get(table_name, {}).items():
        if column not in result.columns:
            continue
        try:
            result[column] = result[column].astype(dtype)
        except (TypeError, ValueError):
            continue

    for column in CATEGORY_COLUMNS_BY_TABLE.get(table_name, []):
        if column not in result.columns:
            continue
        if isinstance(result[column].dtype, pd.CategoricalDtype):
            continue
        result[column] = result[column].astype("category")

    return result


@st.cache_data(show_spinner=False)
def _read_parquet_cached(
    path_str: str,
    columns: tuple[str, ...] | None,
    required_columns: tuple[str, ...] | None,
    name: str,
    optimize: bool,
) -> pd.DataFrame:
    resolved_path = Path(path_str)
    if not resolved_path.exists():
        raise FileNotFoundError(_cache_missing_message(resolved_path))

    try:
        dataframe = pd.read_parquet(resolved_path, columns=None if columns is None else list(columns))
    except Exception as exc:
        raise RuntimeError(
            f"{name} parquet를 읽지 못했습니다. "
            "cache 파일이 손상되었을 수 있습니다. "
            "python scripts/build_parquet.py 를 다시 실행하세요. "
            f"원인: {exc}"
        ) from exc

    if required_columns is not None:
        _validate_cache_schema(dataframe, name, required_columns)
    return _optimize_loaded_frame(dataframe, name) if optimize else dataframe


def load_parquet_table(
    path: str | Path,
    columns: Iterable[str] | None = None,
    required_columns: Iterable[str] | None = None,
    name: str | None = None,
    optimize: bool = True,
) -> pd.DataFrame:
    resolved_path = _resolve_path(path)
    table_name = name or resolved_path.stem
    return _read_parquet_cached(
        str(resolved_path),
        _normalize_columns(columns),
        _normalize_columns(required_columns),
        table_name,
        optimize,
    )


def read_csv_cp949(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_path}")

    read_options: dict[str, Any] = {"encoding": RAW_ENCODING, "low_memory": False}
    read_options.update(kwargs)

    try:
        return pd.read_csv(resolved_path, **read_options)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read CSV file '{resolved_path}' with encoding "
            f"'{read_options.get('encoding', RAW_ENCODING)}': {exc}"
        ) from exc


def read_tsv_cp949(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"TSV file not found: {resolved_path}")

    read_options: dict[str, Any] = {
        "encoding": RAW_ENCODING,
        "sep": "\t",
        "low_memory": False,
    }
    read_options.update(kwargs)

    try:
        return pd.read_csv(resolved_path, **read_options)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read TSV file '{resolved_path}' with encoding "
            f"'{read_options.get('encoding', RAW_ENCODING)}': {exc}"
        ) from exc


def load_dim_dong() -> pd.DataFrame:
    return read_csv_cp949(DIM_DIR / "DimDong.csv")


def load_dim_gusigun() -> pd.DataFrame:
    return read_csv_cp949(DIM_DIR / "DimGusigun.csv")


def load_dim_election() -> pd.DataFrame:
    return read_csv_cp949(DIM_DIR / "DimElection.csv")


def load_dim_party() -> pd.DataFrame:
    return read_csv_cp949(DIM_DIR / "DimParty.csv")


def load_dim_polling_place() -> pd.DataFrame:
    return read_csv_cp949(DIM_DIR / "DimPollingPlace.csv", dtype=str)


def load_dim_dong_alias() -> pd.DataFrame:
    return read_csv_cp949(DIM_DIR / "DimDongAlias.csv", dtype=str)


def load_all_dims() -> dict[str, pd.DataFrame]:
    return {
        "DimDong": load_dim_dong(),
        "DimGusigun": load_dim_gusigun(),
        "DimElection": load_dim_election(),
        "DimParty": load_dim_party(),
        "DimPollingPlace": load_dim_polling_place(),
        "DimDongAlias": load_dim_dong_alias(),
    }


def load_fact_turnout_raw() -> pd.DataFrame:
    dataframe = read_tsv_cp949(FACT_DIR / "FactTurnout.txt", dtype=str)
    check_required_columns(dataframe, FACT_TURNOUT_COLUMNS, "FactTurnout.txt")
    return dataframe.loc[:, FACT_TURNOUT_COLUMNS].dropna(how="all").reset_index(drop=True)


def load_fact_confirmed_electorate_raw() -> pd.DataFrame:
    dataframe = read_csv_cp949(FACT_CONFIRMED_ELECTORATE_PATH, dtype=str)
    check_required_columns(dataframe, FACT_CONFIRMED_ELECTORATE_RAW_COLUMNS, "FactConfirmedElectorate.csv")
    return dataframe.loc[:, FACT_CONFIRMED_ELECTORATE_RAW_COLUMNS].dropna(how="all").reset_index(drop=True)


def load_fact_resident_composition_raw() -> pd.DataFrame:
    dataframe = read_csv_cp949(FACT_RESIDENT_COMPOSITION_PATH, dtype=str)
    check_required_columns(dataframe, FACT_RESIDENT_COMPOSITION_RAW_COLUMNS, "FactResidentComposition.csv")
    return dataframe.loc[:, FACT_RESIDENT_COMPOSITION_RAW_COLUMNS].dropna(how="all").reset_index(drop=True)


def _get_fact_votes_files() -> list[Path]:
    if not FACT_VOTES_DIR.exists():
        raise FileNotFoundError(f"FactVotes folder not found: {FACT_VOTES_DIR}")

    vote_files = sorted(FACT_VOTES_DIR.glob("*.csv"))
    if not vote_files:
        raise FileNotFoundError(f"No FactVotes csv files found in: {FACT_VOTES_DIR}")
    return vote_files


def _prepare_fact_votes_frame(frame: pd.DataFrame, file_path: Path, drop_first_row: bool) -> pd.DataFrame:
    if frame.empty:
        return frame
    if frame.shape[1] != len(FACT_VOTES_RAW_COLUMNS):
        raise ValueError(
            f"FactVotes raw file '{file_path}' has {frame.shape[1]} columns; "
            f"expected {len(FACT_VOTES_RAW_COLUMNS)} columns."
        )

    result = frame.iloc[1:].copy() if drop_first_row else frame.copy()
    if result.empty:
        return result

    result.columns = FACT_VOTES_RAW_COLUMNS
    clean_text_columns(result, FACT_VOTES_TEXT_COLUMNS, remove_breaks=True)
    return result.dropna(how="all")


def iter_fact_votes_raw_chunks(chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    for file_path in _get_fact_votes_files():
        try:
            chunk_reader = read_csv_cp949(
                file_path,
                header=None,
                dtype=str,
                chunksize=chunksize,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load FactVotes raw file '{file_path}': {exc}") from exc

        first_chunk = True
        for chunk in chunk_reader:
            prepared = _prepare_fact_votes_frame(chunk, file_path, drop_first_row=first_chunk)
            first_chunk = False
            if not prepared.empty:
                yield prepared.reset_index(drop=True)


def load_fact_votes_raw_folder() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in iter_fact_votes_raw_chunks():
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=FACT_VOTES_RAW_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_all_raw() -> dict[str, pd.DataFrame]:
    raw_tables: dict[str, pd.DataFrame] = {
        "FactTurnout_Raw": load_fact_turnout_raw(),
        "FactVotes_Raw": load_fact_votes_raw_folder(),
    }
    if FACT_CONFIRMED_ELECTORATE_PATH.exists():
        raw_tables["FactConfirmedElectorate_Raw"] = load_fact_confirmed_electorate_raw()
    if FACT_RESIDENT_COMPOSITION_PATH.exists():
        raw_tables["FactResidentComposition_Raw"] = load_fact_resident_composition_raw()
    return raw_tables


@st.cache_data(show_spinner=False)
def load_cached_dim_table(
    cache_signature: tuple[tuple[str, float], ...],
    dim_name: str,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    del cache_signature
    requested_columns = columns or tuple(CACHED_REQUIRED_COLUMNS[dim_name])
    return load_parquet_table(
        CACHED_DIM_FILES[dim_name],
        columns=requested_columns,
        required_columns=requested_columns,
        name=dim_name,
        optimize=True,
    )


@st.cache_data(show_spinner=False)
def load_cached_fact_table(
    cache_signature: tuple[tuple[str, float], ...],
    fact_name: str,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    del cache_signature
    requested_columns = columns or tuple(CACHED_REQUIRED_COLUMNS[fact_name])
    return load_parquet_table(
        CACHED_FACT_FILES[fact_name],
        columns=requested_columns,
        required_columns=requested_columns,
        name=fact_name,
        optimize=True,
    )


def load_cached_dims(
    cache_signature: tuple[tuple[str, float], ...],
    dim_names: tuple[str, ...] | None = None,
    columns_map: dict[str, list[str] | tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    selected_dim_names = dim_names or tuple(CACHED_DIM_FILES.keys())
    requested_columns = columns_map or {}
    return {
        name: load_cached_dim_table(cache_signature, name, _normalize_columns(requested_columns.get(name)))
        for name in selected_dim_names
    }


def load_cached_facts(
    cache_signature: tuple[tuple[str, float], ...],
    fact_names: tuple[str, ...] | None = None,
    columns_map: dict[str, list[str] | tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    selected_fact_names = fact_names or tuple(CACHED_FACT_FILES.keys())
    requested_columns = columns_map or {}
    return {
        name: load_cached_fact_table(cache_signature, name, _normalize_columns(requested_columns.get(name)))
        for name in selected_fact_names
    }


def _merge_election_metadata(fact_df: pd.DataFrame, dim_election: pd.DataFrame) -> pd.DataFrame:
    if all(column in fact_df.columns for column in ELECTION_METADATA_COLUMNS):
        return fact_df
    return fact_df.merge(dim_election, on="선거KEY", how="left", copy=False)


def _merge_party_metadata(fact_df: pd.DataFrame, dim_party: pd.DataFrame) -> pd.DataFrame:
    requested_columns = [column for column in PARTY_METADATA_COLUMNS if column in dim_party.columns]
    if not requested_columns:
        return fact_df

    metadata_pairs = [
        (column, PARTY_METADATA_OUTPUT_ALIASES.get(column, column))
        for column in requested_columns
        if column not in {"선거KEY", "정당KEY"} and PARTY_METADATA_OUTPUT_ALIASES.get(column, column) not in fact_df.columns
    ]
    if not metadata_pairs:
        return fact_df

    source_columns = [source for source, _ in metadata_pairs]
    rename_map = {source: output for source, output in metadata_pairs if source != output}
    party_meta = (
        dim_party.loc[:, ["선거KEY", "정당KEY", *source_columns]]
        .rename(columns=rename_map)
        .drop_duplicates(subset=["선거KEY", "정당KEY"])
    )
    return fact_df.merge(party_meta, on=["선거KEY", "정당KEY"], how="left", copy=False)


@st.cache_data(show_spinner=False)
def load_fact_confirmed_electorate_enriched(
    cache_signature: tuple[tuple[str, float], ...],
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    normalized_columns = _normalize_columns(columns)
    if normalized_columns is None:
        fact_df = load_cached_fact_table(cache_signature, "FactConfirmedElectorate", None)
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        result = _merge_election_metadata(fact_df, dim_election)
        return _optimize_loaded_frame(result, "FactConfirmedElectorate_Enriched")

    requested_columns = set(normalized_columns)
    fact_columns = tuple(
        dict.fromkeys(["선거KEY", *[column for column in normalized_columns if column not in ELECTION_METADATA_COLUMNS]])
    )
    fact_df = load_cached_fact_table(cache_signature, "FactConfirmedElectorate", fact_columns)

    if requested_columns & (set(ELECTION_METADATA_COLUMNS) - {"선거KEY"}):
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        result = _merge_election_metadata(fact_df, dim_election)
    else:
        result = fact_df

    return _optimize_loaded_frame(result.loc[:, list(normalized_columns)], "FactConfirmedElectorate_Enriched")


@st.cache_data(show_spinner=False)
def load_fact_resident_composition_enriched(
    cache_signature: tuple[tuple[str, float], ...],
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    normalized_columns = _normalize_columns(columns)
    if normalized_columns is None:
        fact_df = load_cached_fact_table(cache_signature, "FactResidentComposition", None)
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        result = _merge_election_metadata(fact_df, dim_election)
        return _optimize_loaded_frame(result, "FactResidentComposition_Enriched")

    requested_columns = set(normalized_columns)
    fact_columns = tuple(
        dict.fromkeys(["선거KEY", *[column for column in normalized_columns if column not in ELECTION_METADATA_COLUMNS]])
    )
    fact_df = load_cached_fact_table(cache_signature, "FactResidentComposition", fact_columns)

    if requested_columns & (set(ELECTION_METADATA_COLUMNS) - {"선거KEY"}):
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        result = _merge_election_metadata(fact_df, dim_election)
    else:
        result = fact_df

    return _optimize_loaded_frame(result.loc[:, list(normalized_columns)], "FactResidentComposition_Enriched")


@st.cache_data(show_spinner=False)
def load_fact_turnout_enriched(
    cache_signature: tuple[tuple[str, float], ...],
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    normalized_columns = _normalize_columns(columns)
    if normalized_columns is None:
        fact_df = load_cached_fact_table(cache_signature, "FactTurnout", None)
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        result = _merge_election_metadata(fact_df, dim_election)
        return _optimize_loaded_frame(result, "FactTurnout_Enriched")

    requested_columns = set(normalized_columns)
    fact_columns = tuple(
        dict.fromkeys(["선거KEY", *[column for column in normalized_columns if column not in ELECTION_METADATA_COLUMNS]])
    )
    fact_df = load_cached_fact_table(cache_signature, "FactTurnout", fact_columns)

    if requested_columns & (set(ELECTION_METADATA_COLUMNS) - {"선거KEY"}):
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        result = _merge_election_metadata(fact_df, dim_election)
    else:
        result = fact_df

    return _optimize_loaded_frame(result.loc[:, list(normalized_columns)], "FactTurnout_Enriched")


@st.cache_data(show_spinner=False)
def load_fact_votes_enriched(
    cache_signature: tuple[tuple[str, float], ...],
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    normalized_columns = _normalize_columns(columns)
    if normalized_columns is None:
        fact_df = load_cached_fact_table(cache_signature, "FactVotes", None)
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        dim_party = load_cached_dim_table(cache_signature, "DimParty", tuple(PARTY_METADATA_COLUMNS))
        result = _merge_party_metadata(_merge_election_metadata(fact_df, dim_election), dim_party)
        return _optimize_loaded_frame(result, "FactVotes_Enriched")

    requested_columns = set(normalized_columns)
    party_request_columns = set(PARTY_METADATA_REQUEST_TO_SOURCE)
    requires_party_metadata = bool(requested_columns & (party_request_columns - {"선거KEY", "정당KEY", "정당명"}))
    requested_fact_columns = [
        column
        for column in normalized_columns
        if column not in ELECTION_METADATA_COLUMNS and column not in party_request_columns
    ]
    requested_party_identity_columns = [column for column in normalized_columns if column in {"정당KEY", "정당명"}]
    fact_columns = tuple(
        dict.fromkeys(
            ["선거KEY", *(["정당KEY"] if requires_party_metadata else []), *requested_fact_columns, *requested_party_identity_columns]
        )
    )
    fact_df = load_cached_fact_table(cache_signature, "FactVotes", fact_columns)

    if requested_columns & (set(ELECTION_METADATA_COLUMNS) - {"선거KEY"}):
        dim_election = load_cached_dim_table(cache_signature, "DimElection", tuple(ELECTION_METADATA_COLUMNS))
        result = _merge_election_metadata(fact_df, dim_election)
    else:
        result = fact_df

    if requires_party_metadata:
        dim_party_columns = tuple(
            dict.fromkeys(
                [
                    "선거KEY",
                    "정당KEY",
                    *[
                        PARTY_METADATA_REQUEST_TO_SOURCE[column]
                        for column in normalized_columns
                        if column in PARTY_METADATA_REQUEST_TO_SOURCE
                    ],
                ]
            )
        )
        dim_party = load_cached_dim_table(cache_signature, "DimParty", dim_party_columns)
        result = _merge_party_metadata(result, dim_party)

    return _optimize_loaded_frame(result.loc[:, list(normalized_columns)], "FactVotes_Enriched")


@st.cache_data(show_spinner=False)
def load_filter_options_from_cache(
    cache_signature: tuple[tuple[str, float], ...],
    dim_names: tuple[str, ...] | None,
    fact_names: tuple[str, ...] | None,
    dim_columns_map: dict[str, list[str] | tuple[str, ...]] | None,
    fact_columns_map: dict[str, list[str] | tuple[str, ...]] | None,
) -> dict[str, object]:
    from src.filters import build_filter_options

    dims = load_cached_dims(cache_signature, dim_names=dim_names, columns_map=dim_columns_map)
    facts = load_cached_facts(cache_signature, fact_names=fact_names, columns_map=fact_columns_map)
    return build_filter_options(facts, dims)


@st.cache_data(show_spinner=False)
def load_global_filter_options(
    cache_signature: tuple[tuple[str, float], ...],
) -> dict[str, object]:
    from src.filters import GLOBAL_FILTER_DIM_COLUMNS, GLOBAL_FILTER_FACT_COLUMNS

    return load_filter_options_from_cache(
        cache_signature,
        dim_names=tuple(GLOBAL_FILTER_DIM_COLUMNS.keys()),
        fact_names=tuple(GLOBAL_FILTER_FACT_COLUMNS.keys()),
        dim_columns_map=GLOBAL_FILTER_DIM_COLUMNS,
        fact_columns_map=GLOBAL_FILTER_FACT_COLUMNS,
    )


def load_app_data_from_cache(
    cache_signature: tuple[tuple[str, float], ...],
    dim_names: tuple[str, ...] | None = None,
    fact_names: tuple[str, ...] | None = None,
    dim_columns_map: dict[str, list[str] | tuple[str, ...]] | None = None,
    fact_columns_map: dict[str, list[str] | tuple[str, ...]] | None = None,
    include_filter_options: bool = True,
) -> dict[str, object]:
    dims = load_cached_dims(cache_signature, dim_names=dim_names, columns_map=dim_columns_map)
    facts = load_cached_facts(cache_signature, fact_names=fact_names, columns_map=fact_columns_map)
    result: dict[str, object] = {
        "dims": dims,
        "facts": facts,
    }
    if include_filter_options:
        result["filter_options"] = load_filter_options_from_cache(
            cache_signature,
            dim_names=dim_names,
            fact_names=fact_names,
            dim_columns_map=dim_columns_map,
            fact_columns_map=fact_columns_map,
        )
    return result


def get_cache_file_signatures() -> tuple[tuple[str, float], ...]:
    cache_paths = {**CACHED_DIM_FILES, **CACHED_FACT_FILES}
    signatures: list[tuple[str, float]] = []
    for name, path in cache_paths.items():
        resolved_path = _resolve_path(path)
        signatures.append((name, resolved_path.stat().st_mtime if resolved_path.exists() else -1.0))
    return tuple(sorted(signatures))
