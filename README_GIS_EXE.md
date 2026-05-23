# GIS분석 전용 EXE

이 빌드는 기존 전체 Streamlit 앱이 아니라 `pages/7_GIS분석.py` 화면만 실행하는 Windows 독립 실행형 패키지입니다.

## 실행 방법

1. `dist\GIS_analysis.exe`를 실행합니다.
2. EXE가 로컬 Streamlit 서버를 시작하고 기본 브라우저에서 GIS분석 화면을 엽니다.
3. 브라우저가 열리지 않으면 콘솔에 표시된 `http://127.0.0.1:<port>` 주소를 직접 여세요.

포트 `8501`부터 순서대로 사용 가능한 포트를 찾고, 모두 사용 중이면 임시 빈 포트를 자동 선택합니다.

## 빠른 실행 배포

`GIS_analysis.exe` 하나짜리 배포본은 매 실행마다 내부 파일을 임시폴더에 풀기 때문에 시작이 느릴 수 있습니다.
빠른 실행이 중요하면 `dist\GIS_analysis_fast.zip`을 배포하고, 받는 사람이 압축을 푼 뒤
`GIS_analysis_fast\GIS_analysis.exe`를 실행하게 하세요.

이 방식은 파일 하나만 보내는 방식은 아니지만, 매번 대용량 압축 해제를 하지 않아 실행 대기 시간이 줄어듭니다.

## 포함 데이터 범위

`gis_data`에 GIS분석 전용 경량 데이터만 포함합니다.

- `202506-P` 경기도 여주시 투표소 위치, 장소명, 주소, 위도, 경도
- `202506-P` 여주시 투표구별 선거일/비선거일 투표 및 후보별 득표 계산에 필요한 turnout/vote/confirmed rows
- 권역별 전략 요약에 필요한 `201806-L1`, `202404-N1`, `202506-P` 여주시 읍면동 득표 rows
- GIS 권역에 쓰이는 여주시 읍면동 경계 GeoJSON 일부
- 권역별 우선순위 목록은 GIS 페이지 코드의 `CUSTOM_GIS_AREAS`를 그대로 사용
- 사전투표소 위치는 `dim_polling_place.parquet`의 `*사전투표소` rows 사용
- 투표구 관할구역은 `여주시 투표구 관할구역.xlsx`를 번들 내부 `gis_data/home/Desktop`로 복사
- VWorld 타일 URL 설정과 hover 미니차트 구성은 기존 GIS 페이지 코드 그대로 사용

전체 원본 `DB/`, 원본 fact/dim CSV/TXT/XLSX, 전체 parquet cache는 EXE에 포함하지 않습니다.

## 다시 빌드

PowerShell에서 프로젝트 루트로 이동한 뒤 실행합니다.

```powershell
.\build_gis_exe.ps1
```

스크립트가 하는 일:

1. `scripts/build_gis_dataset.py`로 `gis_data/` 경량 데이터 재생성
2. 경량 데이터가 원본 cache에서 GIS 화면이 쓰는 subset과 일치하는지 검증
3. PyInstaller가 없으면 현재 venv에 설치
4. `gis_exe.spec`로 `dist\GIS_analysis.exe` 생성

이미 만든 `gis_data/`를 그대로 쓰려면:

```powershell
.\build_gis_exe.ps1 -SkipDataset
```

빠른 실행용 폴더/ZIP 빌드는:

```powershell
.\build_gis_fast_folder.ps1
```

## Known Limitations

- VWorld 배경지도와 hover 후보 사진, Plotly JS CDN을 로드하려면 실행 PC에 인터넷 연결이 필요합니다.
- Windows 보안 정책이나 백신이 첫 실행 시 PyInstaller EXE를 검사하느라 실행이 늦을 수 있습니다.
- 데이터 갱신 시 원본 `cache/*.parquet`와 `여주시 투표구 관할구역.xlsx`를 먼저 갱신한 뒤 다시 빌드해야 합니다.
- EXE는 GIS분석 화면 전용이며, 기존 전체 앱의 다른 페이지는 포함하지 않습니다.
