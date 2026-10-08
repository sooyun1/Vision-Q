# 기능별 상세 PRD

## 1. 메인 대시보드 (Dashboard)

### 목적

전사 검사 현황을 한눈에 모니터링한다.

### 요구사항

1. **KPI 카드**
   - 전체 검사 수, 정상 건수, 불량 건수, 불량률(%)을 출력한다.
2. **차트**
   - Plotly 기반의 결함 유형 파이 차트를 제공한다.
   - 정상/불량 비율 파이 차트를 제공한다.
3. **요약 표**
   - 최근 검사 데이터 5건을 DB와 실시간 연동하여 출력한다.

## 2. 실시간 결함 검사 (Inference & Inspection)

### 목적

이미지를 업로드하고 YOLO 추론을 수행한다.

### 요구사항

1. **모델 선택**
   - 라디오 버튼 또는 드롭다운으로 다음 모델 중 하나를 선택한다.
     - `best.pt/yolov10_best.pt`
     - `best.pt/yolov12_best.pt`
2. **이미지 입력**
   - 사용자가 `.jpg`, `.png`, `.bmp` 파일을 직접 업로드할 수 있다.
   - 또는 `dataset_det/test/images/` 내 실제 샘플 파일을 선택할 수 있다.
3. **AI 추론**
   - Ultralytics YOLO로 추론을 수행한다.
   - Bounding Box와 Confidence Score가 표시된 결과 이미지를 출력한다.
4. **DB 연동**
   - 추론 결과를 `inspections` 테이블에 즉시 `INSERT`한다.

## 3. 데이터 및 학습 관리 (Dataset & Training Analysis)

### 목적

AI팀의 데이터셋 현황을 확인하고 Colab 학습 결과를 모니터링한다.

### 요구사항

1. **데이터셋 현황 탭**
   - `dataset/bad/` 및 `dataset/good/` 폴더별 파일 개수를 집계한다.
   - 각 폴더의 이미지 미리보기를 제공한다.
2. **학습 결과 분석 탭**
   - `v10_experiment`와 `v12_experiment` 결과를 비교한다.
   - 각 실험의 `results.png`, `confusion_matrix.png`를 직접 시각화한다.
   - `results.csv`를 읽어 mAP, Precision, Recall 요약 표를 제공한다.

## 4. 품질 분석 리포트 (Analytics & Report)

### 목적

품질팀이 검사 이력을 분석하고 자동 보고서를 추출할 수 있게 한다.

### 요구사항

1. **검사 이력 조회**
   - `inspections` DB의 전체 이력을 테이블로 출력한다.
   - 정상/불량 판정별 필터링을 제공한다.
2. **시각화**
   - 일별 불량 발생 건수 라인 차트를 제공한다.
   - 모델별 불량 발생 건수 바 차트를 제공한다.
3. **내보내기**
   - pandas DataFrame 기반 CSV 다운로드 기능을 `st.download_button`으로 제공한다.
