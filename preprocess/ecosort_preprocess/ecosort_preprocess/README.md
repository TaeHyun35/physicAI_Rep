# 에코소트 전처리 파이프라인

에코소트(엣지 AI 재활용품 자동 분류기)의 **전처리 단계**만 담은 코드입니다. 학습, 장치 제어, 시뮬레이션은 포함하지 않습니다.

- 실시간 전처리 S1~S14 (「전처리 상세 설계」 문서 구현)
- 학습 데이터 준비: 자체 촬영 데이터, 공개 데이터(TrashNet·TACO·AI Hub) 정렬, 물체 ID 단위 분할
- 학습용 증강
- 점검 도구: 단계별 시각화, 처리 시간 측정, 파라미터 비교 실험

## 1. 전체 흐름

```mermaid
flowchart TD
    BG[빈 라이트박스 10장<br/>data/backgrounds] --> CAL[캘리브레이션<br/>배경 평균 · 이득 맵 · 엣지 맵]

    subgraph 실시간 전처리 S1~S13
      F[원본 프레임] --> S1[S1 1/4 축소] --> S2[S2 노출 검사]
      S2 --> S36[S3~S6 Lab 가중 색차 차분<br/>이진화 · 열림 → 닫힘]
      S36 --> S7[S7 물체 선택<br/>조각 묶기 · 여러 개 판정]
      S7 --> S8{색차 마스크가<br/>비었거나 작거나 조각남?}
      S8 -- 예 --> E[S8 엣지 마스크 보조]
      S8 -- 아니오 --> S9
      E --> S9[S9~S10 원본 해상도 크롭 + 여백]
      S9 --> S11[S11 흐림 검사] --> S12[S12 이득 맵 곱셈<br/>조명 불균일 + 색 보정]
      S12 --> S13[S13 배경색 패딩 → 256]
    end
    CAL --> S36
    CAL --> S12

    OWN[자체 촬영<br/>data/raw/own] --> F
    PUB[공개 데이터<br/>TrashNet · TACO · AI Hub] --> CUT[물체 오려내기<br/>GrabCut / 분할 마스크]
    CUT --> COMP[빈 배경에 합성<br/>크기 · 위치 · 회전 · 그림자] --> F

    S13 --> OUT[data/processed 256x256]
    OUT --> MAN[manifest.csv<br/>물체 ID 단위 분할]
    OUT --> AUG[학습: augment → 224]
    OUT --> CC[검증·장치: 중앙 크롭 224]
    AUG --> N[S14 정규화]
    CC --> N
```

핵심은 **학습 데이터와 장치 입력이 같은 함수(`EcoPreprocessor.process`)를 거친다**는 점입니다. 공개 데이터도 에코소트 배경에 합성한 뒤 같은 전처리를 통과시켜, 학습과 실제 입력의 분포를 맞춥니다.

## 2. 폴더 구조

```
ecosort_preprocess/
├── run_preprocess.py        # 실행 명령 모음 (여기서 시작)
├── ecosort_colab.ipynb      # Google Colab 실행 노트북 (Drive 연동)
├── requirements.txt
└── ecopre/
    ├── config.py            # 경로, 클래스, 공개 데이터 매핑, 전처리 파라미터(PreConfig)
    ├── preprocess.py        # ★ 캘리브레이션 + S1~S13 + 정규화(S14) + 배경 유지 관리
    ├── augment.py           # 학습용 증강 (기하 변환 먼저, 색 변화 나중)
    ├── compose.py           # 오려낸 물체를 배경에 합성, 색 맞춤
    ├── download.py          # TrashNet, TACO 다운로드
    ├── prepare_own.py       # 자체 촬영 데이터 점검(check) + 전처리 + 흐림 기준 자동 설정
    ├── prepare_public.py    # 공개 데이터 오려내기 → 합성 → 전처리
    ├── manifest.py          # 학습/검증/테스트 분할 (물체 ID 단위)
    └── debug_tools.py       # 시각화, 처리 시간 측정, 파라미터 비교, 배경 점검(bgcheck)
```

실행하면 아래 데이터 폴더가 만들어집니다.

```
data/
├── backgrounds/             # (직접 준비) 빈 라이트박스 10장
├── raw/own/{train_pool,test,confuse}/   # (직접 준비) 자체 촬영 원본
├── raw/public/              # 다운로드한 공개 데이터
├── cutouts/                 # 오려낸 물체 (투명 배경 PNG)
├── processed/               # 전처리 결과 256x256 + index_own.csv, index_public.csv
├── manifest.csv             # 분할 목록
└── pre_config.json          # 조정된 전처리 설정 (학습·장치 공용)
```

## 3. 설치

```bash
pip install -r requirements.txt
```

Python 3.8 이상에서 동작합니다 (JetPack 5의 Python 3.8, Colab 포함). Jetson에서는 JetPack에 포함된 OpenCV를 그대로 써도 됩니다.

### Google Colab에서 실행

`ecosort_colab.ipynb`를 Colab에서 열고 위에서부터 실행합니다. Drive의 `MyDrive/ecosort/ecosort_preprocess`를 `/content`로 복사해 작업하고(Drive에서 직접 돌리면 작은 파일 입출력이 매우 느림), 결과는 `data_out.zip`으로 묶어 Drive에 되돌려 둡니다.

| Colab에서 | Jetson에서만 |
|---|---|
| 다운로드, 오려내기, 합성, `own`, `manifest`, `visualize`, `sweep`, 학습, ONNX 내보내기 | 처리 시간 실측(`benchmark`, H-03), TensorRT 엔진 빌드, 카메라 제어 |

`data/pre_config.json`은 학습과 장치가 같이 써야 하므로(원칙 P5) Colab 결과에서 꺼내 Jetson에 반드시 복사합니다.

## 4. 데이터 준비 규칙

**배경:** 장치와 같은 카메라 설정(노출·화이트밸런스·초점 수동 고정)으로 빈 라이트박스를 10장 찍어 `data/backgrounds/`에 넣습니다. 물체 사진과 해상도가 같아야 합니다. `public`(공개 데이터 합성)도 이 배경을 쓰므로, 실제 라이트박스 배경이 생기기 전에는 `download`와 오려내기까지만 의미가 있습니다.

**자체 촬영 파일명:** `클래스_품목_물체ID_조건_번호.jpg`

| 부분 | 예시 | 설명 |
|---|---|---|
| 클래스 | plastic | plastic / can / paper / etc. 헷갈리는 품목(`confuse` 폴더)도 **정답 클래스**를 씀 |
| 품목 | petclear | 자유롭게 (밑줄 `_` 사용 금지) |
| 물체ID | P0042 | 같은 물건이면 같은 ID. 학습/검증을 물체 단위로 나누는 기준 |
| 조건 | L1-C0-S1-F | 조명(L1/L0)-청결(C0/C1)-형태(S0/S1)-시점(F/S) |
| 번호 | 0001 | |

폴더가 곧 용도입니다. `train_pool`은 학습·검증, `test`는 고정 테스트셋(학습과 다른 물건만), `confuse`는 헷갈리는 품목 세트입니다.

- 확장자는 `.jpg/.jpeg/.png`, 대소문자를 구분하지 않습니다.
- 금속 감지 값과 무게는 같은 이름의 `.txt`로 함께 둡니다.
- 촬영 순서·물건 수 목표·조건 배분은 `docs/에코소트_현장조사_체크리스트.md`(개인 진행판)의 샘플 이미지 수집 지침을 따릅니다.

## 5. 실행 순서

```bash
# 1) 공개 데이터 받기
python run_preprocess.py download trashnet        # 약 43MB, 2,527장
python run_preprocess.py download taco --max-images 100   # 선택. 이미지는 Flickr에서 받음
python run_preprocess.py download aihub           # 수동 다운로드 안내 출력

# 2) 자체 촬영 데이터 점검 → 전처리 (흐림 기준값도 자동으로 정해 data/pre_config.json에 저장)
python run_preprocess.py check        # 파일명 규칙·물건 수/사진 수·평가 누수·조건 분포 (배경 없이 가능)
python run_preprocess.py own          # check 결과를 먼저 출력한 뒤 전처리

# 3) 공개 데이터: 오려내기 → 배경 합성 → 같은 전처리
python run_preprocess.py public --trashnet --per-cutout 2 --color-match   # 색 맞춤은 own 이후에만
python run_preprocess.py public --taco                       # TACO를 받았다면
python run_preprocess.py public --folder data/raw/public/aihub --map aihub_map.json

# 4) 학습/검증/테스트 분할
python run_preprocess.py manifest

# 2~4를 한 번에
python run_preprocess.py all --trashnet
```

## 6. 점검 도구

```bash
# 빈 배경 점검: 평균 밝기(목표 200~220), 모서리/중앙 비율(D-01), 포화(D-03), 해상도 혼재(H-02)
python run_preprocess.py bgcheck data/backgrounds

# 단계별 시각화 → debug/viz_<파일명>.jpg
python run_preprocess.py visualize data/raw/own/train_pool --limit 5

# 처리 시간(단계별 평균·p95)과 실패율 → 현장 조사 H-03 기록용
python run_preprocess.py benchmark data/raw/own/train_pool

# 파라미터 비교 (실험 E1: 이진화 임계값, E2: 밝기 가중치)
python run_preprocess.py sweep data/raw/own/train_pool --param thresh --values 12 20 30
python run_preprocess.py sweep data/raw/own/train_pool --param w_l --values 1.0 0.5 0.3
```

시각화 그림은 원본과 크롭 박스, 색차 마스크, 엣지 마스크, 원본 크롭, 색 보정 결과, 256 출력, 224 모델 입력, 증강 예시 4장, 단계별 처리 시간을 한 장에 보여 줍니다.

## 7. 다른 코드에서 쓰는 법

```python
import cv2
from ecopre.config import BACKGROUND_DIR
from ecopre.preprocess import EcoPreprocessor, center_crop, normalize

pre = EcoPreprocessor.from_dir(BACKGROUND_DIR)      # 부팅 시 1회
res = pre.process(cv2.imread("frame.jpg"))          # 256x256 결과
if res.ok:
    x = normalize(center_crop(res.image, 224))      # (3, 224, 224) float32 → 모델 입력
    print(res.flags)                                # edge_assist / edge_fallback / blurry / saturated
else:
    print(res.error)                                # no_object / multi_object / light_error
```

## 8. 파라미터 조정

모든 값은 `ecopre/config.py`의 `PreConfig`에 있고, 주석에 근거가 되는 현장 조사 항목 번호를 적어 두었습니다. 조정한 값은 `data/pre_config.json`에 저장해 두면 학습 데이터 준비와 장치가 같은 값을 씁니다.

| 상황 | 바꿀 값 |
|---|---|
| 그림자가 물체로 잡힘 (D-02) | `w_l`을 0.3으로 |
| 바닥 먼지가 잡힘 (A-05) | `open_k`를 5로 |
| 투명 물체 마스크가 끊김 (B-03) | `close_k`를 9로, `edge_thresh`를 낮춤 |
| 한 물체가 둘로 나뉨 | `group_k`를 키움 |
| 물체를 아예 못 찾음 | `thresh`를 낮춤 (sweep으로 확인) |

## 9. 확인한 동작

가상 라이트박스 데이터 328장(1280×720)과 실제 TrashNet 60장으로 아래를 확인했습니다. 처리 시간은 1코어 CPU 기준이므로, Jetson에서는 다시 측정해야 합니다.

- 자체 데이터 328장 모두 물체 검출 성공, 전처리 평균 6.1ms (p95 7.5ms, 예산 50ms 이내)
- 반투명 페트병처럼 색차로 라벨만 잡히는 경우, 엣지 보조로 병 전체가 잡힘
- TrashNet 60장 중 54장 오려내기 성공. 실패 6장은 골판지·종이가 사진을 꽉 채워 배경과 분리되지 않은 경우
- 공개 데이터 108장을 배경에 합성한 뒤 같은 전처리 통과, 실패 0장
- 분할 결과 학습과 평가에 같은 물체가 들어간 경우 없음

알려진 한계도 있습니다. GrabCut은 원본 배경 일부를 물체로 남기는 경우가 있어 `data/cutouts/`를 눈으로 한 번 훑어보고 잘못된 것을 지우는 것이 좋습니다. 또 TrashNet `metal`에는 캔이 아닌 금속도 섞여 있어 검수가 필요합니다.

## 10. 데이터셋 라이선스

공개 데이터셋은 각각 라이선스가 다릅니다. TACO 이미지는 Flickr 원저작자의 라이선스를 따르고, AI Hub 데이터는 이용 약관에 따른 승인이 필요합니다. 사용 전 각 데이터셋의 원 출처에서 라이선스를 확인하세요.
