"""
에코소트 전처리 설정.
경로, 클래스, 공개 데이터 매핑, 전처리 파라미터를 한 곳에서 관리한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. 경로 (프로젝트 폴더 기준)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PUBLIC_RAW_DIR = DATA_DIR / "raw" / "public"   # 공개 데이터 원본: trashnet/, taco/, aihub/
OWN_RAW_DIR = DATA_DIR / "raw" / "own"         # 자체 촬영 원본: train_pool/, test/, confuse/
BACKGROUND_DIR = DATA_DIR / "backgrounds"      # 빈 라이트박스 프레임 10장 (캘리브레이션용)
CUTOUT_DIR = DATA_DIR / "cutouts"              # 공개 데이터에서 오려낸 물체 (투명 배경 PNG)
PROCESSED_DIR = DATA_DIR / "processed"         # 전처리 결과 (256x256)
MANIFEST_PATH = DATA_DIR / "manifest.csv"      # 학습/검증/테스트 분할 목록
PRE_CONFIG_PATH = DATA_DIR / "pre_config.json" # 현장에서 조정한 전처리 설정 (학습·장치 공용)
DEBUG_DIR = ROOT / "debug"                     # 단계별 시각화 결과

OWN_SETS = ["train_pool", "test", "confuse"]

# 읽어 들일 이미지 확장자. 대소문자는 구분하지 않는다 (Colab·Jetson 같은 Linux에서 .JPG도 읽기 위함)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(folder: Path, recursive: bool = False) -> list:
    """폴더 안의 이미지 파일 목록 (확장자 대소문자 무시, 이름순)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

# ---------------------------------------------------------------------------
# 2. 클래스
# ---------------------------------------------------------------------------
CLASSES = ["plastic", "can", "paper", "etc"]

# TrashNet 폴더명 → 에코소트 클래스 (glass는 1차 버전에서 '기타')
TRASHNET_MAP = {
    "plastic": "plastic", "metal": "can", "paper": "paper",
    "cardboard": "paper", "glass": "etc", "trash": "etc",
}

# TACO 카테고리명 → 에코소트 클래스. 목록에 없으면 'etc'
# (국내 분리배출 기준: 비닐류·빨대·수저·유리·스티로폼은 4개 클래스 중 '기타')
TACO_MAP = {
    "Clear plastic bottle": "plastic", "Other plastic bottle": "plastic",
    "Disposable plastic cup": "plastic", "Other plastic cup": "plastic",
    "Disposable food container": "plastic", "Other plastic container": "plastic",
    "Plastic lid": "plastic", "Spread tub": "plastic", "Tupperware": "plastic",
    "Plastic bottle cap": "plastic", "Other plastic": "plastic",
    "Drink can": "can", "Food Can": "can", "Aerosol": "can",
    "Corrugated carton": "paper", "Drink carton": "paper", "Egg carton": "paper",
    "Meal carton": "paper", "Other carton": "paper", "Pizza box": "paper",
    "Magazine paper": "paper", "Normal paper": "paper", "Paper bag": "paper",
    "Paper cup": "paper", "Toilet tube": "paper", "Wrapping paper": "paper",
}

# ---------------------------------------------------------------------------
# 3. 전처리 파라미터 (전처리 상세 설계 S1~S13 초기값)
#    괄호 안 항목 번호는 현장 조사 체크리스트에서 이 값을 정할 근거가 되는 항목
# ---------------------------------------------------------------------------
@dataclass
class PreConfig:
    scale: int = 4                 # S1 마스크용 축소 배율 (H-02, H-03)
    w_l: float = 0.5               # S4 밝기 차이 가중치, 낮출수록 그림자에 둔감 (D-02)
    thresh: float = 20.0           # S5 이진화 임계값, 실험 E1: 12/20/30
    open_k: int = 3                # S6 열림 커널, 먼지 많으면 5 (A-05)
    close_k: int = 7               # S6 닫힘 커널, 투명 물체 많으면 9 (B-03)
    group_k: int = 11              # S7 조각을 한 물체로 묶는 팽창 크기
    min_area: float = 0.005        # S7 최소 물체 면적 (화면 대비)
    multi_ratio: float = 0.3       # S7 두 번째 물체가 이 비율 이상이면 '여러 개 투입' (C-03)
    small_obj_ratio: float = 0.015 # S8 색차 마스크가 이보다 작으면 엣지로 보조 (B-03)
    edge_thresh: float = 40.0      # S8 엣지 차이 임계값
    margin: float = 0.10           # S9 크롭 여백
    blur_thresh: float = 100.0     # S11 흐림 기준 (prepare_own이 자동 재설정)
    sat_level: int = 250           # S2 포화 픽셀 기준값
    sat_ratio: float = 0.05        # S2 포화 비율 경고 기준 (D-03)
    light_tol: float = 0.25        # S2 배경 대비 밝기 허용 범위 ±25% (A-02, A-03)
    target: float = 235.0          # S12 이득 맵 목표 밝기 (하이라이트 여유)
    gain_sigma: float = 25.0       # S12 이득 맵 평활화 정도 (D-01)
    bg_alpha: float = 0.05         # 배경 갱신 비율
    bg_dirty_ratio: float = 0.01   # 대기 상태 차분 면적이 이보다 크면 청소 필요


TRAIN_SIZE = 256   # 전처리 결과 저장 크기 (학습 증강 여유)
MODEL_SIZE = 224   # 모델 입력 크기
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 공개 데이터 합성 시 물체 크기(긴 변, 프레임 높이 대비)와 낙하 범위 (B-07, D-06)
OBJ_SIZE_RANGE = (0.30, 0.55)
DROP_ZONE = 0.15   # 화면 중심에서 ±15%
