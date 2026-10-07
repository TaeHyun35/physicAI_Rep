"""
학습용 데이터 증강 (전처리 상세 설계 9-3).

외부 라이브러리 없이 OpenCV/numpy로 구현해 Jetson·PC 어디서나 같은 결과를 낸다.
순서 원칙: 기하 변환(회전·반전·왜곡·크롭)을 먼저, 색·화질 변화를 나중에.
입력: 전처리 결과 256x256 BGR  →  출력: 224x224 BGR (정규화는 preprocess.normalize에서)
"""
import cv2
import numpy as np

from .config import MODEL_SIZE
from .preprocess import border_color


def _rotate(img, angle, fill):
    """회전. 빈 영역은 배경색으로 채운다 (검은 모서리가 생기면 학습-추론 불일치)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def _elastic(img, rng, strength, fill):
    """격자 왜곡: 성긴 무작위 변위를 부드럽게 키워 구김·찌그러짐을 흉내 낸다."""
    h, w = img.shape[:2]
    grid = 4
    dx = cv2.resize(rng.uniform(-1, 1, (grid, grid)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.uniform(-1, 1, (grid, grid)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + dx * strength, ys + dy * strength, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def _random_crop(img, rng, scale=(0.85, 1.0)):
    """원본의 85~100% 영역을 잘라 224로 리사이즈 (ROI 크롭 오차 흉내)."""
    h, w = img.shape[:2]
    s = rng.uniform(*scale)
    ch, cw = int(h * s), int(w * s)
    t, l = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    return cv2.resize(img[t:t + ch, l:l + cw], (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)


def _coarse_dropout(img, rng, fill):
    """1~3개의 사각형 가림 (라벨 제거, 얼룩 흉내). 채움색은 배경색 또는 임의 색."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(rng.integers(1, 4)):
        s = int(rng.uniform(0.05, 0.15) * w)
        y, x = rng.integers(0, h - s), rng.integers(0, w - s)
        color = fill if rng.random() < 0.5 else tuple(int(v) for v in rng.integers(0, 256, 3))
        out[y:y + s, x:x + s] = color
    return out


def _brightness_contrast(img, rng, limit=0.10):
    """밝기·대비 ±10% (조명이 고정된 라이트박스라 약하게)."""
    alpha = 1.0 + rng.uniform(-limit, limit)       # 대비
    beta = 255 * rng.uniform(-limit, limit)        # 밝기
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _hue_saturation(img, rng, hue=5, sat=0.10):
    """색조 ±hue (OpenCV 0~179 단위), 채도 ±10%. 색은 클래스 단서라 약하게 (원칙 P6)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-hue, hue + 1)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-sat, sat)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _jpeg(img, rng):
    """웹캠 MJPEG 압축 잡음 흉내 (품질 70~95)."""
    q = int(rng.integers(70, 96))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def augment(img: np.ndarray, rng: np.random.Generator, fill=None,
            hue_range: int = 5, elastic_strength: float = 8.0) -> np.ndarray:
    """학습용 증강 파이프라인. 각 단계 옆 숫자는 적용 확률.
    fill: 회전 등으로 생긴 빈 영역 채움색. 없으면 이미지 테두리 색(= 그 이미지의 배경색)을 쓴다."""
    fill = fill or border_color(img)
    # --- 기하 변환 (먼저) ---
    img = _rotate(img, rng.uniform(0, 360), fill)                            # 1.0 회전 (위에서 촬영하므로 전 범위)
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)                                               # 0.5 좌우 반전
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)                                               # 0.5 상하 반전
    if rng.random() < 0.3:
        img = _elastic(img, rng, elastic_strength, fill)                     # 0.3 구김
    img = _random_crop(img, rng)                                             # 1.0 크롭 → 224
    if rng.random() < 0.3:
        img = _coarse_dropout(img, rng, fill)                                # 0.3 부분 가림
    # --- 색·화질 변화 (나중) ---
    if rng.random() < 0.5:
        img = _brightness_contrast(img, rng)                                 # 0.5 밝기·대비
    if rng.random() < 0.3:
        img = _hue_saturation(img, rng, hue=hue_range)                       # 0.3 색조·채도
    if rng.random() < 0.1:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)                               # 0.1 흐림
    if rng.random() < 0.3:
        img = _jpeg(img, rng)                                                # 0.3 JPEG 압축
    return img
