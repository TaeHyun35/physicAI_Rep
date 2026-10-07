"""
에코소트 전처리 모듈 (전처리 상세 설계 S1~S14 구현).

핵심 원칙 P5: 학습 데이터 준비와 장치 실시간 추론이 '같은 함수'를 쓴다.
  - 학습 데이터·장치 공통: EcoPreprocessor.process(frame, out_size=256) → 256x256
  - 검증·장치 추론:       center_crop(img, 224)   (학습은 augment()의 랜덤 크롭으로 224)
  - 모델 입력 변환:       normalize(img)  (학습 Dataset과 장치 Classifier가 공통 사용)
  - 전처리 설정:          data/pre_config.json 을 학습과 장치가 함께 읽는다
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGENET_MEAN, IMAGENET_STD, PRE_CONFIG_PATH, PreConfig


def load_pre_config() -> PreConfig:
    """현장에서 조정한 전처리 설정(data/pre_config.json)이 있으면 읽고, 없으면 기본값."""
    if PRE_CONFIG_PATH.exists():
        return PreConfig(**json.loads(PRE_CONFIG_PATH.read_text()))
    return PreConfig()


def save_pre_config(cfg: PreConfig):
    """학습과 장치가 같은 전처리 설정을 쓰도록 파일로 저장한다 (원칙 P5)."""
    PRE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRE_CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=1))


@dataclass
class PreResult:
    """전처리 결과. image가 None이면 error에 이유가 들어 있다."""
    image: np.ndarray | None
    error: str | None = None            # light_error / no_object / multi_object
    flags: list = field(default_factory=list)  # edge_fallback / blurry / saturated
    box: tuple | None = None            # 원본 좌표 크롭 영역 (x0, y0, x1, y1)
    sharpness: float = 0.0              # S11 라플라시안 분산 (흐림 기준값 재설정에 사용)

    @property
    def ok(self) -> bool:
        return self.image is not None


def _edge_map(img_bgr: np.ndarray) -> np.ndarray:
    """Sobel 기울기 크기. 투명 물체 대체 마스크(S8)에 사용."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


class _StepTimer:
    """단계별 처리 시간(ms)과 중간 이미지를 debug dict에 기록. debug가 None이면 아무것도 하지 않는다."""

    def __init__(self, debug: dict | None):
        self.debug = debug
        self.t = time.perf_counter()
        if debug is not None:
            debug.setdefault("times_ms", {})
            debug.setdefault("images", {})

    def lap(self, name: str, **images):
        if self.debug is None:
            return
        now = time.perf_counter()
        self.debug["times_ms"][name] = (now - self.t) * 1000
        self.debug["images"].update(images)
        self.t = now


class EcoPreprocessor:
    """빈 라이트박스 배경으로 캘리브레이션한 뒤, 입력 프레임에서 물체를 찾아 잘라내고 보정한다."""

    def __init__(self, bg_frames: list, cfg: PreConfig | None = None):
        self.cfg = cfg or PreConfig()
        # [캘리브레이션] 배경 여러 장을 평균해 센서 잡음을 줄인다.
        bg = np.mean(np.stack([f.astype(np.float32) for f in bg_frames]), axis=0)
        self.frame_shape = bg.shape[:2]

        # [캘리브레이션] 조명·색 보정 이득 맵 = 목표 밝기 / (크게 흐린 배경)
        #   흰 배경이 곧 조명 분포이므로, 곱하면 비네팅과 색 틀어짐이 한 번에 보정된다.
        illum = cv2.GaussianBlur(bg, (0, 0), sigmaX=self.cfg.gain_sigma)
        self.gain = (self.cfg.target / np.maximum(illum, 1.0)).astype(np.float32)
        corrected_bg = np.clip(bg * self.gain, 0, 255)
        self.pad_color = tuple(int(v) for v in corrected_bg.mean(axis=(0, 1)))

        # [캘리브레이션] 축소 배경 (S1~S4와 똑같이 처리해 둔다: 원칙 P4)
        self._set_small_background(self._shrink(bg.astype(np.uint8)).astype(np.float32))

    # ------------------------------------------------------------------ 생성 도우미
    @classmethod
    def from_dir(cls, bg_dir: Path, cfg: PreConfig | None = None) -> "EcoPreprocessor":
        """폴더의 배경 이미지(jpg/png)로 캘리브레이션. cfg가 없으면 저장된 설정(data/pre_config.json)을 쓴다."""
        cfg = cfg or load_pre_config()
        files = sorted([p for p in Path(bg_dir).iterdir() if p.suffix.lower() in (".jpg", ".png")])
        if not files:
            raise FileNotFoundError(f"배경 이미지가 없습니다: {bg_dir}")
        return cls([cv2.imread(str(p)) for p in files], cfg)

    # ------------------------------------------------------------------ 내부 함수
    def _shrink(self, img: np.ndarray) -> np.ndarray:
        """S1: 1/scale 축소 (INTER_AREA는 평균을 내므로 잡음도 줄어든다)."""
        s = self.cfg.scale
        return cv2.resize(img, (img.shape[1] // s, img.shape[0] // s), interpolation=cv2.INTER_AREA)

    def _set_small_background(self, small_bg: np.ndarray):
        self.bg_small = small_bg  # float32, 배경 갱신(7장)에 사용
        u8 = np.clip(small_bg, 0, 255).astype(np.uint8)
        self.bg_brightness = float(u8.mean())
        self.bg_lab = cv2.cvtColor(cv2.GaussianBlur(u8, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
        self.bg_edge = _edge_map(u8)

    def _diff_mask(self, small: np.ndarray) -> np.ndarray:
        """S3~S6: 블러 → Lab → 가중 색차 차분 → 이진화 → 열림 → 닫힘."""
        c = self.cfg
        lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
        d = np.abs(lab - self.bg_lab).astype(np.float32)
        # 그림자는 밝기(L)만 크게 바뀌므로 L에 낮은 가중치. 합 대신 최댓값으로 한 채널 신호도 살린다.
        diff = np.maximum(c.w_l * d[..., 0], np.maximum(d[..., 1], d[..., 2]))
        mask = (diff > c.thresh).astype(np.uint8) * 255
        # 열림(잡음 제거)을 먼저, 닫힘(구멍 메우기)을 나중에: 잡음이 물체에 붙는 것을 막는다.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((c.open_k, c.open_k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((c.close_k, c.close_k), np.uint8))
        return mask

    def _select(self, mask: np.ndarray):
        """
        S7: 물체 선택.
        - 가까이 붙은 조각(캔의 띠와 뚜껑처럼 한 물체가 끊겨 보이는 경우)은 팽창한 마스크로 한 묶음으로 본다.
        - 묶음마다 실제 마스크 면적을 재고, 최소 면적 미만은 버린다.
        - 비슷한 크기의 묶음이 둘 이상이면 'multi' (여러 개 동시 투입).
        """
        total = mask.shape[0] * mask.shape[1]
        grouped = cv2.dilate(mask, np.ones((self.cfg.group_k, self.cfg.group_k), np.uint8))
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.countNonZero(mask[y:y + h, x:x + w])   # 팽창 전 실제 면적
            if area >= self.cfg.min_area * total:
                cands.append((area, (x, y, w, h)))
        if not cands:
            return None, 0.0, False
        cands.sort(key=lambda t: t[0], reverse=True)
        multi = len(cands) > 1 and cands[1][0] >= self.cfg.multi_ratio * cands[0][0]
        return cands[0][1], cands[0][0] / total, multi

    # ------------------------------------------------------------------ 메인 함수
    def process(self, frame: np.ndarray, out_size: int = 256, debug: dict | None = None) -> PreResult:
        """
        원본 프레임 → 보정된 정사각형 물체 이미지 (out_size x out_size, BGR uint8).
        debug에 빈 dict를 넘기면 단계별 처리 시간(ms)과 중간 이미지를 채워 준다 (시각화·실측용).
        """
        c = self.cfg
        if frame.shape[:2] != self.frame_shape:
            raise ValueError(f"프레임 크기 {frame.shape[:2]}가 배경 {self.frame_shape}와 다릅니다.")
        h, w = frame.shape[:2]
        flags = []
        timer = _StepTimer(debug)

        # S1 축소 (위치 찾기는 저해상도로: 원칙 P2)
        small = self._shrink(frame)
        timer.lap("S1_shrink", small=small)

        # S2 노출 검사 (싼 검사를 먼저: 원칙 P1)
        if abs(small.mean() - self.bg_brightness) > c.light_tol * self.bg_brightness:
            return PreResult(None, "light_error")
        if (small.max(axis=2) >= c.sat_level).mean() > c.sat_ratio:
            flags.append("saturated")
        timer.lap("S2_exposure")

        # S3~S6 배경 차분 마스크, S7 물체 선택
        cmask = self._diff_mask(small)
        timer.lap("S3-S6_diff_mask", color_mask=cmask)
        box, area, multi = self._select(cmask)
        timer.lap("S7_select")

        # S8 엣지 마스크 보조 (색차 마스크가 비었거나, 너무 작거나, 조각나 보일 때만)
        #   투명 페트·흰 종이는 색 차이가 작아 일부만 잡히거나 여러 조각으로 끊긴다.
        #   굴절·윤곽에서 생기는 엣지를 더해 물체 전체를 다시 찾는다.
        if box is None or area < c.small_obj_ratio or multi:
            edge = np.clip(_edge_map(small) - self.bg_edge, 0, None)
            emask = (edge > c.edge_thresh).astype(np.uint8) * 255
            emask = cv2.morphologyEx(emask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            nbox, _, nmulti = self._select(cv2.bitwise_or(cmask, emask))
            if nbox is not None and not nmulti:
                # 색차로 아무것도 못 찾은 경우만 'edge_fallback'(판정 기준 상향 대상)으로 표시
                flags.append("edge_fallback" if box is None else "edge_assist")
                box, multi = nbox, False
            timer.lap("S8_edge_assist", edge_mask=emask)
        if multi:
            return PreResult(None, "multi_object")
        if box is None:
            return PreResult(None, "no_object")

        # S9 박스를 원본 좌표로 확대 + 여백 (마스크가 놓친 가장자리까지 포함)
        x, y, bw, bh = [v * c.scale for v in box]
        mx, my = int(bw * c.margin), int(bh * c.margin)
        x0, y0 = max(x - mx, 0), max(y - my, 0)
        x1, y1 = min(x + bw + mx, w), min(y + bh + my, h)

        # S10 원본 해상도에서 크롭 (질감 보존: 원칙 P2)
        crop = frame[y0:y1, x0:x1]
        timer.lap("S9-S10_crop", crop_raw=crop)

        # S11 흐림 검사 (배경이 아닌 물체 영역에서 측정)
        probe_w = 256
        probe = cv2.resize(crop, (probe_w, max(1, int(probe_w * crop.shape[0] / crop.shape[1]))))
        sharp = float(cv2.Laplacian(cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if sharp < c.blur_thresh:
            flags.append("blurry")
        timer.lap("S11_blur_check")

        # S12 조명·색 보정: 이득 맵의 같은 영역을 곱한다 (크롭 후라 계산량이 작다: 원칙 P3)
        crop = np.clip(crop.astype(np.float32) * self.gain[y0:y1, x0:x1], 0, 255).astype(np.uint8)
        timer.lap("S12_color_correct", crop_corrected=crop)

        # S13 배경색으로 정사각형 패딩 → 리사이즈 (늘이지 않아 형태 단서 보존)
        #   패딩색 = 이 크롭 테두리의 중앙값. 조명이 약간 달라져도 크롭과 패딩 사이에 가짜 경계가 생기지 않는다.
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        top, left = (side - ch) // 2, (side - cw) // 2
        crop = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left,
                                  cv2.BORDER_CONSTANT, value=border_color(crop))
        interp = cv2.INTER_AREA if side >= out_size else cv2.INTER_LINEAR
        out = cv2.resize(crop, (out_size, out_size), interpolation=interp)
        timer.lap("S13_pad_resize", output=out)
        return PreResult(out, None, flags, (x0, y0, x1, y1), sharp)

    # ------------------------------------------------------------------ 배경 유지 관리 (7장)
    def idle_dirty_ratio(self, frame: np.ndarray) -> float:
        """물체가 없는 대기 상태 프레임에서 배경과 다른 영역의 비율."""
        mask = self._diff_mask(self._shrink(frame))
        return float((mask > 0).mean())

    def update_background(self, frame: np.ndarray) -> bool:
        """대기 상태에서 배경을 천천히 갱신한다. 얼룩 등 큰 변화가 있으면 갱신하지 않고 False."""
        if self.idle_dirty_ratio(frame) > self.cfg.bg_dirty_ratio:
            return False
        a = self.cfg.bg_alpha
        self._set_small_background((1 - a) * self.bg_small + a * self._shrink(frame).astype(np.float32))
        return True


# ---------------------------------------------------------------------------
# S14 모델 입력 변환 (학습 Dataset과 장치 추론이 공통으로 사용)
# ---------------------------------------------------------------------------
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def normalize(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) → RGB, 0~1, ImageNet 정규화, CHW float32."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb.transpose(2, 0, 1) - _MEAN) / _STD).astype(np.float32)


def border_color(img: np.ndarray, width: int = 4) -> tuple:
    """이미지 바깥 테두리 픽셀의 중앙값 색 (= 그 이미지의 배경색 추정)."""
    b = np.concatenate([img[:width].reshape(-1, 3), img[-width:].reshape(-1, 3),
                        img[:, :width].reshape(-1, 3), img[:, -width:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(b, axis=0))


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """256 저장 이미지 → 224 중앙 크롭 (검증·테스트용)."""
    h, w = img.shape[:2]
    t, l = (h - size) // 2, (w - size) // 2
    return img[t:t + size, l:l + size]
