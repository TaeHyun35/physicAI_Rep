"""
물체 합성 도구 (전처리 상세 설계 9-2).

공개 데이터에서 오려낸 물체(BGRA)를 빈 라이트박스 배경 위에 붙여,
'에코소트 장치가 찍은 것 같은' 프레임을 만든다. 데모용 가상 데이터 생성에도 같은 함수를 쓴다.
"""
import cv2
import numpy as np


def scale_to_long_side(patch: np.ndarray, long_px: int) -> np.ndarray:
    """긴 변이 long_px가 되도록 크기 조정."""
    h, w = patch.shape[:2]
    s = long_px / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(patch, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp)


def rotate_bgra(patch: np.ndarray, angle: float) -> np.ndarray:
    """잘리지 않도록 캔버스를 넓혀 회전 (투명 영역은 alpha=0)."""
    h, w = patch.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(patch, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def composite(frame: np.ndarray, patch: np.ndarray, cx: int, cy: int,
              shadow: float = 0.18, feather: int = 3) -> np.ndarray:
    """
    patch(BGRA)를 frame(BGR)의 (cx, cy) 중심에 붙인다.
    - feather: 경계를 살짝 흐려 '오려 붙인 티'가 나지 않게 한다 (모델이 합성 흔적을 배우지 않도록).
    - shadow : 물체 아래쪽에 흐린 접지 그림자를 만든다 (실제 라이트박스 그림자 흉내).
    """
    out = frame.astype(np.float32)
    H, W = frame.shape[:2]
    ph, pw = patch.shape[:2]
    alpha = patch[..., 3].astype(np.float32) / 255.0
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather / 2)

    def paste(layer_alpha, ox, oy, fn):
        # 화면 밖으로 나가는 부분을 잘라내고 겹치는 영역에만 적용
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + pw, W), min(oy + ph, H)
        if x1 <= x0 or y1 <= y0:
            return
        a = layer_alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox, None]
        fn(out[y0:y1, x0:x1], a, (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox)))

    ox, oy = cx - pw // 2, cy - ph // 2
    if shadow > 0:
        size = max(ph, pw)
        sh_alpha = cv2.GaussianBlur(alpha, (0, 0), max(2, size * 0.03))
        dx, dy = int(size * 0.02), int(size * 0.04)

        def darken(region, a, _):
            region *= (1 - shadow * a)
        paste(sh_alpha, ox + dx, oy + dy, darken)

    color = patch[..., :3].astype(np.float32)

    def blend(region, a, idx):
        region[:] = region * (1 - a) + color[idx] * a
    paste(alpha, ox, oy, blend)
    return np.clip(out, 0, 255).astype(np.uint8)


def reinhard_match(patch: np.ndarray, tgt_mean: np.ndarray, tgt_std: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    물체 픽셀의 Lab 평균·표준편차를 목표 분포 쪽으로 옮긴다 (Reinhard 색 전이, 실험 E4).
    amount=0.5면 절반만 옮겨 원래 색 특징을 너무 잃지 않게 한다.
    """
    bgr, a = patch[..., :3], patch[..., 3] > 127
    if a.sum() < 50:
        return patch
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m, s = lab[a].mean(axis=0), lab[a].std(axis=0) + 1e-6
    matched = (lab - m) / s * tgt_std + tgt_mean
    lab = lab * (1 - amount) + matched * amount
    out = patch.copy()
    out[..., :3] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out
