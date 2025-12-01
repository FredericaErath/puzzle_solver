"""
preprocess.py

Preprocessing v8 (Paranoid Auto-Detection).
Updates:
1. Extremely sensitive detection (1% black pixels triggers shrink).
2. Fixes black lines by aggressively deciding to crop.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import argparse


@dataclass
class EdgeFeatures:
    mean_color: Tuple[float, float, float]
    color_hist: np.ndarray
    color_profile: np.ndarray
    gradient: np.ndarray


@dataclass
class PuzzlePiece:
    id: int
    image: np.ndarray
    mask: np.ndarray
    canvas_corners: np.ndarray
    size: Tuple[int, int]
    edges: Dict[str, EdgeFeatures]
    edge_lines: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    idx = np.argsort(angles)
    pts_ccw = pts[idx]
    sums = pts_ccw.sum(axis=1)
    tl_idx = np.argmin(sums)
    pts_ccw = np.roll(pts_ccw, -tl_idx, axis=0)
    v1 = pts_ccw[1] - pts_ccw[0]
    v2 = pts_ccw[2] - pts_ccw[0]
    cross = np.cross(v1, v2)
    if cross < 0:
        pts_ccw = np.array([pts_ccw[0], pts_ccw[3], pts_ccw[2], pts_ccw[1]], dtype=np.float32)
    return pts_ccw


def load_canvas_rgb(image_path: str, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".rgb":
        if width is None or height is None:
            raise ValueError("Raw .rgb file requires explicit width and height.")
        data = np.fromfile(image_path, dtype=np.uint8)
        planes = data.reshape((3, height, width))
        return np.stack([planes[2], planes[1], planes[0]], axis=2)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"Cannot read {image_path}")
    return img


def warp_and_process(canvas, corners, contour, mode, shrink_px):
    ordered = order_corners(corners)
    (tl, tr, br, bl) = ordered
    w_top = np.linalg.norm(tr - tl);
    w_bot = np.linalg.norm(br - bl)
    max_w = int(max(w_top, w_bot))
    h_left = np.linalg.norm(bl - tl);
    h_right = np.linalg.norm(br - tr)
    max_h = int(max(h_left, h_right))

    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    piece_img = cv2.warpPerspective(canvas, M, (max_w, max_h))

    mask_canvas = np.zeros(canvas.shape[:2], dtype=np.uint8)
    if mode == 'irregular':
        cv2.drawContours(mask_canvas, [contour], -1, 255, -1)
        piece_mask = cv2.warpPerspective(mask_canvas, M, (max_w, max_h))
        piece_mask = cv2.erode(piece_mask, np.ones((3, 3), np.uint8), iterations=1)
    else:
        piece_mask = np.full((max_h, max_w), 255, dtype=np.uint8)

    if shrink_px > 0:
        if max_h > 2 * shrink_px and max_w > 2 * shrink_px:
            piece_img = piece_img[shrink_px:-shrink_px, shrink_px:-shrink_px]
            piece_mask = piece_mask[shrink_px:-shrink_px, shrink_px:-shrink_px]

    _, piece_mask = cv2.threshold(piece_mask, 127, 255, cv2.THRESH_BINARY)
    piece_img = cv2.bitwise_and(piece_img, piece_img, mask=piece_mask)
    return piece_img, piece_mask


def analyze_raw_pieces(canvas, raw_contours):
    solidity_scores = []
    border_black_scores = []
    sample_cnts = raw_contours[:10]

    for cnt in sample_cnts:
        rect = cv2.minAreaRect(cnt)
        box_area = rect[1][0] * rect[1][1]
        cnt_area = cv2.contourArea(cnt)
        if box_area > 0: solidity_scores.append(cnt_area / box_area)

        box = cv2.boxPoints(rect)
        corners = np.array(box, dtype=np.float32)
        temp_img, _ = warp_and_process(canvas, corners, cnt, mode='rect', shrink_px=0)

        h, w = temp_img.shape[:2]
        if h > 10 and w > 10:
            strips = [temp_img[0:2, :, :], temp_img[h - 2:h, :, :], temp_img[:, 0:2, :], temp_img[:, w - 2:w, :]]
            total_px = 0;
            black_px = 0
            for s in strips:
                is_black = np.all(s < 10, axis=2)  # Stricter black threshold
                black_px += np.sum(is_black)
                total_px += s.shape[0] * s.shape[1]
            if total_px > 0: border_black_scores.append(black_px / total_px)

    avg_solidity = np.mean(solidity_scores) if solidity_scores else 1.0
    avg_black_border = np.mean(border_black_scores) if border_black_scores else 0.0

    print(f"[Preprocess Analysis] Avg Solidity: {avg_solidity:.3f}, Border Blackness: {avg_black_border:.3f}")

    if avg_solidity < 0.88:
        return {"type": "irregular", "mode": "irregular", "shrink": 0, "desc": "Irregular (Parrot)"}

    # PARANOID CHECK: If even 1% is black, assume rotation artifacts and shrink.
    elif avg_black_border > 0.01:
        return {"type": "rotated_rect", "mode": "rect", "shrink": 3, "desc": "Rotated (Shrink 3px)"}
    else:
        return {"type": "standard_rect", "mode": "rect", "shrink": 0, "desc": "Standard (Clean)"}


def compute_edge_features(piece, mask, border=2, inner_shift: int = 2):
    """
    为每个 piece 计算四条边的特征，使用向内偏移 inner_shift 像素的 strip，
    避免最外圈黑边问题，并在这里完成 mask 过滤。

    输出:
      EdgeFeatures:
        - mean_color: strip 内有效像素的 LAB 均值
        - color_hist: LAB 直方图 (L, a, b 各 8 bin 合并)
        - color_profile: 沿边方向的一维 LAB profile, shape = (L, 3)
        - gradient: 沿边方向 L 通道的一阶差分 |dL/ds|, shape = (L,)
    """
    h, w, _ = piece.shape
    border = min(border, max(1, h // 2 - inner_shift), max(1, w // 2 - inner_shift))

    # 转 LAB，后续所有特征都基于 LAB
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB).astype(np.float32)

    def extract_edge_strip(edge: str):
        nonlocal lab, mask, h, w, border, inner_shift

        if edge == "top":
            ys = slice(inner_shift, inner_shift + border)
            xs = slice(0, w)
        elif edge == "bottom":
            ys = slice(h - inner_shift - border, h - inner_shift)
            xs = slice(0, w)
        elif edge == "left":
            ys = slice(0, h)
            xs = slice(inner_shift, inner_shift + border)
        elif edge == "right":
            ys = slice(0, h)
            xs = slice(w - inner_shift - border, w - inner_shift)
        else:
            raise ValueError(f"Unknown edge: {edge}")

        reg_lab = lab[ys, xs]  # (strip_h, strip_w, 3)
        reg_msk = mask[ys, xs]  # (strip_h, strip_w)

        valid = reg_msk > 128
        if not np.any(valid):
            # 没有有效像素，返回全零特征
            mean_c = np.array([0., 0., 0.], dtype=np.float32)
            hist = np.zeros(24, dtype=np.float32)
            color_profile = np.zeros((reg_lab.shape[1 if edge in ("top", "bottom") else 0], 3), dtype=np.float32)
            gradient = np.zeros(color_profile.shape[0], dtype=np.float32)
            return mean_c, hist, color_profile, gradient

        # --- 1) mean_color & histogram (LAB) ---
        valid_pixels = reg_lab[valid]  # (N, 3)
        mean_c = valid_pixels.mean(axis=0)

        hist_list = []
        for ch in range(3):
            h_ = cv2.calcHist([valid_pixels[:, ch].astype(np.float32)], [0], None, [8], [0, 256])
            hist_list.append(cv2.normalize(h_, None).flatten())
        hist = np.concatenate(hist_list).astype(np.float32)  # 24-d

        # --- 2) 一维 color_profile (沿边方向) ---
        # 对于 top/bottom：沿 x 方向 (width)
        # 对于 left/right：沿 y 方向 (height)
        if edge in ("top", "bottom"):
            L_len = reg_lab.shape[1]
            color_profile = np.zeros((L_len, 3), dtype=np.float32)
            for x in range(L_len):
                col_valid = valid[:, x]
                if np.any(col_valid):
                    color_profile[x] = reg_lab[:, x, :][col_valid].mean(axis=0)
                else:
                    color_profile[x] = 0.0
            if edge == "bottom":
                color_profile = color_profile[::-1]  # 对齐方向
        else:  # left / right
            L_len = reg_lab.shape[0]
            color_profile = np.zeros((L_len, 3), dtype=np.float32)
            for y in range(L_len):
                row_valid = valid[y, :]
                if np.any(row_valid):
                    color_profile[y] = reg_lab[y, :, :][row_valid].mean(axis=0)
                else:
                    color_profile[y] = 0.0
            if edge == "left":
                color_profile = color_profile[::-1]  # 对齐方向

        # --- 3) gradient profile (一维 L 通道差分) ---
        L_profile = color_profile[:, 0]  # L 通道
        if L_profile.shape[0] > 1:
            grad = np.abs(np.diff(L_profile, prepend=L_profile[0]))
        else:
            grad = np.zeros_like(L_profile)
        gradient = grad.astype(np.float32)

        return mean_c.astype(np.float32), hist, color_profile.astype(np.float32), gradient

    features: Dict[str, EdgeFeatures] = {}
    for edge in ["top", "right", "bottom", "left"]:
        mean_c, hist, prof, grad = extract_edge_strip(edge)
        features[edge] = EdgeFeatures(
            mean_color=(float(mean_c[0]), float(mean_c[1]), float(mean_c[2])),
            color_hist=hist,
            color_profile=prof,
            gradient=grad,
        )

    return features


def compute_edge_lines_for_rotations(img: np.ndarray, mask: np.ndarray):
    """
    为每个 piece 预计算 4 个旋转角度 (0,90,180,270 CW) 下的边缘线，
    语义尽量与旧版 attemptA 一致：
      - 对每个旋转，都在最外一行/列上取 LAB 像素
      - 保留对应的 mask 一维数组
    返回:
      edge_sets: 长度为 4 的 list
        edge_sets[r]: Dict[str, (pixels_line, mask_line)]
    """
    edge_sets: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    img_curr = img.copy()
    msk_curr = mask.copy()

    for _ in range(4):
        img_lab = cv2.cvtColor(img_curr, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = img_lab.shape[:2]

        if len(msk_curr.shape) == 3:
            m = msk_curr[:, :, 0]
        else:
            m = msk_curr

        edges = {
            "top": (img_lab[0, :], m[0, :]),
            "bottom": (img_lab[h - 1, :], m[h - 1, :]),
            "left": (img_lab[:, 0], m[:, 0]),
            "right": (img_lab[:, w - 1], m[:, w - 1]),
        }
        edge_sets.append(edges)

        img_curr = cv2.rotate(img_curr, cv2.ROTATE_90_CLOCKWISE)
        msk_curr = cv2.rotate(msk_curr, cv2.ROTATE_90_CLOCKWISE)

    return edge_sets


def preprocess_puzzle_image(image_path, width=None, height=None, debug=False):
    canvas = load_canvas_rgb(image_path, width, height)
    pad = 50
    canvas = cv2.copyMakeBorder(canvas, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = canvas.shape[0] * canvas.shape[1]
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > total_area * 0.001]
    if not valid_cnts: return [], {}

    config = analyze_raw_pieces(canvas, valid_cnts)
    if debug: print(f"[Preprocess] Auto-Detected: {config['desc']}")

    pieces = []
    for i, c in enumerate(valid_cnts):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        corners = np.array(box, dtype=np.float32)
        img, msk = warp_and_process(canvas, corners, c, mode=config['mode'], shrink_px=config['shrink'])
        edges = compute_edge_features(img, msk)
        edge_lines = compute_edge_lines_for_rotations(img, msk)
        pieces.append(PuzzlePiece(i, img, msk, corners, img.shape[:2], edges, edge_lines))

    return pieces, config


def save_pieces(pieces, out_dir, save_meta=True):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for p in pieces:
        fname = f"piece_{p.id:02d}.png"
        b, g, r = cv2.split(p.image)
        cv2.imwrite(os.path.join(out_dir, fname), cv2.merge([b, g, r, p.mask]))
        item = {"id": p.id, "file": fname, "size": p.size, "edges": {}}
        for k, v in p.edges.items(): item["edges"][k] = {"mean_color": v.mean_color}
        meta.append(item)
    if save_meta:
        with open(os.path.join(out_dir, "pieces_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out_dir", default="pieces_out")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--width", type=int);
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    pieces, config = preprocess_puzzle_image(args.image, args.width, args.height, debug=args.debug)
    save_pieces(pieces, args.out_dir, save_meta=True)
    print(f"Saved {len(pieces)} pieces. Config: {config}")


if __name__ == "__main__":
    main()
