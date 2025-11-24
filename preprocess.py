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

@dataclass
class PuzzlePiece:
    id: int
    image: np.ndarray
    mask: np.ndarray
    canvas_corners: np.ndarray
    size: Tuple[int, int]
    edges: Dict[str, EdgeFeatures]

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
    w_top = np.linalg.norm(tr - tl); w_bot = np.linalg.norm(br - bl)
    max_w = int(max(w_top, w_bot))
    h_left = np.linalg.norm(bl - tl); h_right = np.linalg.norm(br - tr)
    max_h = int(max(h_left, h_right))

    dst = np.array([[0, 0], [max_w-1, 0], [max_w-1, max_h-1], [0, max_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    piece_img = cv2.warpPerspective(canvas, M, (max_w, max_h))

    mask_canvas = np.zeros(canvas.shape[:2], dtype=np.uint8)
    if mode == 'irregular':
        cv2.drawContours(mask_canvas, [contour], -1, 255, -1)
        piece_mask = cv2.warpPerspective(mask_canvas, M, (max_w, max_h))
        piece_mask = cv2.erode(piece_mask, np.ones((3,3), np.uint8), iterations=1)
    else:
        piece_mask = np.full((max_h, max_w), 255, dtype=np.uint8)

    if shrink_px > 0:
        if max_h > 2*shrink_px and max_w > 2*shrink_px:
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
            strips = [temp_img[0:2,:,:], temp_img[h-2:h,:,:], temp_img[:,0:2,:], temp_img[:,w-2:w,:]]
            total_px = 0; black_px = 0
            for s in strips:
                is_black = np.all(s < 10, axis=2) # Stricter black threshold
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

def compute_edge_features(piece, mask, border=2):
    h, w, _ = piece.shape
    border = min(border, h//2, w//2)
    slices = {
        "top": (slice(0, border), slice(0, w)), "bottom": (slice(h - border, h), slice(0, w)),
        "left": (slice(0, h), slice(0, border)), "right": (slice(0, h), slice(w - border, w))
    }
    features = {}
    for edge, (ys, xs) in slices.items():
        reg = piece[ys, xs]; msk = mask[ys, xs]
        valid = msk > 128
        mean_c = reg[valid].mean(axis=0) if np.any(valid) else np.array([0.,0.,0.])

        hist_list = []
        for ch in range(3):
            h_ = cv2.calcHist([reg], [ch], msk, [8], [0, 256])
            hist_list.append(cv2.normalize(h_, None).flatten())

        if edge == "top": prof = reg.mean(axis=0)
        elif edge == "right": prof = reg.mean(axis=1)
        elif edge == "bottom": prof = reg.mean(axis=0)[::-1]
        elif edge == "left": prof = reg.mean(axis=1)[::-1]
        else: prof = reg.mean(axis=0)

        features[edge] = EdgeFeatures(
            mean_color=(float(mean_c[0]), float(mean_c[1]), float(mean_c[2])),
            color_hist=np.concatenate(hist_list),
            color_profile=prof.astype(np.float32)
        )
    return features

def preprocess_puzzle_image(image_path, width=None, height=None, debug=False):
    canvas = load_canvas_rgb(image_path, width, height)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    k = np.ones((3,3), np.uint8)
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
        pieces.append(PuzzlePiece(i, img, msk, corners, img.shape[:2], edges))

    return pieces, config

def save_pieces(pieces, out_dir, save_meta=True):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for p in pieces:
        fname = f"piece_{p.id:02d}.png"
        b,g,r = cv2.split(p.image)
        cv2.imwrite(os.path.join(out_dir, fname), cv2.merge([b,g,r,p.mask]))
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
    parser.add_argument("--width", type=int); parser.add_argument("--height", type=int)
    args = parser.parse_args()
    pieces, config = preprocess_puzzle_image(args.image, args.width, args.height, debug=args.debug)
    save_pieces(pieces, args.out_dir, save_meta=True)
    print(f"Saved {len(pieces)} pieces. Config: {config}")

if __name__ == "__main__":
    main()