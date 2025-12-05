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
            total_px = 0
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
        return {"type": "rotated_rect", "mode": "rect", "shrink": 1, "desc": "Rotated (Shrink 1px)"}
    else:
        return {"type": "standard_rect", "mode": "rect", "shrink": 0, "desc": "Standard (Clean)"}


def dynamic_shrink(image: np.ndarray, mask: np.ndarray, max_shrink: int = 3) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    智能剥离边缘：如果边缘行/列看起来像是伪影（纯色、过暗、过亮），就切掉。
    返回：(new_image, new_mask, actual_shrink_amount)
    """
    h, w = image.shape[:2]
    current_shrink = 0

    # 也就是即使不做 shrink，我们也默认切 1px 以防万一（除非你非常有信心）
    # 如果你想极致保留，可以设 base_shrink = 0
    base_shrink = 1

    # 先做基础裁剪
    if base_shrink > 0:
        image = image[base_shrink:h - base_shrink, base_shrink:w - base_shrink]
        mask = mask[base_shrink:h - base_shrink, base_shrink:w - base_shrink]
        current_shrink += base_shrink

    # 循环检查最外圈，最多再切 max_shrink - base_shrink 次
    for _ in range(max_shrink - base_shrink):
        h, w = image.shape[:2]
        if h < 10 or w < 10: break  # 防止切没了

        # 提取四条边
        top = image[0, :, :]
        bottom = image[h - 1, :, :]
        left = image[:, 0, :]
        right = image[:, w - 1, :]

        # 拼接所有边缘像素
        border_pixels = np.concatenate([top, bottom, left, right], axis=0)

        # 计算边缘的特征
        # 1. 亮度 (Lightness)
        gray = cv2.cvtColor(border_pixels[np.newaxis, :, :], cv2.COLOR_BGR2GRAY).flatten()
        mean_val = np.mean(gray)
        std_val = np.std(gray)  # 标准差，衡量是否有纹理

        # 判断逻辑：
        # A. 几乎纯黑 (Scanner Black) -> mean < 20
        # B. 几乎纯白 (Scanner White) -> mean > 235
        # C. 几乎纯色 (Flat Artifact) -> std < 5 (没有纹理的死板线条)

        is_garbage_edge = (mean_val < 25) or (mean_val > 230) or (std_val < 8.0)

        if is_garbage_edge:
            # 这是一个坏边，切掉 1px
            image = image[1:h - 1, 1:w - 1]
            mask = mask[1:h - 1, 1:w - 1]
            current_shrink += 1
        else:
            # 边缘看起来有丰富的颜色变化（是画的一部分），停止裁剪
            break

    return image, mask, current_shrink


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


def extract_surface_edge(img, mask, side):
    """
    [终极方案] 表面投影采样。
    不固定读取某一行，而是扫描每一列，找到该方向上“第一个有效像素”。
    能完美解决 minAreaRect 矫正不彻底导致的微小倾斜（0.5度误差），
    避免读取到黑色背景或产生断层。
    """
    h, w = img.shape[:2]
    # 使用 Lab 颜色空间更符合人类感知
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # 阈值要高，确保不读到插值产生的半透明噪点
    valid_mask = mask > 200

    edge_pixels = []
    edge_mask_vals = []

    if side == 'top':
        # 从上往下扫：对于每一列 x，找第一个 valid_mask[y, x] 为 True 的点
        for x in range(w):
            col_mask = valid_mask[:, x]
            # np.argmax 在布尔数组中返回第一个 True 的索引，如果没有 True 返回 0
            y = np.argmax(col_mask)
            # 检查是否真的找到了（如果全是 False，argmax 也是 0，需检查该点是否为 True）
            if col_mask[y]:
                edge_pixels.append(img_lab[y, x])
                edge_mask_vals.append(255)  # 既然找到了，就是有效的
            else:
                edge_pixels.append([0, 0, 0])  # 没找到有效点
                edge_mask_vals.append(0)

    elif side == 'bottom':
        # 从下往上扫
        for x in range(w):
            col_mask = valid_mask[:, x]
            # 找最后一个 True。技巧：翻转数组找第一个，然后换算坐标
            rev_idx = np.argmax(col_mask[::-1])
            y = h - 1 - rev_idx
            if col_mask[y]:
                edge_pixels.append(img_lab[y, x])
                edge_mask_vals.append(255)
            else:
                edge_pixels.append([0, 0, 0])
                edge_mask_vals.append(0)

    elif side == 'left':
        # 从左往右扫
        for y in range(h):
            row_mask = valid_mask[y, :]
            x = np.argmax(row_mask)
            if row_mask[x]:
                edge_pixels.append(img_lab[y, x])
                edge_mask_vals.append(255)
            else:
                edge_pixels.append([0, 0, 0])
                edge_mask_vals.append(0)

    elif side == 'right':
        # 从右往左扫
        for y in range(h):
            row_mask = valid_mask[y, :]
            rev_idx = np.argmax(row_mask[::-1])
            x = w - 1 - rev_idx
            if row_mask[x]:
                edge_pixels.append(img_lab[y, x])
                edge_mask_vals.append(255)
            else:
                edge_pixels.append([0, 0, 0])
                edge_mask_vals.append(0)

    return np.array(edge_pixels), np.array(edge_mask_vals, dtype=np.uint8)


def compute_edge_lines_for_rotations(img: np.ndarray, mask: np.ndarray, inner_margin: int = 0):
    """
    [关键修复] 预计算旋转边缘。
    强制内缩采样 (inner_margin=1)，跳过 warp 产生的边缘混叠/黑边/锯齿。
    这是解决旋转拼图（特别是高对比度如 Cafe/Irises）准确度的关键。
    """
    edge_sets = []

    curr_img = img.copy()
    curr_msk = mask.copy()

    for _ in range(4):
        img_lab = cv2.cvtColor(curr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = img_lab.shape[:2]

        if len(curr_msk.shape) == 3:
            m = curr_msk[:, :, 0]
        else:
            m = curr_msk

        # 太小的话就不缩，防止越界
        margin = inner_margin if (h > 2 * inner_margin and w > 2 * inner_margin) else 0

        top_y = margin
        bottom_y = h - 1 - margin
        left_x = margin
        right_x = w - 1 - margin

        edges = {
            "top": (img_lab[top_y, :], m[top_y, :]),
            "bottom": (img_lab[bottom_y, :], m[bottom_y, :]),
            "left": (img_lab[:, left_x], m[:, left_x]),
            "right": (img_lab[:, right_x], m[:, right_x]),
        }
        edge_sets.append(edges)

        curr_img = cv2.rotate(curr_img, cv2.ROTATE_90_CLOCKWISE)
        curr_msk = cv2.rotate(curr_msk, cv2.ROTATE_90_CLOCKWISE)

    return edge_sets


def dynamic_shrink(image: np.ndarray, mask: np.ndarray, base_shrink: int = 1, max_shrink: int = 3) -> Tuple[
    np.ndarray, np.ndarray, int]:
    """
    智能动态去边 v1.1
    增加 base_shrink 参数，允许对直图设置为 0。
    """
    h, w = image.shape[:2]
    current_shrink = 0

    # [Step 1] 基础裁剪 (Base Crop)
    # 对于旋转图，base_shrink 通常为 1；对于直图，应为 0。
    if base_shrink > 0:
        if h > 2 * base_shrink and w > 2 * base_shrink:
            image = image[base_shrink:h - base_shrink, base_shrink:w - base_shrink]
            mask = mask[base_shrink:h - base_shrink, base_shrink:w - base_shrink]
            current_shrink += base_shrink

    # [Step 2] 循环剥离 (Peeling Onion)
    # 即使 base_shrink=0，我们也检查一下是否有明显的黑边/白边（针对扫描不完美的直图）
    # 但我们把 max_shrink 限制住，防止切太多
    remaining_steps = max_shrink - current_shrink

    for _ in range(remaining_steps):
        h, w = image.shape[:2]
        if h < 10 or w < 10: break

        # 提取边缘
        top = image[0, :, :]
        bottom = image[h - 1, :, :]
        left = image[:, 0, :]
        right = image[:, w - 1, :]
        border_pixels = np.concatenate([top, bottom, left, right], axis=0)

        gray = cv2.cvtColor(border_pixels[np.newaxis, :, :], cv2.COLOR_BGR2GRAY).flatten()
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # 判据：只有当边缘极其“假”的时候才切
        # 1. 极暗且无纹理 (Scanner Black)
        is_dark_artifact = (mean_val < 30) and (std_val < 10.0)
        # 2. 极亮 (Paper White)
        is_light_artifact = (mean_val > 235)
        # 3. 极度平滑的线条 (Artificial Border)
        is_flat_line = (std_val < 3.0)

        if is_dark_artifact or is_light_artifact or is_flat_line:
            image = image[1:h - 1, 1:w - 1]
            mask = mask[1:h - 1, 1:w - 1]
            current_shrink += 1
        else:
            break

    return image, mask, current_shrink


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

    # 1. 自动检测类型
    config = analyze_raw_pieces(canvas, valid_cnts)

    # 2. 根据类型决定策略
    # 如果是直图 (standard_rect)，绝对不要预先切边 (base_shrink=0)
    # 如果是旋转图 (rotated_rect)，为了抗锯齿，预先切 1px (base_shrink=1)
    is_rotated = (config['type'] == 'rotated_rect')
    target_base_shrink = 1 if is_rotated else 0

    if debug:
        print(f"[Preprocess] Mode: {config['type']}. Base Shrink: {target_base_shrink}")

    pieces = []
    for i, c in enumerate(valid_cnts):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        corners = np.array(box, dtype=np.float32)

        # 步骤A: 获取原始切片 (shrink_px=0, 交给 dynamic_shrink 处理)
        img, msk = warp_and_process(canvas, corners, c, mode=config['mode'], shrink_px=0)

        # 步骤B: 智能去边
        # 传入我们根据 config 决定的 base_shrink
        img, msk, applied_shrink = dynamic_shrink(img, msk, base_shrink=target_base_shrink, max_shrink=3)

        if img.shape[0] < 5 or img.shape[1] < 5:
            continue

        # 步骤C: 特征计算
        edges = compute_edge_features(img, msk)

        # 步骤D: 旋转边缘计算
        # 注意：如果是直图，inner_margin 保持 0 即可，因为我们信任切边是完美的
        # 如果是旋转图，虽然我们切了边，但为了保险起见，可以在旋转计算时稍微再缩一点点吗？
        # 这里建议统一为 0，因为 dynamic_shrink 已经处理过了。
        edge_lines = compute_edge_lines_for_rotations(img, msk, inner_margin=0)

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



import numpy as np

def is_regular_grid(pieces, tol_ratio: float = 0.05, min_piece_size: int = 5) -> bool:
    """
    判断当前 puzzle 是否为「规则 grid」：
    - 所有拼图块在「忽略旋转」的情况下，长宽基本一致。
    - 忽略旋转的做法：对每个 piece 取 (h, w)，然后 canonical = (min(h, w), max(h, w))。
    - 允许一定比例的浮动（tol_ratio），用于容忍 shrink / warp 误差。

    参数:
      pieces: preprocess_puzzle_image 返回的 PuzzlePiece 列表
      tol_ratio: 尺寸允许的最大相对偏差 (max(|x - mean| / mean))
      min_piece_size: 过滤太小的残片（防止极端噪点影响统计）

    返回:
      True  -> 认为是 regular grid（规则矩形碎片，允许旋转后 w/h 互换）
      False -> 认为是 irregular grid（尺寸差异较大）
    """
    sizes = []

    for p in pieces:
        # PuzzlePiece.size 在 preprocess 里是 img.shape[:2]
        if hasattr(p, "size") and p.size is not None:
            h, w = p.size
        else:
            h, w = p.image.shape[:2]

        # 跳过异常小的碎片
        if h < min_piece_size or w < min_piece_size:
            continue

        # 规一化到「忽略旋转」的尺寸
        a, b = sorted((h, w))  # a <= b
        sizes.append((a, b))

    # 如果有效块数量很少，默认当 regular（交给后续 solver 处理）
    if len(sizes) <= 1:
        return True

    sizes_arr = np.array(sizes, dtype=np.float32)  # (N, 2)
    mins = sizes_arr[:, 0]
    maxs = sizes_arr[:, 1]

    mean_min = float(np.mean(mins))
    mean_max = float(np.mean(maxs))

    # 避免除零
    if mean_min <= 0 or mean_max <= 0:
        return False

    # 计算相对偏差
    dev_min = np.abs(mins - mean_min) / mean_min
    dev_max = np.abs(maxs - mean_max) / mean_max

    max_dev_min = float(np.max(dev_min))
    max_dev_max = float(np.max(dev_max))

    # 只要任一方向的最大相对偏差超过 tol_ratio，就认为不是规则 grid
    is_regular = (max_dev_min <= tol_ratio) and (max_dev_max <= tol_ratio)

    print(f"[GridCheck] mean_size=({mean_min:.1f}, {mean_max:.1f}), "
          f"max_dev=({max_dev_min:.3f}, {max_dev_max:.3f}), "
          f"regular={is_regular}")

    return is_regular



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
