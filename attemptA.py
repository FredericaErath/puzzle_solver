"""
attemptA.py (matching.py v27.1 - Fix Attribute Error)

Fixes:
1. AttributeError: Renamed PieceInstance fields from 'g_rows/g_cols' to 'grid_rows/grid_cols'
   to match the logic used in the solver.
"""

from __future__ import annotations

import argparse
import json
import time
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import cv2
import os
import numpy as np
from preprocess import PuzzlePiece, preprocess_puzzle_image, EdgeFeatures

# ---- Hyperparameters ----
T_SEED_COMPLEXITY = 0.05  # 过滤掉过度平滑的边（归一化后 0~1）
SEED_BIAS = 1.0  # seed 专用的 complexity bias
COMPLEXITY_BIAS = 1.0

LOOKAHEAD_K = 5
LOOKAHEAD_DEPTH = 3

W_COLOR = 1.0
W_GRAD = 0.1


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class Placement:
    piece_id: int
    rotation: int
    y: int
    x: int
    h: int
    w: int


@dataclass
class PieceInstance:
    pid: int
    rot: int
    h: int
    w: int
    grid_rows: int
    grid_cols: int
    edges: Dict[str, Tuple[np.ndarray, np.ndarray]]
    edge_complexity: Dict[str, float]


@dataclass(order=True)
class MoveCandidate:
    cost: float
    pid: int = field(compare=False)
    rot: int = field(compare=False)
    row: int = field(compare=False)  # Grid row
    col: int = field(compare=False)  # Grid col


class PriorityFrontierSolver:
    def __init__(self, pieces, W, H, weights, config, debug, image_debug=False, debug_dir=None):
        self.pieces = pieces
        self.num_pieces = len(pieces)
        self.W = W
        self.H = H
        self.debug = debug
        self.weights = weights
        self.image_debug = image_debug
        self.debug_dir = debug_dir
        self.blur_edges = config.get('blur', False)

        # 1. Calculate Atomic Unit Size
        ws = [p.size[1] for p in pieces]
        hs = [p.size[0] for p in pieces]
        self.unit_w = int(min(ws))
        self.unit_h = int(min(hs))

        # ---- Grid 长边方向 & 每个 piece 合法 rotation ----
        eps = 1e-3
        # 是否真的是长方形 grid（而不是近似正方形）
        self.rect_grid = abs(self.unit_h - self.unit_w) > eps
        # True 表示格子竖着更长（height > width）
        self.grid_vertical = self.unit_h > self.unit_w

        # 对每个 piece 预计算允许使用的 rotations
        self.allowed_rots: Dict[int, List[int]] = {}
        for idx, p in enumerate(self.pieces):
            base_h, base_w = p.size  # (rows, cols)
            allowed: List[int] = []
            for r in range(4):
                # 和 _precompute_variants 一致的几何尺寸逻辑
                if r % 2 == 0:
                    h_rot, w_rot = base_h, base_w
                else:
                    h_rot, w_rot = base_w, base_h

                # grid 近似正方形：不做限制，四个 rotation 都可以
                if not self.rect_grid:
                    allowed.append(r)
                    continue

                # piece 近似正方形：同样无所谓方向，保留
                if abs(h_rot - w_rot) <= eps:
                    allowed.append(r)
                    continue

                if self.grid_vertical:
                    # 竖长格子：只留“竖长”的变体
                    if h_rot >= w_rot:
                        allowed.append(r)
                else:
                    # 横长格子：只留“横长”的变体
                    if w_rot >= h_rot:
                        allowed.append(r)

            # 保险：万一没留下任何 rotation，就 fallback 回四个都允许
            if not allowed:
                allowed = [0, 1, 2, 3]

            self.allowed_rots[idx] = allowed

        # 2. Calculate Target Grid Layout
        self.max_cols = int(round(W / self.unit_w))
        self.max_rows = int(round(H / self.unit_h))

        if self.max_rows * self.max_cols < self.num_pieces:
            side = int(np.ceil(np.sqrt(self.num_pieces)))
            self.max_cols = side
            self.max_rows = side

        if self.debug:
            print(f"[Grid] Unit: {self.unit_w}x{self.unit_h} | Layout Limit: {self.max_rows}x{self.max_cols}")

        self.variants = self._precompute_variants()

        # 统计全局边缘复杂度，用于归一化
        all_complexities = []
        for var_list in self.variants:
            for inst in var_list:
                all_complexities.extend(inst.edge_complexity.values())
        if all_complexities:
            self.edge_complexity_min = float(min(all_complexities))
            self.edge_complexity_max = float(max(all_complexities))
        else:
            self.edge_complexity_min = 0.0
            self.edge_complexity_max = 0.0

        # 复杂边 bias 权重，值越大越偏好复杂边参与匹配
        self.complexity_bias = self.blur_edges and COMPLEXITY_BIAS

        self.grid: Dict[Tuple[int, int], PieceInstance] = {}
        self.used_pids: Set[int] = set()
        self.min_r, self.max_r = 0, 0
        self.min_c, self.max_c = 0, 0

    # ---------------- Edge Feature Rotation Helpers -----------------

    def _flip_edge_features(self, ef: EdgeFeatures) -> EdgeFeatures:
        """
        翻转边缘方向（profile 和 gradient 反向），
        mean_color / hist 不变。
        """
        return EdgeFeatures(
            mean_color=ef.mean_color,
            color_hist=ef.color_hist,
            color_profile=ef.color_profile[::-1].copy(),
            gradient=ef.gradient[::-1].copy(),
        )

    def _rotate_edges(self, base_edges: Dict[str, EdgeFeatures], rot: int) -> Dict[str, EdgeFeatures]:
        """
        根据旋转角度（0/90/180/270 CW）对四条边的 feature 做映射和翻转。
        约定：color_profile 的正方向始终是“从左到右”或“从上到下”与
        拼接逻辑一致。
        """
        if rot % 4 == 0:
            # 0°
            return {
                "top": base_edges["top"],
                "right": base_edges["right"],
                "bottom": base_edges["bottom"],
                "left": base_edges["left"],
            }
        elif rot % 4 == 1:
            # 90° CW
            return {
                "top": self._flip_edge_features(base_edges["left"]),
                "right": base_edges["top"],
                "bottom": self._flip_edge_features(base_edges["right"]),
                "left": base_edges["bottom"],
            }
        elif rot % 4 == 2:
            # 180°
            return {
                "top": self._flip_edge_features(base_edges["bottom"]),
                "right": self._flip_edge_features(base_edges["left"]),
                "bottom": self._flip_edge_features(base_edges["top"]),
                "left": self._flip_edge_features(base_edges["right"]),
            }
        else:
            # 270° CW (90° CCW)
            return {
                "top": base_edges["right"],
                "right": self._flip_edge_features(base_edges["bottom"]),
                "bottom": base_edges["left"],
                "left": self._flip_edge_features(base_edges["top"]),
            }

    def _edge_complexity_from_line(self, pixels: np.ndarray, mask: np.ndarray) -> float:
        """
        边缘复杂度度量（方案一）：
          local_texture = L/a/b 三通道方差 + L 通道梯度能量
          contrast      = L 通道局部对比度（p95 - p5）
          complexity    = local_texture * (1 + γ * contrast_norm)

        直觉：
          - 颜色/亮度变化越丰富 → 方差大
          - 有纹理/细节 → 梯度能量大
          - 有清晰的明暗跨度 → contrast 大
        """
        if pixels is None or mask is None:
            return 0.0

        L = min(len(pixels), len(mask))
        if L < 3:
            return 0.0

        pixels = pixels[:L]
        mask = mask[:L]

        valid = mask > 128
        if not np.any(valid):
            return 0.0

        vals = pixels[valid]  # (N, 3) in LAB
        if vals.shape[0] < 3:
            return 0.0

        # --- 多通道方差 ---
        vals_f = vals.astype(np.float32)
        Lch = vals_f[:, 0]
        ach = vals_f[:, 1]
        bch = vals_f[:, 2]

        var_L = float(np.var(Lch))
        var_a = float(np.var(ach))
        var_b = float(np.var(bch))

        # --- L 通道梯度能量 ---
        grad_energy = 0.0
        if Lch.shape[0] > 2:
            grad = np.diff(Lch)
            grad_energy = float(np.mean(grad ** 2))

        # --- 局部对比度（L 的 95%-5% 分位差）---
        p5 = float(np.percentile(Lch, 5))
        p95 = float(np.percentile(Lch, 95))
        contrast = max(0.0, p95 - p5)

        # 按 0–255 缩放；如果你的 LAB L 不是 0–255，可以把 255 换成 100
        contrast_norm = contrast / 255.0
        contrast_norm = min(contrast_norm, 1.0)

        # --- 综合复杂度 ---
        # 可调权重：梯度和对比度的影响力
        LAMBDA_GRAD = 1.0
        GAMMA_CONTRAST = 1.0

        local_texture = (var_L + var_a + var_b) + LAMBDA_GRAD * grad_energy
        if local_texture < 1e-6:
            return 0.0

        complexity = local_texture * (1.0 + GAMMA_CONTRAST * contrast_norm)
        return float(complexity)

    def _precompute_variants(self):
        all_vars = []
        for p in self.pieces:
            p_vars = []

            base_h, base_w = p.size  # (rows, cols)

            for r in range(4):
                # 旋转后的几何尺寸：与旧版相同
                if r % 2 == 0:
                    h, w = base_h, base_w
                else:
                    h, w = base_w, base_h

                gr = max(1, int(round(h / self.unit_h)))
                gc = max(1, int(round(w / self.unit_w)))

                edges = p.edge_lines[r]  # { 'top': (pixels, mask), ... }

                edge_complexity: Dict[str, float] = {}
                for side, (px, msk) in edges.items():
                    edge_complexity[side] = self._edge_complexity_from_line(px, msk)

                p_vars.append(PieceInstance(
                    pid=p.id,
                    rot=r,
                    h=h,
                    w=w,
                    grid_rows=gr,
                    grid_cols=gc,
                    edges=edges,
                    edge_complexity=edge_complexity,
                ))

            all_vars.append(p_vars)
        return all_vars

    def _aligned_color_mse(self, pa: np.ndarray, pb: np.ndarray, max_shift: int = 3) -> Tuple[
        float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Level 1 + 2:
          - 在 [-max_shift, max_shift] 像素范围内做滑动对齐
          - 在对齐后的子段上，用 Gaussian 模糊后的 LAB 计算 MSE
        返回:
          best_mse, best_subA, best_subB
          若没有足够长度，返回 (1e12, None, None)
        """
        best_mse = 1e12
        best_A = None
        best_B = None

        L = min(len(pa), len(pb))
        if L < 3:
            return best_mse, None, None

        for s in range(-max_shift, max_shift + 1):
            if s < 0:
                # A 往右对齐 B
                a = pa[-s:L]
                b = pb[:L + s]
            elif s > 0:
                # B 往右对齐 A
                a = pa[:L - s]
                b = pb[s:L]
            else:
                a = pa[:L]
                b = pb[:L]

            if len(a) < 3:
                continue

            # Level 2：对对齐后的子段做 1D Gaussian 平滑，强调低频
            # 形状 (len, 3) → (1, len, 3)，再还原回来
            a_blur = cv2.GaussianBlur(a.reshape(1, -1, 3), (1, 5), 0).reshape(-1, 3)
            b_blur = cv2.GaussianBlur(b.reshape(1, -1, 3), (1, 5), 0).reshape(-1, 3)

            diff = a_blur - b_blur
            mse = float(np.mean(np.sum(diff ** 2, axis=1)))

            if mse < best_mse:
                best_mse = mse
                best_A = a_blur
                best_B = b_blur

        return best_mse, best_A, best_B

    def _norm_edge_complexity(self, inst: PieceInstance, side: str) -> float:
        c = float(inst.edge_complexity.get(side, 0.0))
        if self.edge_complexity_max <= self.edge_complexity_min + 1e-6:
            return 0.0
        return (c - self.edge_complexity_min) / (self.edge_complexity_max - self.edge_complexity_min + 1e-6)

    def _apply_complexity_bias(
            self,
            raw_cost: float,
            inst_a: PieceInstance,
            side_a: str,
            inst_b: PieceInstance,
            side_b: str,
    ) -> float:
        """
        根据两条边的复杂度调整 cost：
          - 复杂度越高 → 有更多纹理特征 → 更值得优先匹配
          - 策略：raw_cost / (1 + complexity_bias * max(norm_c_a, norm_c_b))
        """
        if raw_cost >= 500000:
            return raw_cost

        ca = self._norm_edge_complexity(inst_a, side_a)
        cb = self._norm_edge_complexity(inst_b, side_b)
        pair_c = max(ca, cb)

        return raw_cost / (1.0 + self.complexity_bias * pair_c)

    def _compute_edge_diff(self, edge_a, edge_b) -> float:
        """
        Level 1: 允许沿边缘方向 ±2 像素的滑动对齐，缓解 warp / shrink 带来的 pixel shift。
        Level 2: 在对齐后的子段上，对 LAB profile 做 1D Gaussian 平滑，突出低频结构。
        Level 3: 在最优对齐下，加入 L 通道梯度方向相似度 (cosine) 作为辅助项。

        仍保留：
          - mask 过滤
          - near-black (L<5) 剔除
          - 极少有效像素时返回很大 cost
        """
        pixels_a, mask_a = edge_a
        pixels_b, mask_b = edge_b

        # 统一长度
        L = min(len(mask_a), len(mask_b))
        if L < 3:
            return 999999.0

        pixels_a = pixels_a[:L]
        pixels_b = pixels_b[:L]
        mask_a = mask_a[:L]
        mask_b = mask_b[:L]

        # 1) 有效 mask
        valid = (mask_a > 250) & (mask_b > 250)

        # 2) near-black 过滤 (LAB 中 L 通道在索引 0)
        nb_a = pixels_a[:, 0] > 5
        nb_b = pixels_b[:, 0] > 5
        valid = valid & nb_a & nb_b

        if np.sum(valid) < 5:
            return 999999.0

        pa = pixels_a[valid]  # (N,3)
        pb = pixels_b[valid]

        # 若配置了 blur_edges，可以再做一个细微的 2D blur（与原逻辑兼容）
        if self.blur_edges and len(pa) >= 3:
            pa = cv2.GaussianBlur(pa.reshape(-1, 1, 3), (1, 1), 0).reshape(-1, 3)
            pb = cv2.GaussianBlur(pb.reshape(-1, 1, 3), (1, 1), 0).reshape(-1, 3)

        # ---------- Level 1 + 2: 对齐 + 低频 MSE ----------
        color_mse, best_A, best_B = self._aligned_color_mse(pa, pb, max_shift=2)
        if best_A is None or best_B is None:
            return 999999.0

        # ---------- Level 3: 梯度方向相似度 ----------
        # 使用 L 通道的一阶差分，计算 cosine 距离
        LA = best_A[:, 0]
        LB = best_B[:, 0]

        if len(LA) > 3 and len(LB) > 3:
            gA = np.diff(LA)
            gB = np.diff(LB)

            # 避免全 0 梯度
            normA = float(np.linalg.norm(gA)) + 1e-6
            normB = float(np.linalg.norm(gB)) + 1e-6
            cos_sim = float(np.dot(gA, gB) / (normA * normB))
            # cosine 距离: 1 - cos_sim, 范围大致 [0,2]
            grad_cost = 1.0 - cos_sim
        else:
            grad_cost = 0.0

        # ---------- 综合 cost ----------
        # color_mse 是平方误差，grad_cost 是无量纲方向差，给梯度一个较小权重
        w_color = self.weights.get('w_color', W_COLOR)
        w_grad = self.weights.get('w_grad', W_GRAD)

        combined = w_color * color_mse + w_grad * grad_cost

        # 保持和原逻辑一致：返回 sqrt(MSE-like)
        return float(np.sqrt(combined + 1e-8))

    def _find_best_seed(self):
        """
        改进 seed 选择：
        1) 过滤掉过于平滑/背景的边（avoid low-complexity seed）
        2) seed 阶段加更强权重，使复杂边优先被选中
        3) 禁止两条低复杂度边互拼（避免 background-background seed）
        """
        best_score = float('inf')
        best_seed = None

        T_seed_complexity = T_SEED_COMPLEXITY  # 归一化后 0~1
        seed_bias = SEED_BIAS  # seed 专用 complexity bias

        for i in range(self.num_pieces):
            for ri in self.allowed_rots[i]:
                v1 = self.variants[i][ri]

                for j in range(self.num_pieces):
                    if i == j:
                        continue

                    for rj in self.allowed_rots[j]:
                        v2 = self.variants[j][rj]

                        # ---------- RIGHT match ----------
                        c1 = self._norm_edge_complexity(v1, 'right')
                        c2 = self._norm_edge_complexity(v2, 'left')

                        # 两边都太平滑 -> 不拿来做 seed 的右边匹配
                        if not (c1 < T_seed_complexity and c2 < T_seed_complexity):
                            raw = self._compute_edge_diff(v1.edges['right'], v2.edges['left'])
                            seed_score = raw / (1.0 + seed_bias * max(c1, c2))
                            if seed_score < best_score:
                                best_score = seed_score
                                best_seed = (v1, v2, 0, 0, 0, v1.grid_cols)

                        # ---------- BOTTOM match ----------
                        c1b = self._norm_edge_complexity(v1, 'bottom')
                        c2b = self._norm_edge_complexity(v2, 'top')

                        # 两边都太平滑 -> 直接跳过这个 bottom seed
                        if c1b < T_seed_complexity and c2b < T_seed_complexity:
                            continue

                        raw = self._compute_edge_diff(v1.edges['bottom'], v2.edges['top'])
                        seed_score = raw / (1.0 + seed_bias * max(c1b, c2b))
                        if seed_score < best_score:
                            best_score = seed_score
                            best_seed = (v1, v2, 0, 0, v1.grid_rows, 0)

        return best_seed, best_score

    def _is_valid_geometry(self, r, c, inst):
        # 1. Overlap Check
        for dr in range(inst.grid_rows):
            for dc in range(inst.grid_cols):
                if (r + dr, c + dc) in self.grid: return False

        # 2. Bounds Constraint
        new_min_r = min(self.min_r, r)
        new_max_r = max(self.max_r, r + inst.grid_rows - 1)
        new_min_c = min(self.min_c, c)
        new_max_c = max(self.max_c, c + inst.grid_cols - 1)

        h_span = new_max_r - new_min_r + 1
        w_span = new_max_c - new_min_c + 1

        if h_span > self.max_rows: return False
        if w_span > self.max_cols: return False

        return True

    def _evaluate_neighbors(self, neighbors, cand_inst):
        """
        计算候选拼图与邻居的匹配代价。
        改进点：使用 MAX(cost) 而不是 AVG(cost)。
        只要有一条边匹配得很差，整个方案就应该被否决。
        """
        max_cost = 0.0
        count = 0
        possible = True

        for direction, neighbor in neighbors:
            # 1. 计算原始差异
            if direction == 'top':
                raw = self._compute_edge_diff(neighbor.edges['bottom'], cand_inst.edges['top'])
                bias_side_n, bias_side_c = 'bottom', 'top'
            elif direction == 'bottom':
                raw = self._compute_edge_diff(neighbor.edges['top'], cand_inst.edges['bottom'])
                bias_side_n, bias_side_c = 'top', 'bottom'
            elif direction == 'left':
                raw = self._compute_edge_diff(neighbor.edges['right'], cand_inst.edges['left'])
                bias_side_n, bias_side_c = 'right', 'left'
            else:  # right
                raw = self._compute_edge_diff(neighbor.edges['left'], cand_inst.edges['right'])
                bias_side_n, bias_side_c = 'left', 'right'

            # 2. 严格过滤：如果原始差异过大，直接判死刑
            if raw > 500000:
                possible = False
                break

            # 3. 应用复杂度偏置 (Complexity Bias)
            cost = self._apply_complexity_bias(raw, neighbor, bias_side_n, cand_inst, bias_side_c)

            # 4. 记录最大代价
            if cost > max_cost:
                max_cost = cost
            count += 1

        # 如果没有邻居，代价为0
        score = max_cost if count > 0 else 0.0
        return score, possible

    def _simulate_greedy_steps(self, max_depth: int) -> float:
        total_cost = 0.0
        steps = 0

        for _ in range(max_depth):
            frontier_slots = self._get_frontier_slots()
            if not frontier_slots:
                break

            heap = []

            for (r, c) in frontier_slots:
                neighbors = []
                # 这里的逻辑保持不变
                if (r - 1, c) in self.grid: neighbors.append(('top', self.grid[(r - 1, c)]))
                if (r + 1, c) in self.grid: neighbors.append(('bottom', self.grid[(r + 1, c)]))
                if (r, c - 1) in self.grid: neighbors.append(('left', self.grid[(r, c - 1)]))
                if (r, c + 1) in self.grid: neighbors.append(('right', self.grid[(r, c + 1)]))

                if not neighbors:
                    continue

                for pid in range(self.num_pieces):
                    if pid in self.used_pids:
                        continue

                    for rot in self.allowed_rots[pid]:
                        cand = self.variants[pid][rot]
                        if not self._is_valid_geometry(r, c, cand):
                            continue

                        total_cost_local = 0.0
                        count = 0
                        possible = True

                        # 1. 计算常规边缘匹配 Cost (保持原样)
                        for direction, neighbor in neighbors:
                            if direction == 'top':
                                cost = self._compute_edge_diff(neighbor.edges['bottom'], cand.edges['top'])
                            elif direction == 'bottom':
                                cost = self._compute_edge_diff(neighbor.edges['top'], cand.edges['bottom'])
                            elif direction == 'left':
                                cost = self._compute_edge_diff(neighbor.edges['right'], cand.edges['left'])
                            else:  # 'right'
                                cost = self._compute_edge_diff(neighbor.edges['left'], cand.edges['right'])

                            # 这里的阈值如果太宽，会放入垃圾候选，建议稍微收紧，比如 200000
                            if cost > 500000:
                                possible = False
                                break
                            total_cost_local += cost
                            count += 1

                        # 如果边缘匹配已经失败，就没必要算角点了
                        if not possible:
                            continue

                        # ================= [NEW START] 2x2 闭环约束核心逻辑 =================
                        # 只有当 count > 0 (确实有邻居) 时才计算
                        if count > 0:
                            # 1. 计算角点误差
                            corner_diff = self._get_corner_diff(r, c, cand)

                            # 2. 定义惩罚参数
                            # LOOP_THRESHOLD: 超过这个值(LAB距离)，说明颜色肉眼可见不同
                            LOOP_THRESHOLD = 40.0
                            # LOOP_WEIGHT: 闭环约束的权重应该非常高，因为它比边缘更严格
                            LOOP_WEIGHT = 20.0

                            # 3. 施加惩罚
                            if corner_diff > LOOP_THRESHOLD:
                                # 严重不匹配：颜色完全不一样
                                # 标记为不可能，或者加一个巨大的惩罚值
                                possible = False
                            else:
                                # 线性惩罚：差异越大，Cost 越高
                                # 这里乘以 count 是为了让它跟 edge_cost 的量级平衡（因为 edge_cost 是累加的）
                                total_cost_local += (corner_diff * LOOP_WEIGHT) * count
                        # ================= [NEW END] =======================================

                        if possible and count > 0:
                            avg_cost = total_cost_local / count
                            heapq.heappush(heap, MoveCandidate(avg_cost, pid, rot, r, c))

            if not heap:
                break

            best = heapq.heappop(heap)
            # 后面的逻辑保持不变...
            inst = self.variants[best.pid][best.rot]
            for i in range(inst.grid_rows):
                for j in range(inst.grid_cols):
                    self.grid[(best.row + i, best.col + j)] = inst

            self.used_pids.add(best.pid)
            # ... 更新 min/max r/c ...
            self.min_r = min(self.min_r, best.row)
            self.max_r = max(self.max_r, best.row + inst.grid_rows - 1)
            self.min_c = min(self.min_c, best.col)
            self.max_c = max(self.max_c, best.col + inst.grid_cols - 1)

            total_cost += best.cost
            steps += 1

        if steps == 0:
            return 1e9
        return total_cost

    def _evaluate_candidate_with_lookahead(self, cand: MoveCandidate, depth: int) -> float:
        """
        对单个候选 move 进行评分：
          1. 备份当前 solver 状态
          2. 暂时放置这个 candidate
          3. 再向前贪心模拟 depth-1 步，得到额外代价
          4. 恢复原始状态
        返回：归一化后的总评分（越小越好）
        """
        # 备份当前状态
        orig_grid = self.grid.copy()
        orig_used = self.used_pids.copy()
        orig_min_r, orig_max_r = self.min_r, self.max_r
        orig_min_c, orig_max_c = self.min_c, self.max_c

        # 应用当前 candidate
        inst = self.variants[cand.pid][cand.rot]
        for i in range(inst.grid_rows):
            for j in range(inst.grid_cols):
                self.grid[(cand.row + i, cand.col + j)] = inst

        self.used_pids.add(cand.pid)
        self.min_r = min(self.min_r, cand.row)
        self.max_r = max(self.max_r, cand.row + inst.grid_rows - 1)
        self.min_c = min(self.min_c, cand.col)
        self.max_c = max(self.max_c, cand.col + inst.grid_cols - 1)

        total_cost = cand.cost
        steps = 1

        # 向前模拟 depth-1 步
        if depth > 1:
            extra_cost = self._simulate_greedy_steps(depth - 1)
            if extra_cost < 1e8:
                total_cost += extra_cost
                steps += (depth - 1)

        score = total_cost / steps

        # 恢复原始状态
        self.grid = orig_grid
        self.used_pids = orig_used
        self.min_r, self.max_r = orig_min_r, orig_max_r
        self.min_c, self.max_c = orig_min_c, orig_max_c

        return score

    def _get_frontier_slots(self):
        frontier = set()
        for (r, c), inst in self.grid.items():
            # Top
            for k in range(inst.grid_cols):
                if (r - 1, c + k) not in self.grid: frontier.add((r - 1, c + k))
            # Bottom
            for k in range(inst.grid_cols):
                if (r + inst.grid_rows, c + k) not in self.grid: frontier.add((r + inst.grid_rows, c + k))
            # Left
            for k in range(inst.grid_rows):
                if (r + k, c - 1) not in self.grid: frontier.add((r + k, c - 1))
            # Right
            for k in range(inst.grid_rows):
                if (r + k, c + inst.grid_cols) not in self.grid: frontier.add((r + k, c + inst.grid_cols))
        return list(frontier)

    def _save_step_image(self, step_num: int):
        """
        保存当前 grid 状态的图像到 debug_dir，文件名为 step_000{step_num}.png
        """
        if not self.debug_dir:
            return

        # 构建 canvas
        max_y = 0
        max_x = 0
        for (r, c), inst in self.grid.items():
            max_y = max(max_y, (r + inst.grid_rows) * self.unit_h)
            max_x = max(max_x, (c + inst.grid_cols) * self.unit_w)

        final_H = max(self.H, max_y + 10)
        final_W = max(self.W, max_x + 10)

        canvas = np.zeros((final_H, final_W, 4), dtype=np.uint8)

        # 使用 _generate_final_placements 得到的 placement 信息来绘制
        processed = set()
        min_r_global = min([k[0] for k in self.grid.keys()]) if self.grid else 0
        min_c_global = min([k[1] for k in self.grid.keys()]) if self.grid else 0

        for (r, c), inst in self.grid.items():
            if inst.pid in processed:
                continue

            # 检查是否为该 piece 的原点（左上角）
            is_origin = True
            if (r - 1, c) in self.grid and self.grid[(r - 1, c)] is inst:
                is_origin = False
            if (r, c - 1) in self.grid and self.grid[(r, c - 1)] is inst:
                is_origin = False

            if is_origin:
                piece = self.pieces[inst.pid]
                img = piece.image
                msk = piece.mask if len(piece.mask.shape) == 2 else piece.mask[:, :, 0]

                # 应用旋转
                for _ in range(inst.rot):
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

                # 计算在 canvas 上的位置
                y = (r - min_r_global) * self.unit_h
                x = (c - min_c_global) * self.unit_w
                h, w = img.shape[:2]
                h_eff = min(h, final_H - y)
                w_eff = min(w, final_W - x)

                if h_eff > 0 and w_eff > 0:
                    b, g, r_ch = cv2.split(img[:h_eff, :w_eff])
                    patch = cv2.merge([b, g, r_ch, msk[:h_eff, :w_eff]])
                    roi = canvas[y:y + h_eff, x:x + w_eff]
                    valid = msk[:h_eff, :w_eff] > 128
                    roi[valid] = patch[valid]
                    canvas[y:y + h_eff, x:x + w_eff] = roi

                processed.add(inst.pid)

        # 保存图像
        os.makedirs(self.debug_dir, exist_ok=True)
        filename = os.path.join(self.debug_dir, f"step_{step_num:04d}.png")
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(filename, canvas_bgr)
        if self.debug:
            print(f"[Debug Step] Saved {filename}")

    def _generate_final_placements(self):
        placements = []
        processed = set()

        min_r_global = min(k[0] for k in self.grid.keys())
        min_c_global = min(k[1] for k in self.grid.keys())

        for (r, c), inst in self.grid.items():
            if inst.pid in processed: continue

            # Check if this is the top-left origin of the piece in the grid
            is_origin = True
            # If piece spans multiple rows/cols, we only want to add it once at its origin
            # Since we stored the same instance object, we can verify coordinates
            # r, c is the current cell.
            # We check if (r-1, c) contains the SAME instance.
            if (r - 1, c) in self.grid and self.grid[(r - 1, c)] is inst: is_origin = False
            if (r, c - 1) in self.grid and self.grid[(r, c - 1)] is inst: is_origin = False

            if is_origin:
                placements.append(Placement(
                    inst.pid, inst.rot,
                    (r - min_r_global) * self.unit_h,
                    (c - min_c_global) * self.unit_w,
                    inst.h, inst.w
                ))
                processed.add(inst.pid)
        return placements

    def _get_corner_diff(self, r: int, c: int, cand_inst: PieceInstance) -> float:
        """
        计算 2x2 闭环中心的角点一致性误差。
        检查位置 (r,c) 的 cand_inst 与 (r-1,c), (r,c-1), (r-1,c-1) 的中心交点颜色差异。
        返回: 平均颜色距离 (LAB空间)。如果无法构成闭环，返回 0.0。
        """
        # 1. 检查是否存在 2x2 闭环结构
        # 我们正在填 (r, c)，需要检查 TL, T, L 是否都存在
        if not ((r - 1, c) in self.grid and
                (r, c - 1) in self.grid and
                (r - 1, c - 1) in self.grid):
            return 0.0

        # 2. 获取四个邻居实例
        tl = self.grid[(r - 1, c - 1)]  # Top-Left
        t = self.grid[(r - 1, c)]  # Top
        l = self.grid[(r, c - 1)]  # Left
        x = cand_inst  # Current (Candidate)

        # 3. 提取交汇点的像素 (LAB颜色)
        # 注意：这里假设 edges 存储顺序是 Top/Bottom: Left->Right, Left/Right: Top->Bottom
        # 我们可以从 raw pixels 中提取。
        # 必须检查 mask 是否有效，如果 mask 是黑的，说明是透明区域，跳过检查。

        def get_pixel(inst, side, idx):
            # edges[side] 是 (pixels, mask) 元组
            pxs, msk = inst.edges[side]
            if len(pxs) == 0: return None
            # 处理索引 (支持负数索引)
            if idx < 0: idx += len(pxs)
            if idx < 0 or idx >= len(pxs): return None

            # 检查 mask (有效性)
            if msk[idx] < 128: return None
            return pxs[idx]  # LAB color

        # 获取四个角的颜色
        # TL: 它的右下角 -> Bottom 边的最右点 [-1]
        c_tl = get_pixel(tl, 'bottom', -1)
        # T:  它的左下角 -> Bottom 边的最左点 [0]
        c_t = get_pixel(t, 'bottom', 0)
        # L:  它的右上角 -> Top 边的最右点 [-1]
        c_l = get_pixel(l, 'top', -1)
        # X:  它的左上角 -> Top 边的最左点 [0]
        c_x = get_pixel(x, 'top', 0)

        # 4. 计算一致性误差
        valid_colors = [c for c in [c_tl, c_t, c_l, c_x] if c is not None]

        # 如果有效点太少（比如刚好碰上缺口），不进行惩罚
        if len(valid_colors) < 3:
            return 0.0

        # 计算这些颜色的聚合度（例如：到均值的平均距离）
        valid_colors = np.array(valid_colors, dtype=np.float32)
        center = np.mean(valid_colors, axis=0)
        dists = np.linalg.norm(valid_colors - center, axis=1)
        avg_dist = float(np.mean(dists))

        return avg_dist

    def solve_with_lookahead(self, K: int = 3, depth: int = 2):
        print("[PriorityFrontier] Initializing (MCF + MaxCost)...")
        seed, seed_cost = self._find_best_seed()
        if not seed: return None

        v1, v2, r1, c1, r2, c2 = seed

        # 放置 Seed
        self.grid[(r1, c1)] = v1;
        self.used_pids.add(v1.pid)
        self.min_r, self.max_r = r1, r1 + v1.grid_rows - 1
        self.min_c, self.max_c = c1, c1 + v1.grid_cols - 1

        self.grid[(r2, c2)] = v2;
        self.used_pids.add(v2.pid)
        self.min_r = min(self.min_r, r2);
        self.max_r = max(self.max_r, r2 + v2.grid_rows - 1)
        self.min_c = min(self.min_c, c2);
        self.max_c = max(self.max_c, c2 + v2.grid_cols - 1)

        if self.debug:
            print(f"[Seed] Placed P{v1.pid} & P{v2.pid} (Cost {seed_cost:.1f})")
        if self.image_debug: self._save_step_image(0)

        step = 1
        while len(self.used_pids) < self.num_pieces:
            frontier_slots = self._get_frontier_slots()
            if not frontier_slots:
                print("[PriorityFrontier] Dead end.")
                break

            # ==================== [核心修改 A: 优先填坑] ====================
            # 定义一个临时函数计算 (r,c) 周围有多少个已有邻居
            def count_neighbors(pos):
                r, c = pos
                n = 0
                if (r - 1, c) in self.grid: n += 1
                if (r + 1, c) in self.grid: n += 1
                if (r, c - 1) in self.grid: n += 1
                if (r, c + 1) in self.grid: n += 1
                return n

            # 按邻居数量降序排列：邻居越多，约束越强，越应该先填
            frontier_slots.sort(key=count_neighbors, reverse=True)

            # 激进策略：只考虑那个邻居最多的位置（如果有多个并列最多，都考虑）
            # 这能防止算法在容易的边缘“偷懒”，强迫它去填难填的洞
            max_n = count_neighbors(frontier_slots[0])
            best_slots = [loc for loc in frontier_slots if count_neighbors(loc) == max_n]
            # ================================================================

            heap = []

            for (r, c) in best_slots:
                # 获取邻居对象
                neighbors = []
                if (r - 1, c) in self.grid: neighbors.append(('top', self.grid[(r - 1, c)]))
                if (r + 1, c) in self.grid: neighbors.append(('bottom', self.grid[(r + 1, c)]))
                if (r, c - 1) in self.grid: neighbors.append(('left', self.grid[(r, c - 1)]))
                if (r, c + 1) in self.grid: neighbors.append(('right', self.grid[(r, c + 1)]))

                for pid in range(self.num_pieces):
                    if pid in self.used_pids: continue
                    for rot in self.allowed_rots[pid]:
                        cand_inst = self.variants[pid][rot]
                        if not self._is_valid_geometry(r, c, cand_inst): continue

                        # ==================== [核心修改 B: 调用新逻辑] ====================
                        score, possible = self._evaluate_neighbors(neighbors, cand_inst)

                        if possible:
                            # score 已经是 max_cost 了
                            heapq.heappush(heap, MoveCandidate(score, pid, rot, r, c))
                        # ================================================================

            if not heap:
                print("[PriorityFrontier] No valid moves fit geometry.")
                break

            # Lookahead 部分保持不变
            top_candidates = [heapq.heappop(heap) for _ in range(min(K, len(heap)))]
            best_cand = None
            best_score = float('inf')

            for cand in top_candidates:
                score = self._evaluate_candidate_with_lookahead(cand, depth)
                if score < best_score:
                    best_score = score
                    best_cand = cand

            if best_cand is None: break

            # 落子
            inst = self.variants[best_cand.pid][best_cand.rot]
            for i in range(inst.grid_rows):
                for j in range(inst.grid_cols):
                    self.grid[(best_cand.row + i, best_cand.col + j)] = inst

            self.used_pids.add(best_cand.pid)
            self.min_r = min(self.min_r, best_cand.row)
            self.max_r = max(self.max_r, best_cand.row + inst.grid_rows - 1)
            self.min_c = min(self.min_c, best_cand.col)
            self.max_c = max(self.max_c, best_cand.col + inst.grid_cols - 1)

            if self.debug:
                print(
                    f"   -> [Step {step}] Placed P{best_cand.pid} at ({best_cand.row},{best_cand.col}) Score {best_score:.1f}")
            if self.image_debug: self._save_step_image(step)
            step += 1

        return self._generate_final_placements()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def auto_tune_and_solve(pieces, W, H, args_out, pre_config, image_debug=False, debug_dir=None):
    print("\n[Solver v29] Adaptive Strategy...")

    # 1. 根据 preprocess 的 type 判断是否为旋转矩形
    #   type: 'standard_rect' / 'rotated_rect' / 'irregular'
    puzzle_type = pre_config.get("type", "standard_rect")

    # 2. 自动设置权重
    if puzzle_type == "rotated_rect":
        print("   -> Detected ROTATED puzzles. Using TOLERANT mode (Lower Grad Weight).")
        # 只对旋转矩形启用“宽松模式”：降低梯度 + 模糊 + 开启 complexity bias
        weights = {
            "w_color": 1.0,
            "w_grad": 0.5,
        }
        solve_config = {"blur": True}
    else:
        print("   -> Using STRICT mode for STANDARD / IRREGULAR puzzles.")
        # 直矩形 / 不规则保持原来的严格配置
        weights = {
            "w_color": 1.0,
            "w_grad": 2.5,
        }
        solve_config = {"blur": False}

    # 3. 传入 Solver
    solver = PriorityFrontierSolver(
        pieces, W, H,
        config=solve_config,
        weights=weights,  # 传入权重
        debug=True,
        image_debug=image_debug,
        debug_dir=debug_dir
    )

    sol = solver.solve_with_lookahead(K=LOOKAHEAD_K, depth=LOOKAHEAD_DEPTH)

    if sol:
        save_result(pieces, sol, W, H, args_out)
    else:
        print("No solution found.")


def save_result(pieces, solution, W, H, path):
    # --- 1. 保持原有的画图逻辑 ---
    max_y = max(p.y + p.h for p in solution)
    max_x = max(p.x + p.w for p in solution)
    final_H = max(H, max_y + 10)
    final_W = max(W, max_x + 10)

    canvas = np.zeros((final_H, final_W, 4), dtype=np.uint8)

    for p in solution:
        img = pieces[p.piece_id].image
        msk = pieces[p.piece_id].mask if len(pieces[p.piece_id].mask.shape) == 2 else pieces[p.piece_id].mask[:, :, 0]

        # 执行旋转
        for _ in range(p.rotation):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

        y, x = int(p.y), int(p.x)
        h, w = img.shape[:2]

        # 边界检查
        h_eff = min(h, final_H - y)
        w_eff = min(w, final_W - x)
        if h_eff <= 0 or w_eff <= 0: continue

        # 贴图
        b, g, r_ch = cv2.split(img[:h_eff, :w_eff])
        patch = cv2.merge([b, g, r_ch, msk[:h_eff, :w_eff]])
        roi = canvas[y:y + h_eff, x:x + w_eff]
        valid = msk[:h_eff, :w_eff] > 128
        roi[valid] = patch[valid]
        canvas[y:y + h_eff, x:x + w_eff] = roi

    # 保存最终图片
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR))
    print(f"[Output] Saved image to {path}")

    # --- 2. 新增：保存步骤数据到 JSON ---
    json_data = []
    # 按照 solution 的顺序保存，这样动画就会按照算法放置的顺序播放
    for i, p in enumerate(solution):
        json_data.append({
            "step": i,
            "id": int(p.piece_id),
            "r": int(p.rotation),
            "x": int(p.x),
            "y": int(p.y)
        })

    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"[Output] Saved solution data to {json_path}")


def estimate_canvas(pieces):
    area = sum(p.size[0] * p.size[1] for p in pieces)
    s = int(np.sqrt(area))
    return s, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="solved.png")
    parser.add_argument("--target_w", type=int)
    parser.add_argument("--target_h", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image_debug", action="store_true", help="Save intermediate step images to debug_steps/")
    args = parser.parse_args()

    pieces, config = preprocess_puzzle_image(args.image, args.width, args.height, debug=False)
    if not pieces: return
    W, H = (args.target_w, args.target_h) if args.target_w else estimate_canvas(pieces)

    debug_dir = None
    if args.image_debug:
        debug_dir = "debug_steps"

    auto_tune_and_solve(pieces, W, H, args.out, config, image_debug=args.image_debug, debug_dir=debug_dir)


if __name__ == "__main__":
    main()
