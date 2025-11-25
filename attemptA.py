"""
attemptA.py (matching.py v27.1 - Fix Attribute Error)

Fixes:
1. AttributeError: Renamed PieceInstance fields from 'g_rows/g_cols' to 'grid_rows/grid_cols'
   to match the logic used in the solver.
"""

from __future__ import annotations

import argparse
import time
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import cv2
import numpy as np
from preprocess import PuzzlePiece, preprocess_puzzle_image, EdgeFeatures


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



@dataclass(order=True)
class MoveCandidate:
    cost: float
    pid: int = field(compare=False)
    rot: int = field(compare=False)
    row: int = field(compare=False)  # Grid row
    col: int = field(compare=False)  # Grid col


class PriorityFrontierSolver:
    def __init__(self, pieces, W, H, config, debug):
        self.pieces = pieces
        self.num_pieces = len(pieces)
        self.W = W
        self.H = H
        self.debug = debug
        self.blur_edges = config.get('blur', False)

        # 1. Calculate Atomic Unit Size
        ws = [p.size[1] for p in pieces]
        hs = [p.size[0] for p in pieces]
        self.unit_w = int(min(ws))
        self.unit_h = int(min(hs))

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
                "top":    base_edges["top"],
                "right":  base_edges["right"],
                "bottom": base_edges["bottom"],
                "left":   base_edges["left"],
            }
        elif rot % 4 == 1:
            # 90° CW
            return {
                "top":    self._flip_edge_features(base_edges["left"]),
                "right":  base_edges["top"],
                "bottom": self._flip_edge_features(base_edges["right"]),
                "left":   base_edges["bottom"],
            }
        elif rot % 4 == 2:
            # 180°
            return {
                "top":    self._flip_edge_features(base_edges["bottom"]),
                "right":  self._flip_edge_features(base_edges["left"]),
                "bottom": self._flip_edge_features(base_edges["top"]),
                "left":   self._flip_edge_features(base_edges["right"]),
            }
        else:
            # 270° CW (90° CCW)
            return {
                "top":    base_edges["right"],
                "right":  self._flip_edge_features(base_edges["bottom"]),
                "bottom": base_edges["left"],
                "left":   self._flip_edge_features(base_edges["top"]),
            }

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

                edges = p.edge_lines[r]  # ← 预处理阶段已经算好

                p_vars.append(PieceInstance(
                    pid=p.id,
                    rot=r,
                    h=h,
                    w=w,
                    grid_rows=gr,
                    grid_cols=gc,
                    edges=edges,
                ))

            all_vars.append(p_vars)
        return all_vars

    def _aligned_color_mse(self, pa: np.ndarray, pb: np.ndarray, max_shift: int = 3) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
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
        valid = (mask_a > 128) & (mask_b > 128)

        # 2) near-black 过滤 (LAB 中 L 通道在索引 0)
        nb_a = pixels_a[:, 0] > 5
        nb_b = pixels_b[:, 0] > 5
        valid = valid & nb_a & nb_b

        if np.sum(valid) < 5:
            return 999999.0

        pa = pixels_a[valid]   # (N,3)
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
        w_color = 1.0
        w_grad = 0.1

        combined = w_color * color_mse + w_grad * grad_cost

        # 保持和原逻辑一致：返回 sqrt(MSE-like)
        return float(np.sqrt(combined + 1e-8))





    def _find_best_seed(self):
        best_cost = float('inf')
        seed = None

        for i in range(self.num_pieces):
            for ri in range(4):
                v1 = self.variants[i][ri]
                for j in range(self.num_pieces):
                    if i == j: continue
                    for rj in range(4):
                        v2 = self.variants[j][rj]

                        # Right
                        c = self._compute_edge_diff(v1.edges['right'], v2.edges['left'])
                        if c < best_cost:
                            best_cost = c
                            seed = (v1, v2, 0, 0, 0, v1.grid_cols)

                        # Bottom
                        c = self._compute_edge_diff(v1.edges['bottom'], v2.edges['top'])
                        if c < best_cost:
                            best_cost = c
                            seed = (v1, v2, 0, 0, v1.grid_rows, 0)

        return seed, best_cost

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
    
    def _simulate_greedy_steps(self, max_depth: int) -> float:
        """
        在当前 self.grid/self.used_pids 状态下，向前贪心模拟 max_depth 步。
        仅用于 lookahead 评分，不影响真实解（调用方负责备份/恢复状态）。

        返回：这几步的累计 cost（若无法扩展则返回一个很大的代价）。
        """
        total_cost = 0.0
        steps = 0

        for _ in range(max_depth):
            frontier_slots = self._get_frontier_slots()
            if not frontier_slots:
                break

            heap = []

            for (r, c) in frontier_slots:
                neighbors = []
                if (r - 1, c) in self.grid: neighbors.append(('top', self.grid[(r - 1, c)]))
                if (r + 1, c) in self.grid: neighbors.append(('bottom', self.grid[(r + 1, c)]))
                if (r, c - 1) in self.grid: neighbors.append(('left', self.grid[(r, c - 1)]))
                if (r, c + 1) in self.grid: neighbors.append(('right', self.grid[(r, c + 1)]))

                if not neighbors:
                    continue

                for pid in range(self.num_pieces):
                    if pid in self.used_pids:
                        continue

                    for rot in range(4):
                        cand = self.variants[pid][rot]
                        if not self._is_valid_geometry(r, c, cand):
                            continue

                        total_cost_local = 0.0
                        count = 0
                        possible = True

                        for direction, neighbor in neighbors:
                            if direction == 'top':
                                cost = self._compute_edge_diff(neighbor.edges['bottom'], cand.edges['top'])
                            elif direction == 'bottom':
                                cost = self._compute_edge_diff(neighbor.edges['top'], cand.edges['bottom'])
                            elif direction == 'left':
                                cost = self._compute_edge_diff(neighbor.edges['right'], cand.edges['left'])
                            else:  # 'right'
                                cost = self._compute_edge_diff(neighbor.edges['left'], cand.edges['right'])

                            if cost > 500000:
                                possible = False
                                break
                            total_cost_local += cost
                            count += 1

                        if possible and count > 0:
                            avg_cost = total_cost_local / count
                            heapq.heappush(heap, MoveCandidate(avg_cost, pid, rot, r, c))

            if not heap:
                break

            best = heapq.heappop(heap)
            inst = self.variants[best.pid][best.rot]

            for i in range(inst.grid_rows):
                for j in range(inst.grid_cols):
                    self.grid[(best.row + i, best.col + j)] = inst

            self.used_pids.add(best.pid)
            self.min_r = min(self.min_r, best.row)
            self.max_r = max(self.max_r, best.row + inst.grid_rows - 1)
            self.min_c = min(self.min_c, best.col)
            self.max_c = max(self.max_c, best.col + inst.grid_cols - 1)

            total_cost += best.cost
            steps += 1

        if steps == 0:
            return 1e9  # 无法扩展，视为很差
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

    def solve(self):
        print("[PriorityFrontier] Initializing...")
        seed, seed_cost = self._find_best_seed()
        if not seed: return None

        v1, v2, r1, c1, r2, c2 = seed

        self.grid[(r1, c1)] = v1
        self.used_pids.add(v1.pid)
        self.min_r, self.max_r = r1, r1 + v1.grid_rows - 1
        self.min_c, self.max_c = c1, c1 + v1.grid_cols - 1

        self.grid[(r2, c2)] = v2
        self.used_pids.add(v2.pid)
        self.min_r = min(self.min_r, r2)
        self.max_r = max(self.max_r, r2 + v2.grid_rows - 1)
        self.min_c = min(self.min_c, c2)
        self.max_c = max(self.max_c, c2 + v2.grid_cols - 1)

        if self.debug: print(f"[PriorityFrontier] Seed Placed (Cost {seed_cost:.1f}).")

        while len(self.used_pids) < self.num_pieces:
            frontier_slots = self._get_frontier_slots()
            if not frontier_slots:
                print("[PriorityFrontier] Dead end (Boxed in).")
                break

            heap = []

            for (r, c) in frontier_slots:
                neighbors = []
                if (r - 1, c) in self.grid: neighbors.append(('top', self.grid[(r - 1, c)]))
                if (r + 1, c) in self.grid: neighbors.append(('bottom', self.grid[(r + 1, c)]))
                if (r, c - 1) in self.grid: neighbors.append(('left', self.grid[(r, c - 1)]))
                if (r, c + 1) in self.grid: neighbors.append(('right', self.grid[(r, c + 1)]))

                if not neighbors: continue

                for pid in range(self.num_pieces):
                    if pid in self.used_pids: continue

                    for rot in range(4):
                        cand = self.variants[pid][rot]
                        if not self._is_valid_geometry(r, c, cand): continue

                        total_cost = 0
                        count = 0
                        possible = True
                        for direction, neighbor in neighbors:
                            cost = 0
                            if direction == 'top':
                                cost = self._compute_edge_diff(neighbor.edges['bottom'], cand.edges['top'])
                            elif direction == 'bottom':
                                cost = self._compute_edge_diff(neighbor.edges['top'], cand.edges['bottom'])
                            elif direction == 'left':
                                cost = self._compute_edge_diff(neighbor.edges['right'], cand.edges['left'])
                            elif direction == 'right':
                                cost = self._compute_edge_diff(neighbor.edges['left'], cand.edges['right'])

                            if cost > 500000: 
                                possible = False
                                break
                            total_cost += cost
                            count += 1

                        if possible and count > 0:
                            avg_cost = total_cost / count
                            heapq.heappush(heap, MoveCandidate(avg_cost, pid, rot, r, c))

            if not heap:
                print("[PriorityFrontier] No valid moves fit geometry.")
                break

            best = heapq.heappop(heap)

            inst = self.variants[best.pid][best.rot]
            # Assign all cells
            for i in range(inst.grid_rows):
                for j in range(inst.grid_cols):
                    self.grid[(best.row + i, best.col + j)] = inst

            self.used_pids.add(best.pid)
            self.min_r = min(self.min_r, best.row)
            self.max_r = max(self.max_r, best.row + inst.grid_rows - 1)
            self.min_c = min(self.min_c, best.col)
            self.max_c = max(self.max_c, best.col + inst.grid_cols - 1)

            if self.debug: print(f"   -> Placed P{best.pid} at ({best.row},{best.col}) Cost {best.cost:.1f}")

        return self._generate_final_placements()

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
    
    def solve_with_lookahead(self, K: int = 3, depth: int = 2):
        """
        带有限深回溯的求解：
          - 每一步先构建所有候选 MoveCandidate
          - 从中取 cost 最小的前 K 个
          - 对这 K 个分别做 depth 步 lookahead 评分
          - 选评分最小的一个真正落子
        K 和 depth 都不宜过大，默认 K=3, depth=2 对 N≈16 的拼图一般足够。
        """
        print("[PriorityFrontier] Initializing with lookahead...")
        seed, seed_cost = self._find_best_seed()
        if not seed:
            return None

        v1, v2, r1, c1, r2, c2 = seed

        # 放置 seed 的两个 piece
        self.grid[(r1, c1)] = v1
        self.used_pids.add(v1.pid)
        self.min_r, self.max_r = r1, r1 + v1.grid_rows - 1
        self.min_c, self.max_c = c1, c1 + v1.grid_cols - 1

        self.grid[(r2, c2)] = v2
        self.used_pids.add(v2.pid)
        self.min_r = min(self.min_r, r2)
        self.max_r = max(self.max_r, r2 + v2.grid_rows - 1)
        self.min_c = min(self.min_c, c2)
        self.max_c = max(self.max_c, c2 + v2.grid_cols - 1)

        if self.debug:
            print(f"[PriorityFrontier] Seed Placed (Cost {seed_cost:.1f}).")

        while len(self.used_pids) < self.num_pieces:
            frontier_slots = self._get_frontier_slots()
            if not frontier_slots:
                print("[PriorityFrontier] Dead end (Boxed in).")
                break

            heap = []

            # 和 solve 中相同的候选生成逻辑
            for (r, c) in frontier_slots:
                neighbors = []
                if (r - 1, c) in self.grid: neighbors.append(('top', self.grid[(r - 1, c)]))
                if (r + 1, c) in self.grid: neighbors.append(('bottom', self.grid[(r + 1, c)]))
                if (r, c - 1) in self.grid: neighbors.append(('left', self.grid[(r, c - 1)]))
                if (r, c + 1) in self.grid: neighbors.append(('right', self.grid[(r, c + 1)]))

                if not neighbors:
                    continue

                for pid in range(self.num_pieces):
                    if pid in self.used_pids:
                        continue

                    for rot in range(4):
                        cand_inst = self.variants[pid][rot]
                        if not self._is_valid_geometry(r, c, cand_inst):
                            continue

                        total_cost = 0.0
                        count = 0
                        possible = True

                        for direction, neighbor in neighbors:
                            if direction == 'top':
                                cost = self._compute_edge_diff(neighbor.edges['bottom'], cand_inst.edges['top'])
                            elif direction == 'bottom':
                                cost = self._compute_edge_diff(neighbor.edges['top'], cand_inst.edges['bottom'])
                            elif direction == 'left':
                                cost = self._compute_edge_diff(neighbor.edges['right'], cand_inst.edges['left'])
                            else:  # 'right'
                                cost = self._compute_edge_diff(neighbor.edges['left'], cand_inst.edges['right'])

                            if cost > 500000:
                                possible = False
                                break
                            total_cost += cost
                            count += 1

                        if possible and count > 0:
                            avg_cost = total_cost / count
                            heapq.heappush(heap, MoveCandidate(avg_cost, pid, rot, r, c))

            if not heap:
                print("[PriorityFrontier] No valid moves fit geometry.")
                break

            # 取前 K 个候选做 lookahead 评分
            top_candidates = []
            for _ in range(min(K, len(heap))):
                top_candidates.append(heapq.heappop(heap))

            best_cand = None
            best_score = float('inf')

            for cand in top_candidates:
                score = self._evaluate_candidate_with_lookahead(cand, depth)
                if score < best_score:
                    best_score = score
                    best_cand = cand

            if best_cand is None:
                print("[PriorityFrontier] Lookahead could not find a feasible move.")
                break

            # 真正落子
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
                print(f"   -> [Lookahead] Placed P{best_cand.pid} at ({best_cand.row},{best_cand.col}) "
                      f"Cost {best_cand.cost:.1f}, Score {best_score:.1f}")

        return self._generate_final_placements()



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def auto_tune_and_solve(pieces, W, H, args_out, pre_config, lookahead = True):
    print("\n[Solver v27] Priority Frontier Strategy...")
    p_type = pre_config.get('type', 'standard_rect')
    config = {'blur': False}
    if p_type == 'rotated_rect': config['blur'] = True

    solver = PriorityFrontierSolver(pieces, W, H, config, debug=True)

    if lookahead:
        sol = solver.solve_with_lookahead(K=3, depth=2)
    else:
        sol = solver.solve()

    if sol:
        save_result(pieces, sol, W, H, args_out)
    else:
        print("No solution found.")


def save_result(pieces, solution, W, H, path):
    max_y = max(p.y + p.h for p in solution)
    max_x = max(p.x + p.w for p in solution)
    final_H = max(H, max_y + 10)
    final_W = max(W, max_x + 10)

    canvas = np.zeros((final_H, final_W, 4), dtype=np.uint8)

    for p in solution:
        img = pieces[p.piece_id].image
        msk = pieces[p.piece_id].mask if len(pieces[p.piece_id].mask.shape) == 2 else pieces[p.piece_id].mask[:, :, 0]
        for _ in range(p.rotation):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

        y, x = p.y, p.x
        h, w = img.shape[:2]
        h_eff = min(h, final_H - y)
        w_eff = min(w, final_W - x)
        if h_eff <= 0 or w_eff <= 0: continue

        b, g, r = cv2.split(img[:h_eff, :w_eff])
        patch = cv2.merge([b, g, r, msk[:h_eff, :w_eff]])
        roi = canvas[y:y + h_eff, x:x + w_eff]
        valid = msk[:h_eff, :w_eff] > 128
        roi[valid] = patch[valid]
        canvas[y:y + h_eff, x:x + w_eff] = roi

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR))
    print(f"[Output] Saved to {path}")


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
    args = parser.parse_args()

    pieces, config = preprocess_puzzle_image(args.image, args.width, args.height, debug=False)
    if not pieces: return
    W, H = (args.target_w, args.target_h) if args.target_w else estimate_canvas(pieces)
    auto_tune_and_solve(pieces, W, H, args.out, config)


if __name__ == "__main__":
    main()
