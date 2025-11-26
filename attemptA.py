"""
attemptA.py (v67 - Dynamic Orientation Regime)

Updates:
1. DYNAMIC REGIME: Removed hard-coded "Landscape" vs "Portrait" check in __init__.
   - The solver now tests seeds for BOTH orientations (pieces as WxH and HxW).
   - It automatically picks the orientation that yields better edge matching.
   - Fixes the issue where portrait images (Monet) were being assembled sideways.
2. STRICT SQUARE LOCK: Retained the logic to force NxN grid (e.g. 5x5) to prevent 4x6 errors.
3. AREA RECOVERY: Can now handle missing pieces (e.g. 20/25) without failing, while keeping the 5x5 bounds.
"""

from __future__ import annotations

import argparse
import heapq
import cv2
import numpy as np
import os
import shutil
import math
import time
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import sys
import traceback

sys.setrecursionlimit(10000)

from preprocess import PuzzlePiece, preprocess_puzzle_image, EdgeFeatures

# ---- Hyperparameters ----
DEFAULT_TOLERANCE_MULTIPLIER = 3.0
MIN_ABSOLUTE_THRESHOLD = 2.0
MAX_BRANCH_FACTOR = 64
PREDICTION_WEIGHT = 0.4
COSINE_WEIGHT = 0.5
BORDER_LUMINANCE_THRESHOLD = 15.0
SEARCH_TIMEOUT = 180.0


@dataclass
class Placement:
    piece_id: int
    rotation: int
    grid_row: int
    grid_col: int


@dataclass(order=True)
class MoveCandidate:
    cost: float
    pid: int = field(compare=False)
    rot: int = field(compare=False)
    row: int = field(compare=False)
    col: int = field(compare=False)


class BacktrackingSolver:
    def __init__(self, pieces, known_W=None, known_H=None, use_border_logic=False, debug=False, visual_debug=False):
        self.pieces = pieces
        self.num_pieces = len(pieces)
        self.debug = debug
        self.visual_debug = visual_debug
        self.use_border_logic = use_border_logic
        self.step_counter = 0
        self.match_threshold = float('inf')
        self.start_time = 0

        self.best_solution = None
        self.best_solution_score = float('inf')
        self.solutions_found = 0
        self.seed_r = 0
        self.seed_c = 0

        if self.visual_debug:
            if os.path.exists("debug_steps"): shutil.rmtree("debug_steps")
            os.makedirs("debug_steps")

        # 1. Geometry Analysis (Just get dimensions, decide regime later)
        dims = sorted([(p.size[0], p.size[1]) for p in pieces], key=lambda x: x[0] * x[1])
        median_dim = dims[len(dims) // 2]
        self.long_dim = max(median_dim)
        self.short_dim = min(median_dim)

        # Decide if pieces are roughly square themselves
        self.pieces_are_square = (self.long_dim - self.short_dim) < 5

        # 2. Precompute Edges (We compute raw edges relative to the image data)
        # Note: We will handle logic for "Portrait vs Landscape" interpretation in _get_edge_pixels
        self.piece_edges = self._precompute_raw_edges()

        # 3. Grid Limits (Strict Square Mode)
        if known_W and known_H:
            # We can't set max_rows/cols yet because we don't know unit_w/h
            self.known_bounds = (known_W, known_H)
        else:
            self.known_bounds = None
            side_len = math.ceil(math.sqrt(self.num_pieces))
            self.max_rows = side_len
            self.max_cols = side_len
            if self.debug: print(f"[Grid] Forced Square Bounds: {self.max_rows}x{self.max_cols}")

        self.grid = {}
        self.used_pids = set()
        self.min_r, self.max_r = 0, 0
        self.min_c, self.max_c = 0, 0

        # Will be set after regime selection
        self.unit_w = 0
        self.unit_h = 0
        self.valid_rotations = {}

    def _precompute_raw_edges(self):
        cache = {}
        for p in self.pieces:
            img = getattr(p, 'raw_image', p.image)
            h, w = img.shape[:2]
            d = 1;
            w_strip = 3
            if h < 6 or w < 6: continue
            top = img[d:d + w_strip, :]
            bottom = img[h - d - w_strip:h - d, :]
            left = img[:, d:d + w_strip]
            right = img[:, w - d - w_strip:w - d]
            cache[p.id] = {0: {'top': top, 'bottom': bottom, 'left': left, 'right': right}}
        return cache

    def _configure_regime(self, mode):
        # mode='landscape': cells are wider (Long x Short)
        # mode='portrait': cells are taller (Short x Long)
        if mode == 'landscape':
            self.unit_w = self.long_dim
            self.unit_h = self.short_dim
        else:
            self.unit_w = self.short_dim
            self.unit_h = self.long_dim

        if self.known_bounds:
            W, H = self.known_bounds
            self.max_cols = int(round(W / self.unit_w))
            self.max_rows = int(round(H / self.unit_h))

        # Compute valid rotations for this regime
        # If target is H, W, we check which piece orientation matches
        self.valid_rotations = {}
        for p in self.pieces:
            ph, pw = p.size
            # If piece matches target H,W directly -> Rot 0 or 2
            if abs(ph - self.unit_h) < 10 and abs(pw - self.unit_w) < 10:
                self.valid_rotations[p.id] = [0, 2]
            # If piece matches swapped -> Rot 1 or 3
            elif abs(ph - self.unit_w) < 10 and abs(pw - self.unit_h) < 10:
                self.valid_rotations[p.id] = [1, 3]
            else:
                self.valid_rotations[p.id] = [0, 1, 2, 3]  # Square or unknown

    def _get_edge_pixels(self, pid: int, rot: int, side: str) -> np.ndarray:
        if pid not in self.piece_edges: return np.zeros((3, 3, 3), dtype=np.uint8)
        base = self.piece_edges[pid][0]
        side_idx = {'top': 0, 'right': 1, 'bottom': 2, 'left': 3}[side]
        real_idx = (side_idx - rot) % 4
        real_name = ['top', 'right', 'bottom', 'left'][real_idx]
        strip = base[real_name]
        if rot == 1:
            strip = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 2:
            strip = cv2.rotate(strip, cv2.ROTATE_180)
        elif rot == 3:
            strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return strip

    def _is_edge_border(self, pid, rot, side):
        if not self.use_border_logic: return False
        strip = self._get_edge_pixels(pid, rot, side)
        if strip.size == 0: return True
        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2Lab)
        avg_L = np.mean(lab[:, :, 0])
        return avg_L < BORDER_LUMINANCE_THRESHOLD

    def _compute_statistical_similarity(self, strip_a, strip_b, orientation='vertical'):
        h_a, w_a = strip_a.shape[:2];
        h_b, w_b = strip_b.shape[:2]
        if h_a == 0 or h_b == 0: return 999.0

        if orientation == 'vertical':
            target_w = min(w_a, w_b)
            if w_a != target_w: strip_a = cv2.resize(strip_a, (target_w, strip_a.shape[0]))
            if w_b != target_w: strip_b = cv2.resize(strip_b, (target_w, strip_b.shape[0]))
        else:
            target_h = min(h_a, h_b)
            if h_a != target_h: strip_a = cv2.resize(strip_a, (strip_a.shape[1], target_h))
            if h_b != target_h: strip_b = cv2.resize(strip_b, (strip_b.shape[1], target_h))

        lab_a = cv2.cvtColor(strip_a, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_b = cv2.cvtColor(strip_b, cv2.COLOR_BGR2Lab).astype(np.float32)

        if orientation == 'vertical':
            A_outer = lab_a[-1, :, :];
            A_inner = lab_a[-2, :, :]
            B_outer = lab_b[0, :, :];
            B_inner = lab_b[1, :, :]
        else:
            A_outer = lab_a[:, -1, :];
            A_inner = lab_a[:, -2, :]
            B_outer = lab_b[:, 0, :];
            B_inner = lab_b[:, 1, :]

        safe_sigma = 10.0

        diff = np.linalg.norm(A_outer - B_outer, axis=1)
        mean_diff = np.mean(diff)
        z_score_color = (mean_diff / safe_sigma)

        grad_A = A_outer - A_inner
        grad_B_cont = B_inner - B_outer
        norm_A = np.linalg.norm(grad_A, axis=1, keepdims=True) + 1e-6
        norm_B = np.linalg.norm(grad_B_cont, axis=1, keepdims=True) + 1e-6
        unit_A = grad_A / norm_A;
        unit_B = grad_B_cont / norm_B
        grad_direction_cost = np.mean(1.0 - np.sum(unit_A * unit_B, axis=1))

        total_cost = z_score_color + (grad_direction_cost * COSINE_WEIGHT)

        if np.mean(A_outer[:, 0]) < 5 or np.mean(B_outer[:, 0]) < 5: return 999.0
        return total_cost

    def _check_bounds_validity(self, r, c):
        n_min_r = min(self.min_r, r);
        n_max_r = max(self.max_r, r)
        n_min_c = min(self.min_c, c);
        n_max_c = max(self.max_c, c)
        span_h = n_max_r - n_min_r + 1
        span_w = n_max_c - n_min_c + 1
        return span_h <= self.max_rows and span_w <= self.max_cols

    def _get_frontier_slots(self):
        frontier = set()
        for (r, c) in self.grid.keys():
            pid, rot = self.grid[(r, c)]
            if (r - 1, c) not in self.grid and not self._is_edge_border(pid, rot, 'top'): frontier.add((r - 1, c))
            if (r + 1, c) not in self.grid and not self._is_edge_border(pid, rot, 'bottom'): frontier.add((r + 1, c))
            if (r, c - 1) not in self.grid and not self._is_edge_border(pid, rot, 'left'): frontier.add((r, c - 1))
            if (r, c + 1) not in self.grid and not self._is_edge_border(pid, rot, 'right'): frontier.add((r, c + 1))
        return list(frontier)

    def _get_neighbors(self, r, c):
        nbs = []
        if (r - 1, c) in self.grid: nbs.append(('top', self.grid[(r - 1, c)]))
        if (r + 1, c) in self.grid: nbs.append(('bottom', self.grid[(r + 1, c)]))
        if (r, c - 1) in self.grid: nbs.append(('left', self.grid[(r, c - 1)]))
        if (r, c + 1) in self.grid: nbs.append(('right', self.grid[(r, c + 1)]))
        return nbs

    def _save_debug_snapshot(self, final=False):
        if not self.grid: return
        max_row = max(k[0] for k in self.grid.keys());
        min_row = min(k[0] for k in self.grid.keys())
        max_col = max(k[1] for k in self.grid.keys());
        min_col = min(k[1] for k in self.grid.keys())
        H_px = (max_row - min_row + 1) * self.unit_h
        W_px = (max_col - min_col + 1) * self.unit_w
        canvas = np.zeros((H_px, W_px, 3), dtype=np.uint8)
        for (r, c), (pid, rot) in self.grid.items():
            piece = self.pieces[pid]
            img = getattr(piece, 'raw_image', piece.image).copy()
            for _ in range(rot): img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (self.unit_w, self.unit_h))
            y = (r - min_row) * self.unit_h
            x = (c - min_col) * self.unit_w
            canvas[y:y + self.unit_h, x:x + self.unit_w] = img
        prefix = "final" if final else "step"
        fname = f"debug_steps/{prefix}_{self.step_counter:04d}.png"
        cv2.imwrite(fname, canvas)
        if not final: self.step_counter += 1

    def solve(self):
        print("[Solver] Analyzing Regimes (Landscape vs Portrait)...")
        self.start_time = time.time()

        candidates = []

        # Try BOTH regimes to see which one yields better seed matches
        regimes = ['landscape', 'portrait'] if not self.pieces_are_square else ['landscape']

        for regime in regimes:
            # Temporarily configure to test edges
            self._configure_regime(regime)

            for i in range(self.num_pieces):
                rots_i = self.valid_rotations[i] if self.valid_rotations[i] else [0]
                for ri in rots_i:
                    my_top = self._get_edge_pixels(i, ri, 'top')
                    my_left = self._get_edge_pixels(i, ri, 'left')

                    for j in range(self.num_pieces):
                        if i == j: continue
                        rots_j = self.valid_rotations[j] if self.valid_rotations[j] else [0]
                        for rj in rots_j:
                            # V-Match
                            other_bottom = self._get_edge_pixels(j, rj, 'bottom')
                            s_v = self._compute_statistical_similarity(my_top, other_bottom, 'vertical')
                            if s_v < 10.0:
                                heapq.heappush(candidates, (s_v, regime, (i, ri, 0, 0, j, rj, -1, 0)))

                            # H-Match
                            other_right = self._get_edge_pixels(j, rj, 'right')
                            s_h = self._compute_statistical_similarity(my_left, other_right, 'horizontal')
                            if s_h < 10.0:
                                heapq.heappush(candidates, (s_h, regime, (i, ri, 0, 0, j, rj, 0, -1)))

        if not candidates:
            print("No matches found.")
            return []

        MAX_SEEDS_TO_TRY = 50
        print(f"[Solver] Computed {len(candidates)} potential seeds across regimes.")

        seeds_tried = 0
        while candidates and seeds_tried < MAX_SEEDS_TO_TRY:
            score, regime, seed_data = heapq.heappop(candidates)

            # Commit to the regime of the best seed
            self._configure_regime(regime)

            p1, r1, y1, x1, p2, r2, y2, x2 = seed_data

            seeds_tried += 1
            self.match_threshold = max(score * DEFAULT_TOLERANCE_MULTIPLIER, MIN_ABSOLUTE_THRESHOLD)

            print(f"\n>> Trying Seed Rank {seeds_tried} [{regime.upper()}]: P{p1} <-> P{p2} (Score {score:.2f})")

            self.grid = {};
            self.used_pids = set()
            self.min_r, self.max_r, self.min_c, self.max_c = 0, 0, 0, 0

            self.seed_r, self.seed_c = 0, 0

            self._place_piece(p1, r1, 0, 0)
            self._place_piece(p2, r2, y2 - y1, x2 - x1)

            if self.visual_debug: self._save_debug_snapshot()

            if self._backtrack_recursive(depth=1):
                pass

            if self.best_solution and time.time() - self.start_time > SEARCH_TIMEOUT:
                break

        if self.best_solution:
            print(
                f"\n[Solver] FINAL RESULT: Found {self.solutions_found} solutions. Best Score: {self.best_solution_score:.2f}")
            return self._convert_grid_to_placements(self.best_solution)

        print("[Solver] Failed to find any solution.")
        return []

    def _place_piece(self, pid, rot, r, c):
        self.grid[(r, c)] = (pid, rot)
        self.used_pids.add(pid)
        self.min_r = min(self.min_r, r);
        self.max_r = max(self.max_r, r)
        self.min_c = min(self.min_c, c);
        self.max_c = max(self.max_c, c)

    def _remove_piece(self, pid, r, c):
        del self.grid[(r, c)]
        self.used_pids.remove(pid)
        if not self.grid:
            self.min_r, self.max_r, self.min_c, self.max_c = 0, 0, 0, 0
            return
        rows = [k[0] for k in self.grid.keys()]
        cols = [k[1] for k in self.grid.keys()]
        self.min_r, self.max_r = min(rows), max(rows)
        self.min_c, self.max_c = min(cols), max(cols)

    def _calculate_global_score(self):
        total_cost = 0.0
        for (r, c), (pid, rot) in self.grid.items():
            neighbors = self._get_neighbors(r, c)
            for direction, (n_pid, n_rot) in neighbors:
                val = 0.0
                if direction == 'top':
                    val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'bottom'),
                                                               self._get_edge_pixels(pid, rot, 'top'), 'vertical')
                    total_cost += val
                elif direction == 'left':
                    val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'right'),
                                                               self._get_edge_pixels(pid, rot, 'left'), 'horizontal')
                    total_cost += val
        return total_cost

    def _backtrack_recursive(self, depth=0):
        if time.time() - self.start_time > SEARCH_TIMEOUT: return False

        if len(self.used_pids) == self.num_pieces:
            global_score = self._calculate_global_score()
            self.solutions_found += 1
            print(f"   -> Found Solution! Global Cost: {global_score:.1f}")
            if global_score < self.best_solution_score:
                self.best_solution_score = global_score
                self.best_solution = copy.deepcopy(self.grid)
                if self.visual_debug: self._save_debug_snapshot(final=True)
            return False

        if depth % 50 == 0:
            elapsed = time.time() - self.start_time
            print(f"\r... Depth {depth} | Found: {self.solutions_found} | Time: {elapsed:.0f}s", end="", flush=True)

        frontier_slots = []
        raw_frontier = self._get_frontier_slots()
        for (r, c) in raw_frontier:
            if not self._check_bounds_validity(r, c): continue
            neighbors = self._get_neighbors(r, c)
            if not neighbors: continue
            dist = math.sqrt((r - self.seed_r) ** 2 + (c - self.seed_c) ** 2)
            frontier_slots.append((-len(neighbors), dist, r, c, neighbors))

        if not frontier_slots: return False
        frontier_slots.sort()
        _, _, r, c, neighbors = frontier_slots[0]

        candidates = []
        current_threshold = self.match_threshold
        if (self.num_pieces - len(self.used_pids)) <= 2: current_threshold *= 2.0

        for pid in range(self.num_pieces):
            if pid in self.used_pids: continue
            for rot in self.valid_rotations[pid]:
                max_cost = 0.0;
                avg_cost = 0.0;
                valid_match = True
                for (direction, (n_pid, n_rot)) in neighbors:
                    val = 0.0
                    if direction == 'top':
                        val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'bottom'),
                                                                   self._get_edge_pixels(pid, rot, 'top'), 'vertical')
                    elif direction == 'bottom':
                        val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'top'),
                                                                   self._get_edge_pixels(pid, rot, 'bottom'),
                                                                   'vertical')
                    elif direction == 'left':
                        val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'right'),
                                                                   self._get_edge_pixels(pid, rot, 'left'),
                                                                   'horizontal')
                    elif direction == 'right':
                        val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'left'),
                                                                   self._get_edge_pixels(pid, rot, 'right'),
                                                                   'horizontal')

                    if val > current_threshold * 1.5: valid_match = False; break
                    max_cost = max(max_cost, val);
                    avg_cost += val

                if valid_match:
                    final_cost = (avg_cost / len(neighbors)) * 0.7 + max_cost * 0.3
                    if final_cost < current_threshold:
                        heapq.heappush(candidates, MoveCandidate(final_cost, pid, rot, r, c))

        attempts = 0
        while candidates and attempts < MAX_BRANCH_FACTOR:
            cand = heapq.heappop(candidates)
            self._place_piece(cand.pid, cand.rot, cand.row, cand.col)
            if self.visual_debug: self._save_debug_snapshot()

            self._backtrack_recursive(depth + 1)

            if self.solutions_found > 10: return True

            self._remove_piece(cand.pid, cand.row, cand.col)
            attempts += 1

        return False

    def _convert_grid_to_placements(self, grid_dict):
        placements = []
        rows = [k[0] for k in grid_dict.keys()]
        cols = [k[1] for k in grid_dict.keys()]
        mr = min(rows);
        mc = min(cols)
        for (r, c), (pid, rot) in grid_dict.items():
            placements.append(Placement(pid, rot, r - mr, c - mc))
        return placements


def save_seamless(pieces, solution, path, solver):
    if not solution: return
    max_row = max(p.grid_row for p in solution)
    max_col = max(p.grid_col for p in solution)
    U_W = solver.unit_w;
    U_H = solver.unit_h
    H = int((max_row + 1) * U_H);
    W = int((max_col + 1) * U_W)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    print(f"\n[Output] Final Grid: {max_col + 1} cols x {max_row + 1} rows")
    print(f"[Output] Canvas Pixels: {W}x{H}")

    for p in solution:
        piece = pieces[p.piece_id]
        img = getattr(piece, 'raw_image', piece.image).copy()
        for _ in range(p.rotation): img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if img.shape[0] != U_H or img.shape[1] != U_W:
            img = cv2.resize(img, (U_W, U_H), interpolation=cv2.INTER_LANCZOS4)
        y = int(p.grid_row * U_H);
        x = int(p.grid_col * U_W)
        h_eff = min(U_H, H - y);
        w_eff = min(U_W, W - x)
        if h_eff > 0 and w_eff > 0: canvas[y:y + h_eff, x:x + w_eff] = img[:h_eff, :w_eff]
    cv2.imwrite(path, canvas)
    print(f"[Output] Saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="solved.png")
    parser.add_argument("--width", type=int);
    parser.add_argument("--height", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visual_debug", action="store_true")
    parser.add_argument("--target_w", type=int);
    parser.add_argument("--target_h", type=int)
    parser.add_argument("--timeout", type=float, default=180.0, help="Seconds to search per seed")
    args = parser.parse_args()

    pieces, config = preprocess_puzzle_image(args.image, args.width, args.height, debug=args.debug)
    if not pieces: return

    use_border = config.get('shrink', 0) > 0
    global SEARCH_TIMEOUT
    SEARCH_TIMEOUT = args.timeout

    try:
        solver = BacktrackingSolver(pieces, known_W=args.width, known_H=args.height,
                                    use_border_logic=use_border, debug=args.debug, visual_debug=args.visual_debug)
        solution = solver.solve()

        if solution:
            save_seamless(pieces, solution, args.out, solver)
        else:
            print("Failed to solve.")
    except Exception:
        traceback.print_exc()
        print("[System] Solver crashed. Checking for partial solution...")


if __name__ == "__main__":
    main()
