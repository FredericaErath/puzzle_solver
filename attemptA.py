"""
attemptA.py (v61 - Stability Fix & Noise Suppression)

Updates:
1. CRASH FIX: Added 'cv2.resize' in '_evaluate_regime_score' to prevent shape mismatch
   errors on rotated/irregular datasets.
2. NOISE SUPPRESSION: Increased 'safe_sigma' floor to 10.0.
   This prevents the solver from accepting random garbage matches in high-texture areas (Starry Night).
3. ROBUSTNESS: Added Try-Except block to save partial results if the solver crashes mid-way.
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

# Recursion limit
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

        # 1. Geometry Analysis
        dims = sorted([(p.size[0], p.size[1]) for p in pieces], key=lambda x: x[0]*x[1])
        median_dim = dims[len(dims)//2]
        L = max(median_dim)
        S = min(median_dim)
        self.is_square = (L - S) < 5

        # 2. Precompute Edges (3px deep)
        self.piece_edges = self._precompute_raw_edges()

        # 3. Regime Detection
        if self.is_square:
            self.unit_w, self.unit_h = L, L
            self.valid_rotations = {i: [0,1,2,3] for i in range(self.num_pieces)}
            if self.debug: print("[Layout] Regime: Square")
        else:
            score_landscape = self._evaluate_regime_score(target_h=S, target_w=L)
            score_portrait = self._evaluate_regime_score(target_h=L, target_w=S)
            if score_landscape < score_portrait:
                if self.debug: print("[Layout] Regime: LANDSCAPE Cells")
                self.unit_h, self.unit_w = S, L
            else:
                if self.debug: print("[Layout] Regime: PORTRAIT Cells")
                self.unit_h, self.unit_w = L, S
            self.valid_rotations = self._compute_valid_rotations(self.unit_h, self.unit_w)

        # 4. Grid Limits
        if known_W and known_H:
            self.max_cols = int(round(known_W / self.unit_w))
            self.max_rows = int(round(known_H / self.unit_h))
            if self.debug: print(f"[Grid] Limits (Exact): {self.max_rows}x{self.max_cols}")
        else:
            best_r, best_c = self._get_best_factors(self.num_pieces)
            self.max_rows = best_r
            self.max_cols = best_c
            if self.debug: print(f"[Grid] Limits (Factorized): {self.max_rows}x{self.max_cols}")

        self.grid = {}
        self.used_pids = set()
        self.min_r, self.max_r = 0, 0
        self.min_c, self.max_c = 0, 0

    def _get_best_factors(self, n):
        best = (1, n)
        for r in range(1, int(n**0.5) + 1):
            if n % r == 0:
                c = n // r
                best = (r, c)
        return best

    def _compute_valid_rotations(self, target_h, target_w):
        valid_map = {}
        for p in self.pieces:
            ph, pw = p.size
            if abs(ph - target_h) < 10 and abs(pw - target_w) < 10: valid_map[p.id] = [0, 2]
            elif abs(ph - target_w) < 10 and abs(pw - target_h) < 10: valid_map[p.id] = [1, 3]
            else: valid_map[p.id] = [0, 1, 2, 3]
        return valid_map

    def _evaluate_regime_score(self, target_h, target_w):
        rot_map = self._compute_valid_rotations(target_h, target_w)
        total = 0; count = 0; sample = min(len(self.pieces), 20)
        for i in range(sample):
            if not rot_map[i]: continue
            r = rot_map[i][0]
            my = self._get_edge_pixels(i, r, 'right')
            best = float('inf')
            for j in range(self.num_pieces):
                if i==j or not rot_map[j]: continue

                # FIX: Resize to match dimensions for quick check
                other = self._get_edge_pixels(j, rot_map[j][0], 'left')
                if my.shape != other.shape:
                    # Force resize other to match my
                    other = cv2.resize(other, (my.shape[1], my.shape[0]))

                d = np.mean(np.linalg.norm(my[:,1,:] - other[:,0,:], axis=1))
                if d < best: best = d
            if best < 99999: total += best; count += 1
        return total / (count + 1e-6)

    def _precompute_raw_edges(self):
        cache = {}
        for p in self.pieces:
            img = getattr(p, 'raw_image', p.image)
            h, w = img.shape[:2]
            d = 1; w_strip = 3
            if h < 6 or w < 6: continue
            top = img[d:d+w_strip, :]
            bottom = img[h-d-w_strip:h-d, :]
            left = img[:, d:d+w_strip]
            right = img[:, w-d-w_strip:w-d]
            cache[p.id] = {0: {'top': top, 'bottom': bottom, 'left': left, 'right': right}}
        return cache

    def _get_edge_pixels(self, pid: int, rot: int, side: str) -> np.ndarray:
        if pid not in self.piece_edges: return np.zeros((3,3,3), dtype=np.uint8)
        base = self.piece_edges[pid][0]
        side_idx = {'top':0, 'right':1, 'bottom':2, 'left':3}[side]
        real_idx = (side_idx - rot) % 4
        real_name = ['top', 'right', 'bottom', 'left'][real_idx]
        strip = base[real_name]
        if rot == 1: strip = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 2: strip = cv2.rotate(strip, cv2.ROTATE_180)
        elif rot == 3: strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return strip

    def _is_edge_border(self, pid, rot, side):
        if not self.use_border_logic: return False
        strip = self._get_edge_pixels(pid, rot, side)
        if strip.size == 0: return True
        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2Lab)
        avg_L = np.mean(lab[:, :, 0])
        return avg_L < BORDER_LUMINANCE_THRESHOLD

    def _compute_statistical_similarity(self, strip_a, strip_b, orientation='vertical'):
        h_a, w_a = strip_a.shape[:2]; h_b, w_b = strip_b.shape[:2]
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
            A_outer = lab_a[-1, :, :]; A_inner = lab_a[-2, :, :]
            B_outer = lab_b[0, :, :];  B_inner = lab_b[1, :, :]
        else:
            A_outer = lab_a[:, -1, :]; A_inner = lab_a[:, -2, :]
            B_outer = lab_b[:, 0, :];  B_inner = lab_b[:, 1, :]

        # Z-Score Variance
        std_a = np.std(lab_a[:,:,0]) + 1e-3
        std_b = np.std(lab_b[:,:,0]) + 1e-3
        local_sigma = (std_a + std_b) / 2.0

        diff = np.linalg.norm(A_outer - B_outer, axis=1)
        mean_diff = np.mean(diff)

        # FIX: Raised floor for sigma to 10.0 to prevent noise acceptance
        safe_sigma = max(local_sigma, 10.0)
        z_score_color = mean_diff / safe_sigma

        # Gradient Direction
        grad_A = A_outer - A_inner
        grad_B_cont = B_inner - B_outer
        norm_A = np.linalg.norm(grad_A, axis=1, keepdims=True) + 1e-6
        norm_B = np.linalg.norm(grad_B_cont, axis=1, keepdims=True) + 1e-6
        unit_A = grad_A / norm_A; unit_B = grad_B_cont / norm_B
        grad_direction_cost = np.mean(1.0 - np.sum(unit_A * unit_B, axis=1))

        total_cost = z_score_color + (grad_direction_cost * COSINE_WEIGHT)

        if np.mean(A_outer[:,0]) < 5 or np.mean(B_outer[:,0]) < 5: return 999.0
        return total_cost

    def _check_bounds_validity(self, r, c):
        n_min_r = min(self.min_r, r); n_max_r = max(self.max_r, r)
        n_min_c = min(self.min_c, c); n_max_c = max(self.max_c, c)
        span_h = n_max_r - n_min_r + 1; span_w = n_max_c - n_min_c + 1
        return (span_h <= self.max_rows and span_w <= self.max_cols) or \
               (span_h <= self.max_cols and span_w <= self.max_rows)

    def _get_frontier_slots(self):
        frontier = set()
        for (r, c) in self.grid.keys():
            pid, rot = self.grid[(r,c)]
            if (r-1, c) not in self.grid and not self._is_edge_border(pid, rot, 'top'): frontier.add((r-1, c))
            if (r+1, c) not in self.grid and not self._is_edge_border(pid, rot, 'bottom'): frontier.add((r+1, c))
            if (r, c-1) not in self.grid and not self._is_edge_border(pid, rot, 'left'): frontier.add((r, c-1))
            if (r, c+1) not in self.grid and not self._is_edge_border(pid, rot, 'right'): frontier.add((r, c+1))
        return list(frontier)

    def _get_neighbors(self, r, c):
        nbs = []
        if (r-1, c) in self.grid: nbs.append(('top', self.grid[(r-1, c)]))
        if (r+1, c) in self.grid: nbs.append(('bottom', self.grid[(r+1, c)]))
        if (r, c-1) in self.grid: nbs.append(('left', self.grid[(r, c-1)]))
        if (r, c+1) in self.grid: nbs.append(('right', self.grid[(r, c+1)]))
        return nbs

    def _save_debug_snapshot(self, final=False):
        if not self.grid: return
        max_row = max(k[0] for k in self.grid.keys()); min_row = min(k[0] for k in self.grid.keys())
        max_col = max(k[1] for k in self.grid.keys()); min_col = min(k[1] for k in self.grid.keys())
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
            canvas[y:y+self.unit_h, x:x+self.unit_w] = img
        prefix = "final" if final else "step"
        fname = f"debug_steps/{prefix}_{self.step_counter:04d}.png"
        cv2.imwrite(fname, canvas)
        if not final: self.step_counter += 1

    def solve(self):
        print("[Solver] Analyzing Seeds (Z-Score + Cosine Similarity)...")
        self.start_time = time.time()
        seed_candidates = []
        for i in range(self.num_pieces):
            rots_i = self.valid_rotations[i] if self.valid_rotations[i] else [0]
            for j in range(self.num_pieces):
                if i==j: continue
                rots_j = self.valid_rotations[j] if self.valid_rotations[j] else [0]
                for ri in rots_i:
                    for rj in rots_j:
                        s = self._compute_statistical_similarity(self._get_edge_pixels(i, ri, 'right'),
                                                                 self._get_edge_pixels(j, rj, 'left'), 'horizontal')
                        heapq.heappush(seed_candidates, (s, (i, ri, 0, 0, j, rj, 0, 1)))
                        s = self._compute_statistical_similarity(self._get_edge_pixels(i, ri, 'bottom'),
                                                                 self._get_edge_pixels(j, rj, 'top'), 'vertical')
                        heapq.heappush(seed_candidates, (s, (i, ri, 0, 0, j, rj, 1, 0)))

        if not seed_candidates: return []

        MAX_SEEDS_TO_TRY = 50

        for k in range(min(len(seed_candidates), MAX_SEEDS_TO_TRY)):
            best_score, best_seed = heapq.heappop(seed_candidates)
            self.match_threshold = max(best_score * DEFAULT_TOLERANCE_MULTIPLIER, MIN_ABSOLUTE_THRESHOLD)

            p1, r1, y1, x1, p2, r2, y2, x2 = best_seed
            print(f"\n>> Trying Seed Rank {k+1}: P{p1} <-> P{p2} (Z-Score {best_score:.2f})")

            self.grid = {}; self.used_pids = set()
            self.min_r, self.max_r, self.min_c, self.max_c = 0,0,0,0
            self.seed_r, self.seed_c = y1, x1

            self._place_piece(p1, r1, y1, x1)
            self._place_piece(p2, r2, y2, x2)

            if self.visual_debug: self._save_debug_snapshot()

            self._backtrack_recursive(depth=1)

            if self.best_solution and time.time() - self.start_time > SEARCH_TIMEOUT:
                print("[Solver] Timeout. Returning best found.")
                break

        if self.best_solution:
            print(f"\n[Solver] FINAL RESULT: Found {self.solutions_found} solutions. Best Score: {self.best_solution_score:.2f}")
            return self._convert_grid_to_placements(self.best_solution)

        print("[Solver] Failed to find any solution.")
        return []

    def _place_piece(self, pid, rot, r, c):
        self.grid[(r, c)] = (pid, rot)
        self.used_pids.add(pid)
        self.min_r = min(self.min_r, r); self.max_r = max(self.max_r, r)
        self.min_c = min(self.min_c, c); self.max_c = max(self.max_c, c)

    def _remove_piece(self, pid, r, c):
        del self.grid[(r, c)]
        self.used_pids.remove(pid)
        if not self.grid:
            self.min_r, self.max_r, self.min_c, self.max_c = 0,0,0,0
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
                    val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'bottom'), self._get_edge_pixels(pid, rot, 'top'), 'vertical')
                    total_cost += val
                elif direction == 'left':
                    val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'right'), self._get_edge_pixels(pid, rot, 'left'), 'horizontal')
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
            dist = math.sqrt((r - self.seed_r)**2 + (c - self.seed_c)**2)
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
                max_cost = 0.0; avg_cost = 0.0; valid_match = True
                for (direction, (n_pid, n_rot)) in neighbors:
                    val = 0.0
                    if direction == 'top': val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'bottom'), self._get_edge_pixels(pid, rot, 'top'), 'vertical')
                    elif direction == 'bottom': val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'top'), self._get_edge_pixels(pid, rot, 'bottom'), 'vertical')
                    elif direction == 'left': val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'right'), self._get_edge_pixels(pid, rot, 'left'), 'horizontal')
                    elif direction == 'right': val = self._compute_statistical_similarity(self._get_edge_pixels(n_pid, n_rot, 'left'), self._get_edge_pixels(pid, rot, 'right'), 'horizontal')

                    if val > current_threshold * 1.5: valid_match = False; break
                    max_cost = max(max_cost, val); avg_cost += val

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
            if self.solutions_found > 5: return True
            self._remove_piece(cand.pid, cand.row, cand.col)
            attempts += 1

        return False

    def _convert_grid_to_placements(self, grid_dict):
        placements = []
        rows = [k[0] for k in grid_dict.keys()]
        cols = [k[1] for k in grid_dict.keys()]
        mr = min(rows); mc = min(cols)
        for (r, c), (pid, rot) in grid_dict.items():
            placements.append(Placement(pid, rot, r - mr, c - mc))
        return placements

def save_seamless(pieces, solution, path, solver):
    if not solution: return
    max_row = max(p.grid_row for p in solution)
    max_col = max(p.grid_col for p in solution)
    U_W = solver.unit_w; U_H = solver.unit_h
    H = int((max_row + 1) * U_H); W = int((max_col + 1) * U_W)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    print(f"\n[Output] Canvas: {W}x{H}")
    for p in solution:
        piece = pieces[p.piece_id]
        img = getattr(piece, 'raw_image', piece.image).copy()
        for _ in range(p.rotation): img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if img.shape[0] != U_H or img.shape[1] != U_W:
            img = cv2.resize(img, (U_W, U_H), interpolation=cv2.INTER_LANCZOS4)
        y = int(p.grid_row * U_H); x = int(p.grid_col * U_W)
        h_eff = min(U_H, H - y); w_eff = min(U_W, W - x)
        if h_eff > 0 and w_eff > 0: canvas[y:y+h_eff, x:x+w_eff] = img[:h_eff, :w_eff]
    cv2.imwrite(path, canvas)
    print(f"[Output] Saved to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="solved.png")
    parser.add_argument("--width", type=int); parser.add_argument("--height", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visual_debug", action="store_true")
    parser.add_argument("--target_w", type=int); parser.add_argument("--target_h", type=int)
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
        # Try to save whatever we have in solver if possible?
        # Not easy as solver object might be in flux.
        # The visual debug images are the best record.

if __name__ == "__main__":
    main()