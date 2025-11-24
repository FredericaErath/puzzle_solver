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
from preprocess import PuzzlePiece, preprocess_puzzle_image


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
    # Fixed names to match logic usage
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
        self.W = W;
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

    def _precompute_variants(self):
        all_vars = []
        for p in self.pieces:
            p_vars = []
            img = p.image
            if len(p.mask.shape) == 3:
                msk = p.mask[:, :, 0]
            else:
                msk = p.mask
            for r in range(4):
                h, w = img.shape[:2]
                gr = max(1, int(round(h / self.unit_h)))
                gc = max(1, int(round(w / self.unit_w)))

                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                edges = {
                    'top': (img_lab[0, :], msk[0, :]), 'bottom': (img_lab[h - 1, :], msk[h - 1, :]),
                    'left': (img_lab[:, 0], msk[:, 0]), 'right': (img_lab[:, w - 1], msk[:, w - 1])
                }
                # Fixed: pass gr, gc to grid_rows, grid_cols
                p_vars.append(PieceInstance(p.id, r, h, w, gr, gc, edges))
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)
            all_vars.append(p_vars)
        return all_vars

    def _compute_edge_diff(self, edge_a, edge_b):
        pixels_a, mask_a = edge_a
        pixels_b, mask_b = edge_b
        valid = (mask_a > 128) & (mask_b > 128)
        nb_a = pixels_a[:, 0] > 5;
        nb_b = pixels_b[:, 0] > 5
        valid = valid & nb_a & nb_b

        if np.sum(valid) < 3: return 999999.0

        pA = pixels_a[valid];
        pB = pixels_b[valid]
        if self.blur_edges and len(pA) >= 3:
            pA = cv2.GaussianBlur(pA.reshape(-1, 1, 3), (1, 1), 0).reshape(-1, 3)
            pB = cv2.GaussianBlur(pB.reshape(-1, 1, 3), (1, 1), 0).reshape(-1, 3)

        mse = np.mean(np.sum((pA - pB) ** 2, axis=1))
        return float(np.sqrt(mse))

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
                        l = min(len(v1.edges['right'][0]), len(v2.edges['left'][0]))
                        if l > 0:
                            c = self._compute_edge_diff(
                                (v1.edges['right'][0][:l], v1.edges['right'][1][:l]),
                                (v2.edges['left'][0][:l], v2.edges['left'][1][:l])
                            )
                            if c < best_cost:
                                best_cost = c
                                seed = (v1, v2, 0, 0, 0, v1.grid_cols)

                        # Bottom
                        l = min(len(v1.edges['bottom'][0]), len(v2.edges['top'][0]))
                        if l > 0:
                            c = self._compute_edge_diff(
                                (v1.edges['bottom'][0][:l], v1.edges['bottom'][1][:l]),
                                (v2.edges['top'][0][:l], v2.edges['top'][1][:l])
                            )
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
        self.min_r = min(self.min_r, r2);
        self.max_r = max(self.max_r, r2 + v2.grid_rows - 1)
        self.min_c = min(self.min_c, c2);
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

                        total_cost = 0;
                        count = 0;
                        possible = True
                        for direction, neighbor in neighbors:
                            cost = 0
                            if direction == 'top':
                                l = min(len(neighbor.edges['bottom'][0]), len(cand.edges['top'][0]))
                                cost = self._compute_edge_diff(
                                    (neighbor.edges['bottom'][0][:l], neighbor.edges['bottom'][1][:l]),
                                    (cand.edges['top'][0][:l], cand.edges['top'][1][:l]))
                            elif direction == 'bottom':
                                l = min(len(neighbor.edges['top'][0]), len(cand.edges['bottom'][0]))
                                cost = self._compute_edge_diff(
                                    (neighbor.edges['top'][0][:l], neighbor.edges['top'][1][:l]),
                                    (cand.edges['bottom'][0][:l], cand.edges['bottom'][1][:l]))
                            elif direction == 'left':
                                l = min(len(neighbor.edges['right'][0]), len(cand.edges['left'][0]))
                                cost = self._compute_edge_diff(
                                    (neighbor.edges['right'][0][:l], neighbor.edges['right'][1][:l]),
                                    (cand.edges['left'][0][:l], cand.edges['left'][1][:l]))
                            elif direction == 'right':
                                l = min(len(neighbor.edges['left'][0]), len(cand.edges['right'][0]))
                                cost = self._compute_edge_diff(
                                    (neighbor.edges['left'][0][:l], neighbor.edges['left'][1][:l]),
                                    (cand.edges['right'][0][:l], cand.edges['right'][1][:l]))

                            if cost > 500000: possible = False; break
                            total_cost += cost;
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def auto_tune_and_solve(pieces, W, H, args_out, pre_config):
    print("\n[Solver v27] Priority Frontier Strategy...")
    p_type = pre_config.get('type', 'standard_rect')
    config = {'blur': False}
    if p_type == 'rotated_rect': config['blur'] = True

    solver = PriorityFrontierSolver(pieces, W, H, config, debug=True)
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
        h_eff = min(h, final_H - y);
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
    s = int(np.sqrt(area));
    return s, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="solved.png")
    parser.add_argument("--target_w", type=int);
    parser.add_argument("--target_h", type=int)
    parser.add_argument("--width", type=int);
    parser.add_argument("--height", type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pieces, config = preprocess_puzzle_image(args.image, args.width, args.height, debug=False)
    if not pieces: return
    W, H = (args.target_w, args.target_h) if args.target_w else estimate_canvas(pieces)
    auto_tune_and_solve(pieces, W, H, args.out, config)


if __name__ == "__main__":
    main()
