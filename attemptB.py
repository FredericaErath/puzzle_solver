#!/usr/bin/env python3
"""
puzzle_solver_v3.py - Improved Jigsaw Puzzle Solver with Global Optimization

Key improvement: Instead of greedy ordering, try all permutations of rows/columns
and select the one with the highest total edge matching score.

Usage:
python puzzle_solver_v3.py puzzle.png --out solved.png --debug
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from itertools import permutations, combinations
from collections import defaultdict
import cv2
import numpy as np

from preprocess import preprocess_puzzle_image, PuzzlePiece


@dataclass
class Piece:
    id: int
    image: np.ndarray
    mask: np.ndarray
    size: Tuple[int, int]
    edge_lines: List[Dict]


def extract_pieces(path, debug=False):
    data, cfg = preprocess_puzzle_image(path, debug=debug)
    if debug:
        print(f"[Extract] {len(data)} pieces, type={cfg.get('type')}")
    return [Piece(p.id, p.image, p.mask, p.size, p.edge_lines) for p in data], cfg


def get_size(p, rot):
    h, w = p.size
    return (w, h) if rot % 2 else (h, w)


def edge_score(ea, eb):
    pa, ma = ea
    pb, mb = eb
    la, lb = len(ma), len(mb)
    if la < 5 or lb < 5:
        return 0.0
    r = min(la, lb) / max(la, lb)
    if r < 0.6:
        return 0.0
    n = min(la, lb, 80)
    if la != n:
        i = np.linspace(0, la - 1, n).astype(int)
        pa, ma = pa[i], ma[i]
    if lb != n:
        i = np.linspace(0, lb - 1, n).astype(int)
        pb, mb = pb[i], mb[i]
    v = (ma > 200) & (mb > 200)
    if np.sum(v) < 5:
        return 0.0
    d = pa[v].astype(np.float32) - pb[v].astype(np.float32)
    return r / (1 + np.sqrt(np.mean(np.sum(d ** 2, axis=1))) / 30)


def cluster_values(vals, k):
    """K-means clustering for 1D values"""
    if k >= len(vals):
        return sorted(set(vals))
    vals_sorted = sorted(vals)
    c = [vals_sorted[i * len(vals_sorted) // k] for i in range(k)]

    for _ in range(20):
        g = [[] for _ in range(k)]
        for v in vals_sorted:
            g[min(range(k), key=lambda i: abs(v - c[i]))].append(v)
        nc = [int(np.mean(x)) if x else c[i] for i, x in enumerate(g)]
        if nc == c:
            break
        c = nc

    return sorted(c)


def infer_grid_config(pieces, debug=False):
    """Infer the best grid configuration for square output"""
    n = len(pieces)
    area = sum(p.size[0] * p.size[1] for p in pieces)
    target = int(np.sqrt(area))

    if debug:
        print(f"[Grid] {n} pieces, target={target}x{target}")

    all_heights = [p.size[0] for p in pieces]
    all_widths = [p.size[1] for p in pieces]

    best_config = None
    best_score = float('inf')

    for nr in range(2, min(n + 1, 11)):
        if n % nr != 0:
            continue
        nc = n // nr

        row_heights = cluster_values(all_heights, nr)
        col_widths = cluster_values(all_widths, nc)

        sum_h = sum(row_heights)
        sum_w = sum(col_widths)

        aspect_diff = abs(sum_h - sum_w)
        target_diff = abs(sum_h - target) + abs(sum_w - target)

        score = aspect_diff * 2 + target_diff

        if debug:
            print(f"[Grid] {nr}x{nc}: h={row_heights} (sum={sum_h}), w={col_widths} (sum={sum_w}), score={score}")

        if score < best_score:
            best_score = score
            best_config = (nr, nc, row_heights, col_widths)

    return best_config


class GlobalOptimizationSolver:
    """
    Solver that uses global optimization to find the best arrangement.

    Strategy:
    1. Group pieces by size into rows
    2. For each piece, determine its best rotation
    3. Try all permutations of column order within each row
    4. Try all permutations of row order
    5. Select the arrangement with highest total edge score
    """

    def __init__(self, pieces, nr, nc, row_heights, col_widths, debug=False):
        self.pieces = pieces
        self.piece_map = {p.id: p for p in pieces}
        self.nr = nr
        self.nc = nc
        self.row_heights = row_heights
        self.col_widths = col_widths
        self.debug = debug

        # Precompute edge scores for all piece pairs and rotations
        self.edge_cache = self._build_edge_cache()

    def _build_edge_cache(self):
        """Precompute all edge matching scores"""
        cache = {}
        n = len(self.pieces)

        for p1 in self.pieces:
            for r1 in range(4):
                for p2 in self.pieces:
                    if p1.id == p2.id:
                        continue
                    for r2 in range(4):
                        # Right-Left score
                        rl = edge_score(
                            p1.edge_lines[r1]['right'],
                            p2.edge_lines[r2]['left']
                        )
                        cache[(p1.id, r1, 'R', p2.id, r2)] = rl

                        # Bottom-Top score
                        bt = edge_score(
                            p1.edge_lines[r1]['bottom'],
                            p2.edge_lines[r2]['top']
                        )
                        cache[(p1.id, r1, 'B', p2.id, r2)] = bt

        return cache

    def _assign_to_rows(self):
        """Assign pieces to rows based on height"""
        rows = {h: [] for h in self.row_heights}

        for p in self.pieces:
            # Find best rotation and row assignment
            best_row = None
            best_rot = 0
            best_diff = float('inf')

            for rot in range(4):
                h, w = get_size(p, rot)
                for rh in self.row_heights:
                    diff = abs(h - rh)
                    if diff < best_diff:
                        best_diff = diff
                        best_row = rh
                        best_rot = rot

            rows[best_row].append((p.id, best_rot))

        return rows

    def _get_best_rotation_for_cell(self, pid, target_h, target_w):
        """Find the rotation that best fits the cell"""
        p = self.piece_map[pid]
        best_rot = 0
        best_diff = float('inf')

        for rot in range(4):
            h, w = get_size(p, rot)
            diff = abs(h - target_h) + abs(w - target_w)
            if diff < best_diff:
                best_diff = diff
                best_rot = rot

        return best_rot

    def _score_arrangement(self, grid):
        """
        Score a complete arrangement.
        grid[r][c] = (pid, rot)
        """
        total = 0.0

        for r in range(self.nr):
            for c in range(self.nc):
                pid, rot = grid[r][c]

                # Right neighbor
                if c + 1 < self.nc:
                    npid, nrot = grid[r][c + 1]
                    total += self.edge_cache.get((pid, rot, 'R', npid, nrot), 0)

                # Bottom neighbor
                if r + 1 < self.nr:
                    npid, nrot = grid[r + 1][c]
                    total += self.edge_cache.get((pid, rot, 'B', npid, nrot), 0)

        return total

    def _try_all_column_orders(self, row_pieces):
        """
        Try all permutations of pieces within a row.
        Returns list of (score, ordered_pieces) sorted by score.
        """
        if len(row_pieces) <= 1:
            return [(0, row_pieces)]

        results = []

        # For each permutation of the row
        for perm in permutations(row_pieces):
            # Calculate horizontal edge score
            score = 0
            for i in range(len(perm) - 1):
                pid1, rot1 = perm[i]
                pid2, rot2 = perm[i + 1]
                score += self.edge_cache.get((pid1, rot1, 'R', pid2, rot2), 0)
            results.append((score, list(perm)))

        results.sort(key=lambda x: -x[0])
        return results

    def _score_row_order(self, ordered_rows, row_order):
        """Score a particular ordering of rows"""
        total = 0

        for i in range(len(row_order) - 1):
            rh1, rh2 = row_order[i], row_order[i + 1]
            row1 = ordered_rows[rh1]
            row2 = ordered_rows[rh2]

            # Sum vertical edge scores
            for c in range(min(len(row1), len(row2))):
                pid1, rot1 = row1[c]
                pid2, rot2 = row2[c]
                total += self.edge_cache.get((pid1, rot1, 'B', pid2, rot2), 0)

        return total

    def solve(self):
        """Main solving method with global optimization"""

        # Step 1: Assign pieces to rows
        rows = self._assign_to_rows()

        if self.debug:
            print(f"[Solver] Row assignments:")
            for rh in self.row_heights:
                print(f"  Row h={rh}: {len(rows[rh])} pieces")

        # Step 2: Find best column order for each row (try all permutations)
        best_row_orders = {}
        for rh in self.row_heights:
            row_pieces = rows[rh]

            # Get top N best orderings for this row
            orderings = self._try_all_column_orders(row_pieces)
            best_row_orders[rh] = orderings[:min(10, len(orderings))]  # Keep top 10

            if self.debug:
                best_score, best_order = orderings[0]
                print(f"  Row h={rh}: best col score={best_score:.3f}, order={[p[0] for p in best_order]}")

        # Step 3: Try all row permutations with multiple column orderings
        best_grid = None
        best_total_score = -1

        row_perms = list(permutations(self.row_heights))

        if self.debug:
            print(f"[Solver] Trying {len(row_perms)} row permutations with multiple column orders...")

        # For more thorough search, try multiple column orderings
        # Generate all combinations of top-3 column orderings for each row
        from itertools import product

        # Get top-3 column orderings for each row
        top_k = min(3, len(list(permutations(range(self.nc)))))
        col_order_options = {}
        for rh in self.row_heights:
            col_order_options[rh] = [order for _, order in best_row_orders[rh][:top_k]]

        total_combinations = 0
        for row_perm in row_perms:
            # Try all combinations of column orderings
            col_options_list = [col_order_options[rh] for rh in row_perm]

            for col_combo in product(*col_options_list):
                total_combinations += 1

                # Build grid
                grid = list(col_combo)

                # Score this arrangement
                score = self._score_arrangement(grid)

                if score > best_total_score:
                    best_total_score = score
                    best_grid = [row[:] for row in grid]
                    best_row_order = row_perm

        if self.debug:
            print(f"[Solver] Tried {total_combinations} combinations")
            print(f"[Solver] Best score: {best_total_score:.3f}")
            print(f"[Solver] Best row order: {best_row_order}")

        # Step 4: Try more column order combinations for the best row order
        if best_grid is not None:
            best_grid, best_total_score = self._refine_solution(
                best_grid, best_row_order, best_row_orders, best_total_score
            )

        if best_grid is None:
            return None

        # Convert grid to solution format
        return self._build_solution(best_grid, best_row_order)

    def _refine_solution(self, grid, row_order, best_row_orders, current_best_score):
        """Try more column combinations for the best row order"""

        # For each row, try alternative column orderings
        improved = True
        iterations = 0
        max_iterations = 50

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for row_idx, rh in enumerate(row_order):
                # Try each alternative column ordering for this row
                for _, col_order in best_row_orders[rh][1:]:
                    # Create new grid with this column order
                    new_grid = [row[:] for row in grid]
                    new_grid[row_idx] = col_order

                    score = self._score_arrangement(new_grid)

                    if score > current_best_score:
                        current_best_score = score
                        grid = new_grid
                        improved = True
                        break

                if improved:
                    break

        if self.debug and iterations > 1:
            print(f"[Solver] Refined in {iterations} iterations, final score: {current_best_score:.3f}")

        return grid, current_best_score

    def _build_solution(self, grid, row_order):
        """Convert grid to solution format"""
        solution = []
        y = 0

        for row_idx, rh in enumerate(row_order):
            x = 0
            row = grid[row_idx]

            for col_idx, (pid, rot) in enumerate(row):
                p = self.piece_map[pid]
                h, w = get_size(p, rot)

                # Use actual piece size, not cell size
                cw = self.col_widths[col_idx] if col_idx < len(self.col_widths) else w

                solution.append({
                    'id': pid,
                    'rot': rot,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
                x += w
            y += rh

        return solution


def save_result(pieces, sol, path):
    if not sol:
        print("[Error] No solution to save")
        return

    my = max(p['y'] + p['h'] for p in sol)
    mx = max(p['x'] + p['w'] for p in sol)

    canvas = np.zeros((my, mx, 3), dtype=np.uint8)
    piece_map = {p.id: p for p in pieces}

    for p in sol:
        piece = piece_map[p['id']]
        img, msk = piece.image.copy(), piece.mask.copy()

        for _ in range(p['rot']):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

        ph, pw = img.shape[:2]
        y, x = p['y'], p['x']

        h = min(ph, my - y)
        w = min(pw, mx - x)

        if h > 0 and w > 0 and y >= 0 and x >= 0:
            roi = canvas[y:y + h, x:x + w]
            v = msk[:h, :w] > 128
            roi[v] = img[:h, :w][v]

    cv2.imwrite(path, canvas)
    print(f"[Output] {path} ({mx}x{my})")

    with open(path.replace('.png', '.json'), 'w') as f:
        json.dump(sol, f, indent=2)


def solve_irregular_from_pieces(
    raw_pieces: List[PuzzlePiece],
    out_path: str,
    debug: bool = False,
) -> None:
    """
    Wrapper 给 solve_puzzle.py 调用：
    - 直接使用外部 preprocess 得到的 PuzzlePiece 列表
    - 在本文件中转换为 Piece，推断 grid 配置，调用 GlobalOptimizationSolver
    - 最终通过 save_result 写出 PNG + 同名 JSON

    参数:
      raw_pieces : preprocess_puzzle_image 返回的 PuzzlePiece 列表
      out_path   : 输出图片路径 (建议以 .png 结尾，这样 save_result 的 .json 替换逻辑才正常)
      debug      : 是否打印调试信息
    """
    if not raw_pieces:
        print("[Wrapper] No pieces to solve (empty list).")
        return

    # 1) 把外部的 PuzzlePiece 转成本文件使用的 Piece
    pieces = [
        Piece(
            id=p.id,
            image=p.image,
            mask=p.mask,
            size=p.size,
            edge_lines=p.edge_lines,
        )
        for p in raw_pieces
    ]

    print(f"[Wrapper] Solving irregular grid with {len(pieces)} pieces -> {out_path}")

    # 2) 推断 grid 配置
    config = infer_grid_config(pieces, debug=debug)
    if config is None:
        print("[Wrapper] Could not infer grid configuration.")
        return

    nr, nc, row_heights, col_widths = config
    if debug:
        print(f"[Wrapper] Grid: {nr}x{nc}")

    # 3) Global optimization 求解
    solver = GlobalOptimizationSolver(
        pieces,
        nr,
        nc,
        row_heights,
        col_widths,
        debug=debug,
    )
    sol = solver.solve()

    # 4) 保存结果
    if sol and len(sol) == len(pieces):
        save_result(pieces, sol, out_path)
        print("[Wrapper] Done.")
    else:
        print("[Wrapper] No solution found or incomplete solution.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", help="Puzzle image with scattered pieces")
    ap.add_argument("--out", default="solved.png")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print(f"[Main] Loading: {args.puzzle}")
    pieces, cfg = extract_pieces(args.puzzle, debug=args.debug)
    if not pieces:
        print("[Error] No pieces found")
        return 1

    print(f"[Main] {len(pieces)} pieces")

    # Infer grid configuration
    config = infer_grid_config(pieces, debug=args.debug)
    if config is None:
        print("[Error] Could not infer grid configuration")
        return 1

    nr, nc, row_heights, col_widths = config
    print(f"[Main] Grid: {nr}x{nc}")

    # Solve with global optimization
    solver = GlobalOptimizationSolver(pieces, nr, nc, row_heights, col_widths, debug=args.debug)
    sol = solver.solve()

    if sol and len(sol) == len(pieces):
        save_result(pieces, sol, args.out)
        print("[Main] Done!")
        return 0

    print("[Error] No solution found")
    return 1


if __name__ == "__main__":
    exit(main())