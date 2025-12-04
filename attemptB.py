#!/usr/bin/env python3
"""
puzzle_solver_v6.py - Robust Puzzle Solver

Key improvements:
1. Better edge matching using multiple features (color profile, gradient, histogram)
2. Seamless output without black gaps - proper piece scaling and placement
3. Robust grid detection that handles various piece size combinations
4. Improved solver with better search strategy

Usage:
python puzzle_solver_v6.py input.png --out solved.png --debug
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from itertools import combinations_with_replacement, permutations, product
from collections import Counter, defaultdict
import cv2
import os
import numpy as np

# Import preprocess functions
from preprocess import preprocess_puzzle_image, PuzzlePiece as PreprocessPiece


@dataclass
class Piece:
    id: int
    image: np.ndarray
    mask: np.ndarray
    size: Tuple[int, int]  # (height, width)
    edge_lines: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]  # For all 4 rotations


@dataclass
class Placement:
    piece_id: int
    rotation: int  # 0, 1, 2, 3 for 0, 90, 180, 270 degrees
    row: int
    col: int
    y: int
    x: int
    h: int
    w: int


def extract_pieces_with_preprocess(image_path, debug=False):
    """Extract pieces using preprocess.py"""
    pieces_data, config = preprocess_puzzle_image(image_path, debug=debug)

    if debug:
        print(f"[Extract] Found {len(pieces_data)} pieces")
        print(f"[Extract] Config: {config}")

    pieces = []
    for p in pieces_data:
        pieces.append(Piece(
            id=p.id,
            image=p.image,
            mask=p.mask,
            size=p.size,
            edge_lines=p.edge_lines
        ))

    if debug:
        for p in pieces[:5]:
            print(f"   P{p.id}: size={p.size}")

    return pieces, config


def sizes_match(s1, s2, tol=8):
    """Check if two sizes match within tolerance"""
    return abs(s1[0] - s2[0]) <= tol and abs(s1[1] - s2[1]) <= tol


def cluster_dimensions(dims, tol=8):
    """Cluster similar dimensions together and return cluster representatives"""
    if not dims:
        return []
    dims = sorted(dims)
    clusters = []
    current_cluster = [dims[0]]

    for d in dims[1:]:
        if d - current_cluster[0] <= tol:  # Compare with cluster start
            current_cluster.append(d)
        else:
            clusters.append(current_cluster)
            current_cluster = [d]
    clusters.append(current_cluster)

    # Return representative value for each cluster (most common, or median)
    result = []
    for c in clusters:
        counter = Counter(c)
        most_common = counter.most_common(1)[0][0]
        result.append(most_common)
    return result


def analyze_piece_dimensions(pieces, debug=False):
    """Analyze piece dimensions to find row/column patterns"""
    # Collect all dimensions
    heights = []
    widths = []

    for p in pieces:
        heights.append(p.size[0])
        widths.append(p.size[1])

    # Count occurrences
    h_counter = Counter(heights)
    w_counter = Counter(widths)

    if debug:
        print(f"[Analyze] Height counts: {dict(h_counter.most_common(10))}")
        print(f"[Analyze] Width counts: {dict(w_counter.most_common(10))}")

    # Cluster dimensions
    unique_heights = cluster_dimensions(heights, tol=5)
    unique_widths = cluster_dimensions(widths, tol=5)

    if debug:
        print(f"[Analyze] Unique heights: {unique_heights}")
        print(f"[Analyze] Unique widths: {unique_widths}")

    return unique_heights, unique_widths, h_counter, w_counter


def find_grid_structure_v3(pieces, debug=False):
    """
    Robust grid detection algorithm.

    Strategy:
    1. Analyze piece dimensions to find natural clusters
    2. For each possible grid configuration (n_rows x n_cols)
    3. Try to match pieces to grid cells considering rotations
    """
    n = len(pieces)

    # Total area for target size estimation
    total_area = sum(p.size[0] * p.size[1] for p in pieces)
    target_side = int(np.sqrt(total_area) + 0.5)

    if debug:
        print(f"[Grid] {n} pieces, total area: {total_area}, target side: {target_side}")

    # Analyze dimensions
    unique_h, unique_w, h_counter, w_counter = analyze_piece_dimensions(pieces, debug)

    # All possible dimensions (considering rotation)
    all_dims = set()
    for p in pieces:
        all_dims.add(p.size[0])
        all_dims.add(p.size[1])
    all_dims = sorted(all_dims)

    if debug:
        print(f"[Grid] All unique dimensions: {all_dims}")

    # Try different grid configurations
    best_grid = None
    best_score = float('inf')

    for n_rows in range(1, n + 1):
        if n % n_rows != 0:
            continue
        n_cols = n // n_rows

        if debug:
            print(f"[Grid] Trying {n_rows}x{n_cols}...")

        # Try to find valid row heights and column widths
        result = find_valid_grid_assignment(pieces, n_rows, n_cols, all_dims, target_side, debug)

        if result is not None:
            row_heights, col_widths, score = result
            if debug:
                print(f"[Grid] Found valid: rows={row_heights}, cols={col_widths}, score={score:.2f}")

            if score < best_score:
                best_score = score
                best_grid = (row_heights, col_widths)

    if best_grid is None:
        if debug:
            print("[Grid] No valid grid structure found")
        return None

    if debug:
        print(f"[Grid] Best grid: rows={best_grid[0]}, cols={best_grid[1]}")

    return best_grid


def find_valid_grid_assignment(pieces, n_rows, n_cols, all_dims, target_side, debug=False):
    """
    Find valid row heights and column widths for given grid configuration.
    Uses constraint satisfaction approach.
    """
    n = len(pieces)
    tol = 8

    # Build piece size lookup (both orientations)
    piece_sizes = []
    for p in pieces:
        piece_sizes.append({
            'id': p.id,
            'sizes': [(p.size[0], p.size[1]), (p.size[1], p.size[0])]
        })

    # Collect all possible heights and widths
    possible_heights = set()
    possible_widths = set()
    for p in pieces:
        possible_heights.add(p.size[0])
        possible_heights.add(p.size[1])
        possible_widths.add(p.size[0])
        possible_widths.add(p.size[1])

    possible_heights = sorted(possible_heights)
    possible_widths = sorted(possible_widths)

    # Try combinations with smart pruning
    best_result = None
    best_score = float('inf')
    search_count = 0
    max_search = 100000

    # Generate row height combinations
    for row_heights in combinations_with_replacement(possible_heights, n_rows):
        row_sum = sum(row_heights)

        # Prune: row sum should be reasonable
        if row_sum < target_side * 0.5 or row_sum > target_side * 1.5:
            continue

        for col_widths in combinations_with_replacement(possible_widths, n_cols):
            col_sum = sum(col_widths)

            # Prune: col sum should be reasonable
            if col_sum < target_side * 0.5 or col_sum > target_side * 1.5:
                continue

            # Prune: should be roughly square
            if abs(row_sum - col_sum) > target_side * 0.3:
                continue

            search_count += 1
            if search_count > max_search:
                break

            # Check if this grid can be filled
            row_list = list(row_heights)
            col_list = list(col_widths)

            if can_fill_grid_exact(row_list, col_list, pieces, tol):
                # Score based on how square it is
                score = abs(row_sum - col_sum)
                if score < best_score:
                    best_score = score
                    best_result = (row_list, col_list, score)

        if search_count > max_search:
            break

    return best_result


def can_fill_grid_exact(rows, cols, pieces, tol=8):
    """
    Check if grid can be filled with pieces exactly.
    Uses bipartite matching approach.
    """
    n_cells = len(rows) * len(cols)
    if n_cells != len(pieces):
        return False

    # Build list of required cell sizes
    needed = []
    for r in rows:
        for c in cols:
            needed.append((r, c))

    # Build list of available piece sizes (with rotation options)
    available = []
    for p in pieces:
        available.append({
            'id': p.id,
            'sizes': [(p.size[0], p.size[1]), (p.size[1], p.size[0])]
        })

    # Try to match using greedy assignment
    used = [False] * len(available)

    for target_h, target_w in needed:
        found = False
        for i, piece_info in enumerate(available):
            if used[i]:
                continue

            for ph, pw in piece_info['sizes']:
                if abs(ph - target_h) <= tol and abs(pw - target_w) <= tol:
                    used[i] = True
                    found = True
                    break

            if found:
                break

        if not found:
            return False

    return all(used)


def compute_edge_score_v2(edge_a, edge_b):
    """
    Improved edge matching score using multiple features.
    Lower score = better match.
    """
    pixels_a, mask_a = edge_a
    pixels_b, mask_b = edge_b

    len_a, len_b = len(mask_a), len(mask_b)

    if len_a < 3 or len_b < 3:
        return 999999.0

    # Resample to same length
    target_len = min(len_a, len_b, 50)

    def resample(pixels, mask, target):
        if len(mask) == target:
            return pixels.copy(), mask.copy()
        indices = np.linspace(0, len(mask) - 1, target).astype(int)
        return pixels[indices].copy(), mask[indices].copy()

    pa, ma = resample(pixels_a, mask_a, target_len)
    pb, mb = resample(pixels_b, mask_b, target_len)

    # Valid mask
    valid = (ma > 200) & (mb > 200)
    n_valid = np.sum(valid)

    if n_valid < 3:
        return 999999.0

    pa = pa.astype(np.float32)
    pb = pb.astype(np.float32)

    # 1. Direct color difference (LAB space)
    pa_valid = pa[valid]
    pb_valid = pb[valid]

    color_diff = np.mean(np.sqrt(np.sum((pa_valid - pb_valid) ** 2, axis=1)))

    # 2. Gradient continuity - check if gradients align
    def compute_gradient(arr):
        if len(arr) < 2:
            return np.zeros_like(arr)
        grad = np.diff(arr, axis=0, prepend=arr[:1])
        return grad

    grad_a = compute_gradient(pa_valid)
    grad_b = compute_gradient(pb_valid)
    grad_diff = np.mean(np.abs(grad_a - grad_b))

    # 3. Histogram similarity
    def compute_hist(pixels):
        hist = np.zeros(24)
        for ch in range(3):
            h, _ = np.histogram(pixels[:, ch], bins=8, range=(0, 256))
            hist[ch * 8:(ch + 1) * 8] = h / (len(pixels) + 1e-8)
        return hist

    hist_a = compute_hist(pa_valid)
    hist_b = compute_hist(pb_valid)
    hist_diff = np.sum(np.abs(hist_a - hist_b))

    # Combine scores with weights
    score = color_diff * 1.0 + grad_diff * 0.3 + hist_diff * 5.0

    return score


class GridSolver:
    """Solve puzzle by filling grid with pieces using improved matching"""

    def __init__(self, pieces, row_heights, col_widths, debug=False):
        self.pieces = pieces
        self.row_heights = row_heights
        self.col_widths = col_widths
        self.n_rows = len(row_heights)
        self.n_cols = len(col_widths)
        self.debug = debug
        self.tol = 8

        self.grid = [[None] * self.n_cols for _ in range(self.n_rows)]
        self.used = [False] * len(pieces)
        self.best_solution = None
        self.best_score = float('inf')
        self.iterations = 0
        self.max_iterations = 1000000

        # Precompute which pieces can fit in which cells
        self._precompute_candidates()

    def _precompute_candidates(self):
        """Precompute which pieces can fit in each cell"""
        self.cell_candidates = {}

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                target_h = self.row_heights[r]
                target_w = self.col_widths[c]
                candidates = []

                for pid, p in enumerate(self.pieces):
                    for rot in range(4):
                        ph, pw = self._get_piece_size(pid, rot)
                        if sizes_match((ph, pw), (target_h, target_w), self.tol):
                            candidates.append((pid, rot))

                self.cell_candidates[(r, c)] = candidates

                if self.debug and len(candidates) == 0:
                    print(f"[Warning] No candidates for cell ({r},{c}) size=({target_h},{target_w})")

    def _get_piece_size(self, pid, rot):
        """Get piece size for given rotation"""
        h, w = self.pieces[pid].size
        if rot % 2 == 1:  # 90 or 270 degrees
            return (w, h)
        return (h, w)

    def _get_candidates(self, r, c):
        """Get candidate pieces for cell (r, c) sorted by edge matching score"""
        candidates = []

        for pid, rot in self.cell_candidates[(r, c)]:
            if self.used[pid]:
                continue

            p = self.pieces[pid]
            score = 0.0
            count = 0

            # Top neighbor
            if r > 0 and self.grid[r - 1][c] is not None:
                npid, nrot = self.grid[r - 1][c]
                my_top = p.edge_lines[rot]['top']
                their_bottom = self.pieces[npid].edge_lines[nrot]['bottom']
                score += compute_edge_score_v2(my_top, their_bottom)
                count += 1

            # Left neighbor
            if c > 0 and self.grid[r][c - 1] is not None:
                npid, nrot = self.grid[r][c - 1]
                my_left = p.edge_lines[rot]['left']
                their_right = self.pieces[npid].edge_lines[nrot]['right']
                score += compute_edge_score_v2(my_left, their_right)
                count += 1

            if count > 0:
                score /= count

            candidates.append((pid, rot, score))

        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x[2])
        return candidates

    def _backtrack(self, cell_idx, current_score=0):
        """Backtracking search with pruning"""
        self.iterations += 1

        if self.iterations > self.max_iterations:
            return self.best_solution is not None

        if cell_idx >= self.n_rows * self.n_cols:
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_solution = [[self.grid[r][c] for c in range(self.n_cols)]
                                      for r in range(self.n_rows)]
            return True

        # Pruning
        if self.best_solution is not None and current_score > self.best_score * 2:
            return False

        r = cell_idx // self.n_cols
        c = cell_idx % self.n_cols

        candidates = self._get_candidates(r, c)

        if not candidates:
            return False

        # Limit candidates to try based on position
        if cell_idx < 4:
            max_candidates = min(len(candidates), 20)
        else:
            max_candidates = min(len(candidates), 8)

        found_any = False
        for pid, rot, score in candidates[:max_candidates]:
            self.used[pid] = True
            self.grid[r][c] = (pid, rot)

            if self._backtrack(cell_idx + 1, current_score + score):
                found_any = True

            self.used[pid] = False
            self.grid[r][c] = None

            # Early exit if we found a good solution
            if found_any and self.best_score < 100:
                break

        return found_any

    def solve(self):
        """Solve the puzzle"""
        if self.debug:
            print(f"[Solve] Grid: {self.n_rows}x{self.n_cols}")
            print(f"[Solve] Row heights: {self.row_heights}")
            print(f"[Solve] Col widths: {self.col_widths}")

        # Check if all cells have candidates
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if not self.cell_candidates[(r, c)]:
                    if self.debug:
                        print(f"[Solve] Cell ({r},{c}) has no candidates!")
                    return None

        self._backtrack(0)

        if self.debug:
            print(f"[Solve] Iterations: {self.iterations}")

        if self.best_solution is None:
            return None

        if self.debug:
            print(f"[Solve] Best score: {self.best_score:.2f}")

        # Build placements from best solution
        placements = []
        y = 0
        for r in range(self.n_rows):
            x = 0
            for c in range(self.n_cols):
                pid, rot = self.best_solution[r][c]
                # Use actual grid cell size, not piece size
                h = self.row_heights[r]
                w = self.col_widths[c]

                placements.append(Placement(
                    piece_id=pid,
                    rotation=rot,
                    row=r,
                    col=c,
                    y=y,
                    x=x,
                    h=h,
                    w=w
                ))
                x += w
            y += self.row_heights[r]

        return placements


def save_result_seamless(pieces, solution, row_heights, col_widths, path):
    """
    Save result image ensuring seamless output without gaps.
    Key: use exact grid dimensions, scale pieces to fit cells exactly.
    """
    total_h = sum(row_heights)
    total_w = sum(col_widths)

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    for p in solution:
        piece = pieces[p.piece_id]
        img = piece.image.copy()
        msk = piece.mask.copy()

        # Apply rotation
        for _ in range(p.rotation):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

        # Target cell size
        target_h = p.h
        target_w = p.w

        # Current piece size
        piece_h, piece_w = img.shape[:2]

        # Scale piece to exactly fit cell
        if piece_h != target_h or piece_w != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Place on canvas (directly, no mask filtering to avoid gaps)
        y, x = p.y, p.x

        # Ensure bounds
        h_place = min(target_h, total_h - y)
        w_place = min(target_w, total_w - x)

        if h_place > 0 and w_place > 0:
            # Direct placement without mask (pieces should fill completely)
            canvas[y:y + h_place, x:x + w_place] = img[:h_place, :w_place]

    cv2.imwrite(path, canvas)
    print(f"[Output] Saved to {path}")
    print(f"[Output] Size: {total_w}x{total_h}")

    # Save JSON
    json_data = [{"id": p.piece_id, "r": p.rotation, "row": p.row, "col": p.col,
                  "x": p.x, "y": p.y, "h": p.h, "w": p.w} for p in solution]
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="solved.png")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"[Main] Loading: {args.image}")

    # Extract pieces using preprocess
    pieces, config = extract_pieces_with_preprocess(args.image, debug=args.debug)
    if not pieces:
        print("[Error] No pieces extracted")
        return 1

    print(f"[Main] {len(pieces)} pieces extracted")

    # Find grid structure
    grid = find_grid_structure_v3(pieces, debug=True)
    if not grid:
        print("[Error] Cannot find valid grid structure")
        return 1

    row_heights, col_widths = grid

    # Solve
    solver = GridSolver(pieces, row_heights, col_widths, debug=True)
    solution = solver.solve()

    if solution:
        save_result_seamless(pieces, solution, row_heights, col_widths, args.out)
        print("[Main] Success!")
        return 0
    else:
        print("[Error] No solution found")
        return 1


if __name__ == "__main__":
    exit(main())