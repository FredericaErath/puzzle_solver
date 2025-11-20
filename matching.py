"""
matching.py

Global grid-based puzzle solver (no rotation) for the
Computational Image Puzzle Solver.

This file assumes you already have preprocess.py producing
a list of PuzzlePiece objects with per-edge features:
  - mean_color  : (B,G,R)
  - color_hist  : concatenated histograms for B/G/R
  - color_profile: 1D BGR profile along each edge
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from preprocess import PuzzlePiece, preprocess_puzzle_image


# ---------------------------------------------------------------------
# Edge distance
# ---------------------------------------------------------------------

# A very large but finite penalty used instead of +inf
BAD_COST = 1e6


def edge_distance(
    edge_a,
    edge_b,
    max_len_diff_ratio: float = 0.12,
    mean_color_gate: Optional[float] = 80.0,
    alpha_profile: float = 0.2,
) -> float:
    """
    Distance between two edges.

    Uses:
      - L2 distance between colour histograms
      - profile MSE along the edge (with edge_b reversed)
      - an extra large penalty when mean colours are very different

    We deliberately avoid returning +inf (except for crazy length
    mismatches), so that greedy initialisation always has a finite cost.
    """
    prof_a = edge_a.color_profile.astype(np.float32)
    prof_b = edge_b.color_profile.astype(np.float32)

    len_a = prof_a.shape[0]
    len_b = prof_b.shape[0]
    max_len = max(len_a, len_b)
    if max_len == 0:
        # Degenerate edge, treat as very bad but finite
        return BAD_COST

    # Length gate – if lengths are wildly different, treat as impossible
    rel_diff = abs(len_a - len_b) / max_len
    if rel_diff > max_len_diff_ratio:
        return BAD_COST

    # Histogram distance
    hist_a = edge_a.color_hist.astype(np.float32)
    hist_b = edge_b.color_hist.astype(np.float32)
    hist_dist = float(np.linalg.norm(hist_a - hist_b))

    # Profile MSE (reverse second profile to align the boundary)
    prof_b_rev = prof_b[::-1, :]
    L = min(len_a, len_b)
    if L > 0:
        diff = prof_a[:L] - prof_b_rev[:L]
        mse = float(np.mean(diff * diff))
    else:
        mse = 0.0

    # Optional extra penalty if mean colours are very different
    penalty = 0.0
    if mean_color_gate is not None:
        mc_a = np.array(edge_a.mean_color, dtype=np.float32)
        mc_b = np.array(edge_b.mean_color, dtype=np.float32)
        mean_diff = float(np.linalg.norm(mc_a - mc_b))
        if mean_diff > mean_color_gate:
            # big but finite penalty
            penalty = BAD_COST * 0.1 + mean_diff

    cost = hist_dist + alpha_profile * mse + penalty
    return cost


# ---------------------------------------------------------------------
# Grid solver
# ---------------------------------------------------------------------


@dataclass
class GridSolution:
    rows: int
    cols: int
    assignment: List[int]  # r*cols + c -> piece_id
    total_cost: float


class GridSolver:
    """
    Global optimiser for placing N pieces into an R x C grid (R*C == N),
    using a branch-and-bound search.

    Assumptions:
      - all pieces are axis-aligned, no rotation
      - grid is regular (e.g., 4x4)
      - each piece is used exactly once
    """

    def __init__(
        self,
        pieces: List[PuzzlePiece],
        max_len_diff_ratio: float = 0.12,
        mean_color_gate: Optional[float] = 80.0,
        alpha_profile: float = 0.2,
        debug: bool = False,
    ):
        self.pieces = pieces
        self.N = len(pieces)
        self.debug = debug

        self.rows, self.cols = self._infer_grid_dims(self.N)
        if self.rows * self.cols != self.N:
            raise ValueError(
                f"Cannot form a grid from N={self.N} pieces "
                f"(rows={self.rows}, cols={self.cols})"
            )

        if self.debug:
            print(f"[GridSolver] Using grid {self.rows} x {self.cols} for {self.N} pieces.")

        self.max_len_diff_ratio = max_len_diff_ratio
        self.mean_color_gate = mean_color_gate
        self.alpha_profile = alpha_profile

        # Precompute adjacency costs:
        #   horiz_cost[i, j] = cost of j to the RIGHT of i
        #   vert_cost[i, j]  = cost of j BELOW i
        self.horiz_cost = np.zeros((self.N, self.N), dtype=np.float32)
        self.vert_cost = np.zeros((self.N, self.N), dtype=np.float32)
        self._build_cost_tables()

        # Branch-and-bound state
        self.best_cost = float("inf")
        self.best_assignment: Optional[List[int]] = None
        self.node_count = 0

    # ---------------- internal helpers ---------------------

    def _infer_grid_dims(self, N: int) -> Tuple[int, int]:
        """Pick a (rows, cols) factorisation of N that is as square as possible."""
        best_r, best_c = 1, N
        best_diff = N - 1
        for r in range(1, N + 1):
            if N % r != 0:
                continue
            c = N // r
            diff = abs(r - c)
            if diff < best_diff:
                best_diff = diff
                best_r, best_c = r, c
        return best_r, best_c

    def _build_cost_tables(self) -> None:
        N = self.N
        if self.debug:
            print("[GridSolver] Precomputing pairwise edge costs ...")

        for i in range(N):
            pi = self.pieces[i]
            e_i_right = pi.edges["right"]
            e_i_bottom = pi.edges["bottom"]

            for j in range(N):
                pj = self.pieces[j]
                if i == j:
                    # self-adjacency is never used, but give a big cost anyway
                    self.horiz_cost[i, j] = BAD_COST
                    self.vert_cost[i, j] = BAD_COST
                    continue

                e_j_left = pj.edges["left"]
                e_j_top = pj.edges["top"]

                self.horiz_cost[i, j] = edge_distance(
                    e_i_right,
                    e_j_left,
                    max_len_diff_ratio=self.max_len_diff_ratio,
                    mean_color_gate=self.mean_color_gate,
                    alpha_profile=self.alpha_profile,
                )

                self.vert_cost[i, j] = edge_distance(
                    e_i_bottom,
                    e_j_top,
                    max_len_diff_ratio=self.max_len_diff_ratio,
                    mean_color_gate=self.mean_color_gate,
                    alpha_profile=self.alpha_profile,
                )

        if self.debug:
            print("[GridSolver] horiz_cost sample:",
                  float(self.horiz_cost.min()), float(self.horiz_cost.max()))
            print("[GridSolver] vert_cost  sample:",
                  float(self.vert_cost.min()), float(self.vert_cost.max()))

    # ------------------------------------------------------

    def _incremental_cost(
        self,
        assignment: List[Optional[int]],
        pos: int,
        pid: int,
    ) -> float:
        """
        Cost added by placing piece `pid` into cell `pos`, considering only
        left and upper neighbours (the ones already decided).
        """
        r = pos // self.cols
        c = pos % self.cols
        inc = 0.0

        # left neighbour
        if c > 0:
            left_pid = assignment[pos - 1]
            if left_pid is not None:
                inc += float(self.horiz_cost[left_pid, pid])

        # upper neighbour
        if r > 0:
            up_pid = assignment[pos - self.cols]
            if up_pid is not None:
                inc += float(self.vert_cost[up_pid, pid])

        return inc

    def _full_cost(self, assignment: List[int]) -> float:
        """
        Compute total cost of a full assignment (all grid cells filled).
        """
        total = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                pid = assignment[idx]

                # right neighbour
                if c + 1 < self.cols:
                    right_pid = assignment[idx + 1]
                    total += float(self.horiz_cost[pid, right_pid])

                # down neighbour
                if r + 1 < self.rows:
                    down_pid = assignment[idx + self.cols]
                    total += float(self.vert_cost[pid, down_pid])
        return total

    # ------------------------------------------------------

    def _greedy_initial(self) -> None:
        """
        Build a quick greedy solution to get an initial upper bound.
        """
        N = self.N
        assignment: List[Optional[int]] = [None] * N
        used = [False] * N

        for pos in range(N):
            best_pid = None
            best_inc = float("inf")

            for pid in range(N):
                if used[pid]:
                    continue
                inc = self._incremental_cost(assignment, pos, pid)
                if inc < best_inc:
                    best_inc = inc
                    best_pid = pid

            if best_pid is None:
                # should not happen, but just in case
                for pid in range(N):
                    if not used[pid]:
                        best_pid = pid
                        best_inc = 0.0
                        break

            assignment[pos] = best_pid
            used[best_pid] = True

        full_assign = [pid for pid in assignment]  # type: ignore
        cost = self._full_cost(full_assign)
        self.best_assignment = full_assign
        self.best_cost = cost

        if self.debug:
            print(f"[GridSolver] Greedy initial cost = {cost:.3f}")

    # ------------------------------------------------------

    def _search(
        self,
        pos: int,
        assignment: List[Optional[int]],
        used: List[bool],
        current_cost: float,
    ) -> None:
        """
        Depth-first search with branch-and-bound.
        """
        self.node_count += 1

        # prune if already worse than best solution
        if current_cost >= self.best_cost:
            return

        if pos == self.N:
            full_assign = [pid for pid in assignment]  # type: ignore
            total = self._full_cost(full_assign)
            if total < self.best_cost:
                self.best_cost = total
                self.best_assignment = full_assign
                if self.debug:
                    print(f"[GridSolver] Found better solution: cost={total:.3f}")
            return

        # Order candidates by incremental cost (small first)
        candidates: List[Tuple[float, int]] = []
        for pid in range(self.N):
            if used[pid]:
                continue
            inc = self._incremental_cost(assignment, pos, pid)
            candidates.append((inc, pid))

        candidates.sort(key=lambda x: x[0])

        for inc, pid in candidates:
            new_cost = current_cost + inc
            if new_cost >= self.best_cost:
                continue

            assignment[pos] = pid
            used[pid] = True
            self._search(pos + 1, assignment, used, new_cost)
            used[pid] = False
            assignment[pos] = None

    # ------------------------------------------------------

    def solve(self) -> GridSolution:
        """
        Run the solver and return the best solution found.
        """
        # 1) Greedy initial bound
        self._greedy_initial()

        assignment: List[Optional[int]] = [None] * self.N
        used = [False] * self.N
        self.node_count = 0

        if self.debug:
            print("[GridSolver] Starting exhaustive search with pruning ...")

        self._search(0, assignment, used, 0.0)

        if self.best_assignment is None:
            raise RuntimeError("No solution found.")

        if self.debug:
            print(f"[GridSolver] Search finished. Visited {self.node_count} nodes.")
            print(f"[GridSolver] Best total cost = {self.best_cost:.3f}")

        return GridSolution(
            rows=self.rows,
            cols=self.cols,
            assignment=self.best_assignment,
            total_cost=self.best_cost,
        )


# ---------------------------------------------------------------------
# Rendering the solution
# ---------------------------------------------------------------------


def save_solution_image(
    pieces: List[PuzzlePiece],
    solution: GridSolution,
    out_path: str,
    margin: int = 0,
) -> None:
    """
    Render a solved grid into a single image and save it.

    We use an average piece size as the cell size.
    """
    rows, cols = solution.rows, solution.cols
    N = len(pieces)
    assert rows * cols == N

    heights = [p.size[0] for p in pieces]
    widths = [p.size[1] for p in pieces]
    cell_h = int(round(float(np.mean(heights))))
    cell_w = int(round(float(np.mean(widths))))

    H = rows * cell_h + (rows + 1) * margin
    W = cols * cell_w + (cols + 1) * margin

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            pid = solution.assignment[idx]
            piece = pieces[pid]

            y0 = margin + r * (cell_h + margin)
            x0 = margin + c * (cell_w + margin)

            resized = cv2.resize(piece.image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = resized

    cv2.imwrite(out_path, canvas)
    print(f"[save_solution_image] Saved assembled puzzle to {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Global grid puzzle solver (no rotation)."
    )
    parser.add_argument("image", help="Path to puzzle canvas (png/jpg/rgb).")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width for raw .rgb images (if needed).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height for raw .rgb images (if needed).",
    )
    parser.add_argument(
        "--out",
        default="assembled.png",
        help="Output image for assembled puzzle.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information.",
    )

    args = parser.parse_args()

    print("Preprocessing input image...")
    pieces = preprocess_puzzle_image(
        args.image,
        width=args.width,
        height=args.height,
        debug=args.debug,
    )
    print(f"Extracted {len(pieces)} pieces")

    print("Building grid solver...")
    solver = GridSolver(
        pieces,
        max_len_diff_ratio=0.12,
        mean_color_gate=80.0,
        alpha_profile=0.2,
        debug=args.debug,
    )

    print("Solving with global search...")
    solution = solver.solve()
    print(
        f"Solved grid {solution.rows}x{solution.cols} "
        f"with total cost {solution.total_cost:.3f}"
    )

    print("Saving solution image...")
    save_solution_image(pieces, solution, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
