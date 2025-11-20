"""
matching.py

Puzzle piece matching and global arrangement for the
Computational Image Puzzle Solver.

This version is designed to work with the user's existing
`preprocess.py` (in BGR space, with RGB planar support for .rgb files).

Key ideas
---------
1. Use the per–edge features produced in preprocess:
   - mean_color      : (B,G,R)
   - color_hist      : concatenated B/G/R histograms
   - color_profile   : 1-D sequence of BGR values along the edge

2. Define a robust edge distance that combines
   - histogram L2 distance
   - profile mean squared error (MSE) along the edge
   plus some sanity checks (length and mean-color gates).

3. Assume the puzzle is a regular grid with equal-sized pieces
   (this is true for the provided macaw and Starry Night puzzles).
   We then solve a global assignment problem with backtracking:
   assign each piece to one cell in an R x C grid to minimize
   the sum of edge distances between neighbouring cells.

   Because we optimise *globally*, every piece is used exactly once
   and we avoid the "greedy local, globally wrong" behaviour.

4. At the moment we assume **no rotations** for the puzzle pieces.
   Supporting rotations would blow up the search space (4^N),
   so it would require additional heuristics and is omitted here.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from preprocess import PuzzlePiece, preprocess_puzzle_image


# --------------------------------------------------------------------
# Edge distance
# --------------------------------------------------------------------


def edge_length_from_piece(piece: PuzzlePiece, side: str) -> int:
    """
    Convenience helper to get the length (number of samples) of
    the specified edge's colour profile.
    """
    return piece.edges[side].color_profile.shape[0]


def edge_distance(
    edge_a,
    edge_b,
    max_len_diff_ratio: float = 0.08,
    mean_color_gate: Optional[float] = 80.0,
    alpha_profile: float = 0.2,
) -> float:
    """
    Compute a distance between two edges.

    We combine:

      1) Colour histogram L2 distance
      2) Per-position profile MSE (after reversing edge_b so
         that the physical boundary is aligned)

    plus some simple gates so obviously mismatched edges get
    a very large cost (treated as impossible).

    Parameters
    ----------
    edge_a, edge_b :
        EdgeFeatures objects from preprocess.PuzzlePiece.edges.
    max_len_diff_ratio : float
        Maximum relative difference in edge length that we allow.
        If |lenA - lenB| / max(lenA, lenB) > ratio, cost = +inf.
    mean_color_gate : float or None
        If not None, we compute Euclidean distance between mean colours
        of the two edges; if this is larger than the gate, cost = +inf.
    alpha_profile : float
        Weight for the profile MSE term relative to histogram distance.

    Returns
    -------
    float
        Non-negative cost (smaller is better).
    """
    prof_a = edge_a.color_profile.astype(np.float32)
    prof_b = edge_b.color_profile.astype(np.float32)

    len_a = prof_a.shape[0]
    len_b = prof_b.shape[0]
    max_len = max(len_a, len_b)
    if max_len == 0:
        # Shouldn't happen, but be safe.
        return float("inf")

    # --- Length gate -------------------------------------------------
    rel_diff = abs(len_a - len_b) / max_len
    if rel_diff > max_len_diff_ratio:
        return float("inf")

    # --- Mean-colour gate --------------------------------------------
    if mean_color_gate is not None:
        mc_a = np.array(edge_a.mean_color, dtype=np.float32)
        mc_b = np.array(edge_b.mean_color, dtype=np.float32)
        mean_diff = float(np.linalg.norm(mc_a - mc_b))
        if mean_diff > mean_color_gate:
            return float("inf")

    # -----------------------------------------------------------------
    # Histograms: already normalised in preprocess.
    hist_a = edge_a.color_hist.astype(np.float32)
    hist_b = edge_b.color_hist.astype(np.float32)
    hist_dist = float(np.linalg.norm(hist_a - hist_b))

    # -----------------------------------------------------------------
    # Profile distance:
    # prof_a has shape (La, 3), prof_b has (Lb, 3).
    # Reverse B so that the boundary direction matches.
    prof_b_rev = prof_b[::-1, :]

    L = min(len_a, len_b)
    if L <= 0:
        profile_mse = 0.0
    else:
        diff = prof_a[:L] - prof_b_rev[:L]
        mse = float(np.mean(diff * diff))
        profile_mse = mse

    cost = hist_dist + alpha_profile * profile_mse
    return cost


# --------------------------------------------------------------------
# Grid solver (global backtracking)
# --------------------------------------------------------------------


@dataclass
class GridSolution:
    """
    Represents a solved R x C grid.

    Attributes
    ----------
    rows, cols : int
        Grid shape.
    assignment : List[int]
        assignment[r * cols + c] = piece_id.
    total_cost : float
        Sum of neighbour edge costs.
    """
    rows: int
    cols: int
    assignment: List[int]
    total_cost: float


class GridSolver:
    """
    Global optimiser for a rectangular puzzle grid.

    We assign each piece to one grid cell (no rotations) so that the
    sum of edge distances between neighbouring cells is minimised.

    This is done by depth-first search with branch-and-bound pruning.
    For N=16 or similar this is fast enough, especially because we
    also start from a cheap greedy solution to get a good upper bound.
    """

    def __init__(
        self,
        pieces: List[PuzzlePiece],
        max_len_diff_ratio: float = 0.08,
        mean_color_gate: float = 80.0,
        alpha_profile: float = 0.2,
        debug: bool = False,
    ):
        self.pieces = pieces
        self.N = len(pieces)
        self.debug = debug

        # Infer grid dimensions (rows, cols) from N.
        self.rows, self.cols = self._infer_grid_dims(self.N)
        if self.rows * self.cols != self.N:
            raise ValueError(
                f"Cannot form a grid from N={self.N} pieces "
                f"(rows={self.rows}, cols={self.cols})."
            )

        if self.debug:
            print(f"[GridSolver] Using grid {self.rows} x {self.cols} for {self.N} pieces.")

        # Pre-compute adjacency costs for all ordered pairs.
        self.max_len_diff_ratio = max_len_diff_ratio
        self.mean_color_gate = mean_color_gate
        self.alpha_profile = alpha_profile

        self.horiz_cost = np.full((self.N, self.N), np.inf, dtype=np.float32)
        self.vert_cost = np.full((self.N, self.N), np.inf, dtype=np.float32)
        self._build_cost_tables()

        # State for search
        self.best_cost = float("inf")
        self.best_assignment: Optional[List[int]] = None
        self.node_count = 0

    # -----------------------------------------------------------------

    def _infer_grid_dims(self, N: int) -> Tuple[int, int]:
        """
        Pick a (rows, cols) factorisation of N that is as square as possible.
        """
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

    # -----------------------------------------------------------------

    def _build_cost_tables(self) -> None:
        """
        Build matrices:

            horiz_cost[i, j] : cost for placing piece j to the RIGHT of piece i
            vert_cost[i, j]  : cost for placing piece j BELOW piece i
        """
        N = self.N
        if self.debug:
            print("[GridSolver] Precomputing pairwise edge costs ...")

        for i in range(N):
            pi = self.pieces[i]
            edge_i_right = pi.edges["right"]
            edge_i_bottom = pi.edges["bottom"]

            for j in range(N):
                if i == j:
                    continue
                pj = self.pieces[j]
                edge_j_left = pj.edges["left"]
                edge_j_top = pj.edges["top"]

                # Cost for j to the RIGHT of i
                cost_h = edge_distance(
                    edge_i_right,
                    edge_j_left,
                    max_len_diff_ratio=self.max_len_diff_ratio,
                    mean_color_gate=self.mean_color_gate,
                    alpha_profile=self.alpha_profile,
                )
                self.horiz_cost[i, j] = cost_h

                # Cost for j BELOW i
                cost_v = edge_distance(
                    edge_i_bottom,
                    edge_j_top,
                    max_len_diff_ratio=self.max_len_diff_ratio,
                    mean_color_gate=self.mean_color_gate,
                    alpha_profile=self.alpha_profile,
                )
                self.vert_cost[i, j] = cost_v

        if self.debug:
            finite_h = np.isfinite(self.horiz_cost).sum()
            finite_v = np.isfinite(self.vert_cost).sum()
            print(f"[GridSolver] horiz_cost finite entries: {finite_h}/{N*N}")
            print(f"[GridSolver] vert_cost  finite entries: {finite_v}/{N*N}")

    # -----------------------------------------------------------------

    def _incremental_cost(
        self,
        assignment: List[int],
        pos: int,
        piece_id: int,
    ) -> float:
        """
        Additional cost of placing `piece_id` into cell `pos`,
        given partial assignment (left/up neighbours possibly filled).
        """
        r = pos // self.cols
        c = pos % self.cols
        inc = 0.0

        # Left neighbour
        if c > 0:
            left_pid = assignment[pos - 1]
            if left_pid is not None:
                cost = float(self.horiz_cost[left_pid, piece_id])
                if not math.isfinite(cost):
                    return float("inf")
                inc += cost

        # Upper neighbour
        if r > 0:
            up_pid = assignment[pos - self.cols]
            if up_pid is not None:
                cost = float(self.vert_cost[up_pid, piece_id])
                if not math.isfinite(cost):
                    return float("inf")
                inc += cost

        return inc

    # -----------------------------------------------------------------

    def _full_cost(self, assignment: List[int]) -> float:
        """
        Compute total edge cost for a full assignment.
        """
        total = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                pid = assignment[idx]

                # Right neighbour
                if c + 1 < self.cols:
                    right_pid = assignment[idx + 1]
                    total += float(self.horiz_cost[pid, right_pid])

                # Down neighbour
                if r + 1 < self.rows:
                    down_pid = assignment[idx + self.cols]
                    total += float(self.vert_cost[pid, down_pid])
        return total

    # -----------------------------------------------------------------

    def _greedy_initial(self) -> None:
        """
        Quick greedy solution to get an initial upper bound.
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
                # Fallback: pick any remaining piece
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

    # -----------------------------------------------------------------

    def _search(
        self,
        pos: int,
        assignment: List[Optional[int]],
        used: List[bool],
        current_cost: float,
    ) -> None:
        """
        Depth-first search with pruning.
        """
        self.node_count += 1

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

        # Candidate ordering: try pieces with smaller incremental cost first
        candidate_costs: List[Tuple[float, int]] = []
        for pid in range(self.N):
            if used[pid]:
                continue
            inc = self._incremental_cost(assignment, pos, pid)
            if not math.isfinite(inc):
                continue
            candidate_costs.append((inc, pid))

        if not candidate_costs:
            for pid in range(self.N):
                if not used[pid]:
                    candidate_costs.append((0.0, pid))

        candidate_costs.sort(key=lambda x: x[0])

        for inc, pid in candidate_costs:
            new_cost = current_cost + inc
            if new_cost >= self.best_cost:
                continue

            assignment[pos] = pid
            used[pid] = True
            self._search(pos + 1, assignment, used, new_cost)
            used[pid] = False
            assignment[pos] = None

    # -----------------------------------------------------------------

    def solve(self) -> GridSolution:
        """
        Run solver and return the best grid solution found.
        """
        # 1) Greedy initial solution to set upper bound
        self._greedy_initial()

        # 2) Backtracking search
        assignment: List[Optional[int]] = [None] * self.N
        used = [False] * self.N
        self.node_count = 0

        if self.debug:
            print("[GridSolver] Starting exhaustive search with pruning ...")

        self._search(pos=0, assignment=assignment, used=used, current_cost=0.0)

        if self.best_assignment is None:
            raise RuntimeError("GridSolver did not find any solution.")

        if self.debug:
            print(f"[GridSolver] Search finished. Visited {self.node_count} nodes.")
            print(f"[GridSolver] Best total cost = {self.best_cost:.3f}")

        return GridSolution(
            rows=self.rows,
            cols=self.cols,
            assignment=self.best_assignment,
            total_cost=self.best_cost,
        )


# --------------------------------------------------------------------
# Rendering / visualisation
# --------------------------------------------------------------------


def save_solution_image(
    pieces: List[PuzzlePiece],
    solution: GridSolution,
    out_path: str,
    margin: int = 0,
) -> None:
    """
    Render the solved grid into a single image and save it.

    We assume no rotations: each piece is placed axis-aligned.  We use
    an average piece size to define the cell size in the assembled image.
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
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = resized

    cv2.imwrite(out_path, canvas)
    print(f"[save_solution_image] Saved assembled puzzle to {out_path}")


# --------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Global grid-based puzzle solver (no rotations)."
    )
    parser.add_argument("image", help="Path to puzzle canvas (png/jpg/rgb).")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width for raw .rgb input (required for .rgb).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height for raw .rgb input (required for .rgb).",
    )
    parser.add_argument(
        "--out",
        default="assembled.png",
        help="Output image file for the assembled puzzle.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose debugging information.",
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
        max_len_diff_ratio=0.08,
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
