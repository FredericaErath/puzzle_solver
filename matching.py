"""
matching.py

Edge matching + puzzle solving logic for the Computational Image Puzzle Solver.

This module assumes that preprocessing has already produced a list of
`PuzzlePiece` objects (see `preprocess.py`), each with per-edge color
features (histogram + 1D color profile in clockwise order).

Main responsibilities:
- Define edge-to-edge distance (matching).
- Build candidate matches between all piece edges.
- Greedy assembly of pieces into a global layout (x, y, rot).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import heapq
import cv2
import argparse
from preprocess import PuzzlePiece, EdgeFeatures, preprocess_puzzle_image


# ----------------------------------------------------------
# Basic types
# ----------------------------------------------------------

# We use side names consistent with preprocess.py
SIDES = ["top", "right", "bottom", "left"]


@dataclass(frozen=True)
class EdgeRef:
    """
    Reference to a specific edge of a specific piece.
    """
    piece_id: int
    side: str  # "top", "right", "bottom", or "left"


@dataclass
class EdgeMatch:
    """
    One candidate match between two edges.

    Attributes
    ----------
    src : EdgeRef
        First edge.
    dst : EdgeRef
        Second edge.
    distance : float
        Matching cost: smaller is better.
    """
    src: EdgeRef
    dst: EdgeRef
    distance: float


@dataclass
class Placement:
    """
    Pose of a puzzle piece in the global solved layout.

    We represent each piece as an axis-aligned rectangle with a rotation
    that is a multiple of 90 degrees (0, 90, 180, 270).

    Attributes
    ----------
    x : float
        X coordinate of the piece's top-left corner in global space.
    y : float
        Y coordinate of the piece's top-left corner in global space.
    rot : int
        Rotation in degrees: one of {0, 90, 180, 270}.
    """
    x: float
    y: float
    rot: int  # {0, 90, 180, 270}


@dataclass
class Solution:
    """
    Final solver result.

    Attributes
    ----------
    placements : Dict[int, Placement]
        Mapping from piece_id to its final placement.
    matches : List[EdgeMatch]
        List of edge matches actually used in the final assembly.
    """
    placements: Dict[int, Placement]
    matches: List[EdgeMatch]


# ----------------------------------------------------------
# Utility functions for sides / rotations
# ----------------------------------------------------------

def opposite_side(side: str) -> str:
    """
    Logical opposite of a side name (in local coordinates, before rotation).
    """
    if side == "top":
        return "bottom"
    if side == "bottom":
        return "top"
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError(f"Unknown side: {side}")


def rotate_side(side: str, rot: int) -> str:
    """
    Given a local side name ("top"/"right"/"bottom"/"left") and a rotation
    (0, 90, 180, 270 degrees clockwise), return which side becomes "top"
    in the rotated coordinate system.

    This helper is useful when you want to know which edge in piece.edges
    corresponds to a given global direction, once the piece is rotated.

    Note: For now we only define the mapping; the solver will decide how
    to use it when computing placements.

    Examples (rot = 90, clockwise):
        original "top"   becomes "right"
        original "right" becomes "bottom"
        original "bottom" becomes "left"
        original "left"  becomes "top"
    """
    rot = rot % 360
    if rot not in (0, 90, 180, 270):
        raise ValueError(f"Rotation must be multiple of 90, got {rot}")

    idx = SIDES.index(side)
    # Each 90° clockwise rotation moves index +1 in SIDES order:
    # ["top", "right", "bottom", "left"]
    steps = rot // 90
    new_idx = (idx + steps) % 4
    return SIDES[new_idx]


def edge_length(piece: PuzzlePiece, side: str) -> int:
    """
    Return the geometric length of a given edge of a piece in the rectified
    (unrotated) patch.

    top/bottom use the piece width
    left/right use the piece height
    """
    h, w = piece.size  # size = (height, width)
    if side in ("top", "bottom"):
        return w
    if side in ("left", "right"):
        return h
    raise ValueError(f"Unknown side: {side}")


# ----------------------------------------------------------
# Edge distance (matching)
# ----------------------------------------------------------

def edge_distance(
    edge_a: EdgeFeatures,
    len_a: int,
    edge_b: EdgeFeatures,
    len_b: int,
    max_length_diff_ratio: float = 0.1,
    alpha: float = 0.3,
) -> float:
    """
    Compute a matching distance between two edges using:
      - color histogram distance (coarse)
      - 1D color profile distance (fine, with reversed profile for B)

    The lower the distance, the better the match.

    Parameters
    ----------
    edge_a, edge_b : EdgeFeatures
        Features for the two edges to compare.
    len_a, len_b : int
        Geometric length (in pixels) of the two edges.
        Used to enforce that very mismatched lengths are not matched.
    max_length_diff_ratio : float
        Maximum allowed relative length difference. If exceeded, returns +inf.
    alpha : float
        Weight for histogram distance in [0, 1]. The profile distance
        weight is (1 - alpha).

    Returns
    -------
    float
        Matching cost. math.inf if the edges should not be matched at all.
    """
    # 1) Length compatibility check
    max_len = max(len_a, len_b)
    if max_len <= 0:
        return math.inf

    if abs(len_a - len_b) / max_len > max_length_diff_ratio:
        # Too different, reject this pair
        return math.inf

    # 2) Histogram distance (L2)
    hist_a = edge_a.color_hist
    hist_b = edge_b.color_hist
    d_hist = float(np.linalg.norm(hist_a - hist_b))

    # 3) 1D color profile distance (MSE), with profile_b reversed
    prof_a = edge_a.color_profile
    prof_b = edge_b.color_profile[::-1]  # face-to-face comparison

    L = min(len(prof_a), len(prof_b))
    if L <= 0:
        d_prof = float("inf")
    else:
        diff = prof_a[:L] - prof_b[:L]
        d_prof = float(np.mean(diff ** 2))

    # 4) Combined distance
    return alpha * d_hist + (1.0 - alpha) * d_prof


# ----------------------------------------------------------
# Solver class: builds candidates + assembles pieces
# ----------------------------------------------------------

class PuzzleSolver:
    """
    Combined matching + solver for rectangular puzzle pieces.

    Usage:
        solver = PuzzleSolver(pieces)
        solution = solver.solve()

    It uses:
      - edge_distance() to build candidate edge matches
      - a greedy expansion strategy to place pieces in 2D space
    """

    def __init__(
        self,
        pieces: List[PuzzlePiece],
        max_length_diff_ratio: float = 0.1,
        alpha: float = 0.3,
        max_candidates_per_edge: int = 10,
        max_match_distance: float = 5.0,
    ) -> None:
        self.pieces = pieces
        self.max_length_diff_ratio = max_length_diff_ratio
        self.alpha = alpha
        self.max_candidates_per_edge = max_candidates_per_edge
        self.max_match_distance = max_match_distance

        # Edge candidates:
        #   candidates[(piece_id, side)] = List[EdgeMatch] (sorted by distance)
        self.candidates: Dict[Tuple[int, str], List[EdgeMatch]] = {}

        # Final placements
        self.placements: Dict[int, Placement] = {}

        # Edge matches actually used in assembly
        self.used_matches: List[EdgeMatch] = []

    # ------------------------------------------------------
    # Candidate building
    # ------------------------------------------------------

    def build_edge_candidates(self) -> None:
        """
        For each edge of each piece, compute top-K best matching edges from
        all *other* pieces, based on edge_distance().

        This optimized version:
            - Flattens all edges into a 1D list.
            - Only computes distance once for each unordered pair (a, b)
            with a < b.
            - Adds the result to both edges' candidate lists.

        Result is stored in self.candidates as:
            self.candidates[(piece_id, side)] = List[EdgeMatch]
        """
        # 1) Flatten all edges into a list of EdgeRef
        edges: List[EdgeRef] = []
        for i, _piece in enumerate(self.pieces):
            for side in SIDES:
                edges.append(EdgeRef(piece_id=i, side=side))

        num_edges = len(edges)

        # 2) Precompute edge lengths
        lengths: Dict[EdgeRef, int] = {}
        for ref in edges:
            piece = self.pieces[ref.piece_id]
            lengths[ref] = edge_length(piece, ref.side)

        # 3) Initialize candidate dict: EdgeRef -> List[EdgeMatch]
        candidates_by_ref: Dict[EdgeRef, List[EdgeMatch]] = {
            ref: [] for ref in edges
        }

        # 4) Only compute distances for (a < b) pairs
        for a in range(num_edges):
            ref_a = edges[a]
            piece_a = self.pieces[ref_a.piece_id]
            feat_a = piece_a.edges[ref_a.side]
            len_a = lengths[ref_a]

            for b in range(a + 1, num_edges):
                ref_b = edges[b]

                # Never match edges from the same piece
                if ref_b.piece_id == ref_a.piece_id:
                    continue

                piece_b = self.pieces[ref_b.piece_id]
                feat_b = piece_b.edges[ref_b.side]
                len_b = lengths[ref_b]

                d = edge_distance(
                    feat_a,
                    len_a,
                    feat_b,
                    len_b,
                    max_length_diff_ratio=self.max_length_diff_ratio,
                    alpha=self.alpha,
                )
                if not math.isfinite(d):
                    continue

                # One computation, two directed matches:
                #   ref_a -> ref_b
                #   ref_b -> ref_a
                m_ab = EdgeMatch(src=ref_a, dst=ref_b, distance=d)
                m_ba = EdgeMatch(src=ref_b, dst=ref_a, distance=d)

                candidates_by_ref[ref_a].append(m_ab)
                candidates_by_ref[ref_b].append(m_ba)

        # 5) For each edge, keep only top-K candidates and convert key
        #    from EdgeRef to (piece_id, side) to match external API
        final: Dict[Tuple[int, str], List[EdgeMatch]] = {}

        for ref, matches in candidates_by_ref.items():
            # sort by distance ascending
            matches.sort(key=lambda m: m.distance)
            if self.max_candidates_per_edge > 0:
                matches = matches[: self.max_candidates_per_edge]

            key = (ref.piece_id, ref.side)
            final[key] = matches

        self.candidates = final


    # ------------------------------------------------------
    # Geometry helpers for placement
    # ------------------------------------------------------

    def piece_rotated_size(self, piece_id: int, rot: int) -> Tuple[int, int]:
        """
        Return (height, width) of a piece after applying rotation.
        If rot is 90 or 270, height and width are swapped.
        """
        h, w = self.pieces[piece_id].size
        rot = rot % 180
        if rot == 0:
            return h, w
        else:
            return w, h

    def compute_neighbor_pose(
        self,
        base_id: int,
        base_pose: Placement,
        base_side: str,
        neighbor_id: int,
        neighbor_side: str,
    ) -> Optional[Placement]:
        """
        Given that base_id is already placed with base_pose, and we decide to
        match (base_id, base_side) with (neighbor_id, neighbor_side), compute
        a possible pose (x, y, rot) for the neighbor.

        Strategy:
        - For rot_n in {0, 90, 180, 270}:
            * compute global direction of base_side under base_pose.rot
            * compute global direction of neighbor_side under rot_n
            * if they are opposite, compute neighbor (x, y) so that:
                - the two edges touch
                - their centers align along the shared edge

        Returns
        -------
        Placement or None
            None if no consistent pose can be found.
        """
        # Rotated size of the base piece
        hb, wb = self.piece_rotated_size(base_id, base_pose.rot)

        # Global direction of the base edge
        base_global_side = rotate_side(base_side, base_pose.rot)  # reuse helper
        needed_neighbor_global = opposite_side(base_global_side)

        for rot_n in (0, 90, 180, 270):
            # After rotation rot_n, neighbor_side faces which global direction?
            neighbor_global_side = rotate_side(neighbor_side, rot_n)
            if neighbor_global_side != needed_neighbor_global:
                continue

            # This rotation makes the edges face each other, compute translation
            hn, wn = self.piece_rotated_size(neighbor_id, rot_n)

            # Base top-left
            bx = base_pose.x
            by = base_pose.y

            # Align based on global side
            if base_global_side == "right":
                # base right edge at x = bx + wb
                # neighbor left edge at x = nx
                nx = bx + wb
                # align vertical centers
                cy_base = by + hb / 2.0
                ny = cy_base - hn / 2.0

            elif base_global_side == "left":
                # base left edge at x = bx
                # neighbor right edge at x = nx + wn
                nx = bx - wn
                cy_base = by + hb / 2.0
                ny = cy_base - hn / 2.0

            elif base_global_side == "bottom":
                # base bottom edge at y = by + hb
                # neighbor top edge at y = ny
                ny = by + hb
                cx_base = bx + wb / 2.0
                nx = cx_base - wn / 2.0

            elif base_global_side == "top":
                # base top edge at y = by
                # neighbor bottom edge at y = ny + hn
                ny = by - hn
                cx_base = bx + wb / 2.0
                nx = cx_base - wn / 2.0

            else:
                continue

            return Placement(x=float(nx), y=float(ny), rot=rot_n)

        # No rotation made the sides face correctly
        return None


    def check_overlap(self, new_piece_id: int, new_pose: Placement) -> bool:
        """
        Check whether the new piece (with given pose) significantly overlaps
        any already placed piece (axis-aligned rectangle intersection test).

        Returns
        -------
        bool
            True if there is a conflict (overlap), False otherwise.
        """
        h_new, w_new = self.piece_rotated_size(new_piece_id, new_pose.rot)
        x1 = new_pose.x
        y1 = new_pose.y
        x2 = x1 + w_new
        y2 = y1 + h_new

        for pid, pose in self.placements.items():
            h, w = self.piece_rotated_size(pid, pose.rot)
            bx1 = pose.x
            by1 = pose.y
            bx2 = bx1 + w
            by2 = by1 + h

            # Strict inequalities: touching at border is OK (area=0)
            if x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1:
                overlap_w = min(x2, bx2) - max(x1, bx1)
                overlap_h = min(y2, by2) - max(y1, by1)
                if overlap_w > 1e-3 and overlap_h > 1e-3:
                    return True

        return False
    

    def edge_best_distance(self, piece_id: int, side: str) -> float:
        """
        Return the distance of the best candidate match for a given edge.
        If there are no candidates, return +inf.
        """
        key = (piece_id, side)
        cands = self.candidates.get(key, [])
        if not cands:
            return float("inf")
        return cands[0].distance

    # ------------------------------------------------------
    # Greedy solving
    # ------------------------------------------------------

    def choose_seed_match(self) -> Optional[EdgeMatch]:
        """
        Choose a global best (lowest-distance) edge match from all candidates
        as the initial seed pair.

        Returns
        -------
        EdgeMatch or None
            None if no candidates exist.
        """
        best: Optional[EdgeMatch] = None
        for matches in self.candidates.values():
            for m in matches:
                if best is None or m.distance < best.distance:
                    best = m
        return best

    def solve(self) -> Solution:
        """
        Main entry point: run matching + greedy assembly.

        Strategy (greedy, no backtracking):
        1. Build edge candidates (if not already built).
        2. Pick the best global edge match as seed.
        3. Place the two seed pieces.
        4. Maintain a frontier of unmatched edges on placed pieces,
           prioritized by their best candidate distance.
        5. Iteratively:
            - pop one frontier edge with smallest best distance
            - scan its candidate list, try to place a new neighbor piece
              that doesn't overlap existing ones and whose match distance
              is <= max_match_distance
            - accept the first valid match, update placements & frontier
        6. Stop when:
            - all pieces are placed, or
            - no more frontier moves possible.

        Returns
        -------
        Solution
            placements and used edge matches.
        """
        if not self.candidates:
            self.build_edge_candidates()

        seed = self.choose_seed_match()
        if seed is None:
            # No candidates at all; return empty solution
            return Solution(placements={}, matches=[])

        # Initialize placements
        self.placements = {}
        self.used_matches = []

        # matched_edges[(piece_id, side)] tells whether this edge is already used
        matched_edges: Dict[Tuple[int, str], bool] = {}
        for i, _p in enumerate(self.pieces):
            for side in SIDES:
                matched_edges[(i, side)] = False

        # --------------------------------------------------
        # 1) Place the seed pair
        # --------------------------------------------------
        def try_place_seed(seed_match: EdgeMatch) -> bool:
            # Try placing src as (0, 0, rot=0), solve dst pose
            src_id = seed_match.src.piece_id
            dst_id = seed_match.dst.piece_id
            src_side = seed_match.src.side
            dst_side = seed_match.dst.side

            base_pose = Placement(x=0.0, y=0.0, rot=0)
            self.placements[src_id] = base_pose

            pose_dst = self.compute_neighbor_pose(
                base_id=src_id,
                base_pose=base_pose,
                base_side=src_side,
                neighbor_id=dst_id,
                neighbor_side=dst_side,
            )
            if pose_dst is None or self.check_overlap(dst_id, pose_dst):
                # rollback and fail
                self.placements.clear()
                return False

            self.placements[dst_id] = pose_dst

            matched_edges[(src_id, src_side)] = True
            matched_edges[(dst_id, dst_side)] = True
            self.used_matches.append(seed_match)
            return True

        if not try_place_seed(seed):
            # swap roles and try again
            swapped = EdgeMatch(
                src=seed.dst,
                dst=seed.src,
                distance=seed.distance,
            )
            if not try_place_seed(swapped):
                return Solution(placements={}, matches=[])

        # --------------------------------------------------
        # 2) Initialize frontier: all unmatched edges of placed pieces
        # --------------------------------------------------
        
        frontier_heap: List[Tuple[float, int, str]] = [] # (best_dist, piece_id, side)
        for pid in self.placements.keys():
            for side in SIDES:
                if matched_edges[(pid, side)]:
                    continue
                best_d = self.edge_best_distance(pid, side)
                if math.isfinite(best_d):
                    heapq.heappush(frontier_heap, (best_d, pid, side))

        # --------------------------------------------------
        # 3) Greedy expansion
        # --------------------------------------------------
        while frontier_heap and len(self.placements) < len(self.pieces):
            best_d, i, side_i = heapq.heappop(frontier_heap)

            if matched_edges[(i, side_i)]:
                continue
            if i not in self.placements:
                continue

            base_pose = self.placements[i]
            base_key = (i, side_i)
            candidates_for_edge = self.candidates.get(base_key, [])

            placed_new_piece = False

            for m in candidates_for_edge:
                if m.distance > self.max_match_distance:
                    continue

                j = m.dst.piece_id
                side_j = m.dst.side

                if matched_edges[(i, side_i)]:
                    break

                if j == i:
                    continue

                if j not in self.placements:
                    neighbor_pose = self.compute_neighbor_pose(
                        base_id=i,
                        base_pose=base_pose,
                        base_side=side_i,
                        neighbor_id=j,
                        neighbor_side=side_j,
                    )
                    if neighbor_pose is None:
                        continue
                    if self.check_overlap(j, neighbor_pose):
                        continue

                    self.placements[j] = neighbor_pose
                    matched_edges[(i, side_i)] = True
                    matched_edges[(j, side_j)] = True
                    self.used_matches.append(m)
                    placed_new_piece = True

                    for side2 in SIDES:
                        if not matched_edges[(j, side2)]:
                            best2 = self.edge_best_distance(j, side2)
                            if math.isfinite(best2):
                                heapq.heappush(frontier_heap, (best2, j, side2))
                    break

                else:
                    continue

        return Solution(placements=self.placements, matches=self.used_matches)
    

def save_solution_image(
    pieces: List[PuzzlePiece],
    placements: Dict[int, Placement],
    out_path: str = "solution.png"
) -> None:
    """
    Render the solved puzzle layout into a single image and save it.

    Parameters
    ----------
    pieces : List[PuzzlePiece]
        Original rectified pieces from preprocess.py
    placements : Dict[int, Placement]
        Result of solver.solve(): piece_id -> Placement(x, y, rot)
    out_path : str
        Output file path, e.g. "solution.png"
    """

    if not placements:
        print("[save_solution_image] No placements provided, nothing to save.")
        return

    # Collect all coordinates to determine global canvas bounds
    xs = []
    ys = []

    for pid, pose in placements.items():
        h, w = pieces[pid].size
        if pose.rot % 180 == 90:
            w, h = h, w  # width/height swap

        xs.extend([pose.x, pose.x + w])
        ys.extend([pose.y, pose.y + h])

    # Determine canvas size (allowing negative coords)
    min_x = int(min(xs))
    min_y = int(min(ys))
    max_x = int(max(xs))
    max_y = int(max(ys))

    # Shift so that all coordinates are positive
    shift_x = -min_x
    shift_y = -min_y

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    # Create blank canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Paste each piece onto the canvas
    for pid, pose in placements.items():
        piece_img = pieces[pid].image.copy()

        # Apply rotation
        rot = pose.rot % 360
        if rot == 90:
            piece_img = cv2.rotate(piece_img, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            piece_img = cv2.rotate(piece_img, cv2.ROTATE_180)
        elif rot == 270:
            piece_img = cv2.rotate(piece_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = piece_img.shape[:2]

        # target location on canvas
        tx = int(pose.x + shift_x)
        ty = int(pose.y + shift_y)

        # Paste using mask
        mask = np.any(piece_img > 0, axis=2).astype(np.uint8)  # 1 where piece exists
        roi = canvas[ty:ty + h, tx:tx + w]

        # Only overwrite non-black pixels
        roi[mask == 1] = piece_img[mask == 1]

    # Save result
    cv2.imwrite(out_path, canvas)
    print(f"[save_solution_image] Saved assembled puzzle to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test puzzle solver end-to-end.")
    parser.add_argument("image", help="Path to puzzle canvas image (e.g., parrot_puzzle.png)")
    parser.add_argument("--out", default="assembled.png", help="Output assembled puzzle image")
    parser.add_argument("--debug", action="store_true", help="Print debug information")

    args = parser.parse_args()

    # 1. Preprocess canvas into pieces
    print("Preprocessing input image...")
    pieces = preprocess_puzzle_image(args.image, debug=args.debug)
    print(f"Extracted {len(pieces)} pieces")

    # 2. Initialize solver
    print("Building solver...")
    solver = PuzzleSolver(pieces, 
                            max_length_diff_ratio=0.05, 
                            alpha = 0.2,
                            max_candidates_per_edge = 10,
                            max_match_distance = 800,)

    # 3. Solve puzzle
    print("Solving...")
    solution = solver.solve()
    print(f"Solver used {len(solution.matches)} matches")
    print(f"Placed {len(solution.placements)} / {len(pieces)} pieces")

    # 4. Render and save final assembled result
    print("Saving solution image...")
    save_solution_image(pieces, solution.placements, out_path=args.out)

    print(f"Done. Final puzzle saved to {args.out}")


if __name__ == "__main__":
    main()