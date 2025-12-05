#!/usr/bin/env python3
"""
puzzle_solver_hybrid.py - Hybrid Jigsaw Puzzle Solver

Combines:
1. Size-based grid grouping (ensures square output)
2. Edge-matching for ordering (ensures correct arrangement)
3. Template matching (when original available)

Usage:
python puzzle_solver_hybrid.py puzzle.png --out solved.png --debug
python puzzle_solver_hybrid.py puzzle.png --original orig.png --out solved.png
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from itertools import combinations
from collections import defaultdict
import cv2
import numpy as np

from preprocess import preprocess_puzzle_image


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


# ============= Template Matching =============

def template_match_solve(pieces, original, debug=False):
    total_area = sum(p.size[0] * p.size[1] for p in pieces)
    target = int(np.sqrt(total_area))

    if debug:
        print(f"[Template] Target size: {target}x{target}")

    orig_small = cv2.resize(original, (target, target))

    def find_best_match(piece):
        best = None
        img = piece.image.copy()
        mask = piece.mask.copy()

        for rot in range(4):
            if rot > 0:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

            mask_bin = (mask > 128).astype(np.uint8) * 255
            result = cv2.matchTemplate(orig_small, img, cv2.TM_CCORR_NORMED, mask=mask_bin)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if best is None or max_val > best[0]:
                h, w = img.shape[:2]
                best = (max_val, max_loc[0], max_loc[1], rot, h, w)

        return best

    all_matches = []
    for p in pieces:
        score, x, y, rot, h, w = find_best_match(p)
        all_matches.append({
            'id': p.id, 'rot': rot, 'x': x, 'y': y,
            'w': w, 'h': h, 'score': score
        })
        if debug:
            print(f"[Template] P{p.id}: ({x},{y}) rot={rot}, score={score:.4f}")

    # Sort by score and resolve overlaps
    all_matches = sorted(all_matches, key=lambda m: -m['score'])

    def overlaps(m1, m2, tol=5):
        return not (m1['x'] + m1['w'] <= m2['x'] + tol or
                    m2['x'] + m2['w'] <= m1['x'] + tol or
                    m1['y'] + m1['h'] <= m2['y'] + tol or
                    m2['y'] + m2['h'] <= m1['y'] + tol)

    final = []
    for m in all_matches:
        conflict = False
        for placed in final:
            if overlaps(m, placed):
                conflict = True
                break
        if not conflict:
            final.append(m)

    return final if len(final) == len(pieces) else None


# ============= Hybrid Grid + Edge Solver =============

def cluster_heights(heights, target_rows):
    """Cluster heights into target_rows groups using k-means"""
    vals = sorted(heights)
    k = target_rows
    c = [vals[i * len(vals) // k] for i in range(k)]

    for _ in range(20):
        g = [[] for _ in range(k)]
        for v in vals:
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

        # Cluster heights into nr groups
        row_heights = cluster_heights(all_heights, nr)
        col_widths = cluster_heights(all_widths, nc)

        sum_h = sum(row_heights)
        sum_w = sum(col_widths)

        aspect_diff = abs(sum_h - sum_w)
        target_diff = abs(sum_h - target) + abs(sum_w - target)

        score = aspect_diff * 2 + target_diff

        if debug:
            print(f"[Grid] {nr}x{nc}: h_sum={sum_h}, w_sum={sum_w}, score={score}")

        if score < best_score:
            best_score = score
            best_config = (nr, nc, row_heights)

    return best_config


def hybrid_solve(pieces, debug=False):
    """Solve using size-based grouping + edge-based ordering"""
    n = len(pieces)

    # Step 1: Infer grid configuration
    config = infer_grid_config(pieces, debug)
    if config is None:
        if debug:
            print("[Hybrid] Could not infer grid")
        return None

    nr, nc, row_heights = config
    if debug:
        print(f"[Hybrid] Using {nr}x{nc} grid, row_heights={row_heights}")

    # Helper to get piece by id
    piece_map = {p.id: p for p in pieces}

    # Step 2: Assign pieces to rows by height
    rows = {h: [] for h in row_heights}
    for p in pieces:
        best_row = min(row_heights, key=lambda rh: abs(rh - p.size[0]))
        rows[best_row].append(p.id)

    # Verify each row has nc pieces
    for rh, pids in rows.items():
        if len(pids) != nc:
            if debug:
                print(f"[Hybrid] Row {rh} has {len(pids)} pieces, expected {nc}")
            # This might still work, continue

    # Step 3: Order pieces within each row using edge matching
    def order_pieces(piece_ids):
        if len(piece_ids) <= 1:
            return piece_ids

        # Compute right-left scores
        scores = {}
        for p1_id in piece_ids:
            p1 = piece_map[p1_id]
            for p2_id in piece_ids:
                if p1_id == p2_id:
                    continue
                p2 = piece_map[p2_id]
                s = edge_score(p1.edge_lines[0]['right'], p2.edge_lines[0]['left'])
                scores[(p1_id, p2_id)] = s

        # Find leftmost piece (no good left neighbor)
        threshold = 0.4
        left_candidates = []
        for pid in piece_ids:
            has_left = any(scores.get((other, pid), 0) > threshold
                           for other in piece_ids if other != pid)
            if not has_left:
                left_candidates.append(pid)

        # Greedy ordering
        best_order = None
        best_total = -1

        for start in (left_candidates if left_candidates else piece_ids):
            order = [start]
            remaining = set(piece_ids) - {start}
            total_score = 0

            while remaining:
                current = order[-1]
                best_next = max(remaining, key=lambda x: scores.get((current, x), 0))
                total_score += scores.get((current, best_next), 0)
                order.append(best_next)
                remaining.remove(best_next)

            if total_score > best_total:
                best_total = total_score
                best_order = order

        return best_order

    ordered_rows = {}
    for rh in row_heights:
        ordered_rows[rh] = order_pieces(rows[rh])
        if debug:
            widths = [piece_map[pid].size[1] for pid in ordered_rows[rh]]
            print(f"[Hybrid] Row {rh}: {ordered_rows[rh]}, widths={widths}")

    # Step 4: Order rows using bottom-top edge matching
    def get_row_order(ordered_rows, row_heights):
        if len(row_heights) <= 1:
            return row_heights

        # Compute row-row scores
        scores = {}
        for rh1 in row_heights:
            for rh2 in row_heights:
                if rh1 == rh2:
                    continue
                total = 0
                count = 0
                for i, pid1 in enumerate(ordered_rows[rh1]):
                    if i < len(ordered_rows[rh2]):
                        pid2 = ordered_rows[rh2][i]
                        p1 = piece_map[pid1]
                        p2 = piece_map[pid2]
                        s = edge_score(p1.edge_lines[0]['bottom'], p2.edge_lines[0]['top'])
                        total += s
                        count += 1
                scores[(rh1, rh2)] = total / max(count, 1)

        # Find top row
        threshold = 0.4
        top_candidates = []
        for rh in row_heights:
            has_top = any(scores.get((other, rh), 0) > threshold
                          for other in row_heights if other != rh)
            if not has_top:
                top_candidates.append(rh)

        # Greedy ordering
        remaining = set(row_heights)
        start = top_candidates[0] if top_candidates else row_heights[0]
        order = [start]
        remaining.remove(start)

        while remaining:
            current = order[-1]
            best_next = max(remaining, key=lambda x: scores.get((current, x), 0))
            order.append(best_next)
            remaining.remove(best_next)

        return order

    row_order = get_row_order(ordered_rows, row_heights)
    if debug:
        print(f"[Hybrid] Row order: {row_order}")

    # Step 5: Build solution
    solution = []
    y = 0
    for rh in row_order:
        x = 0
        for pid in ordered_rows[rh]:
            p = piece_map[pid]
            solution.append({
                'id': pid,
                'rot': 0,
                'x': x,
                'y': y,
                'w': p.size[1],
                'h': p.size[0]
            })
            x += p.size[1]
        y += rh

    return solution


# ============= Edge-based Assembly (fallback) =============

def edge_based_assembly(pieces, debug=False):
    n = len(pieces)

    if debug:
        print("[Edge] Computing matches...")

    matches = []
    for p1 in range(n):
        for p2 in range(n):
            if p1 == p2:
                continue
            for r1 in range(4):
                for r2 in range(4):
                    rl = edge_score(pieces[p1].edge_lines[r1]['right'],
                                    pieces[p2].edge_lines[r2]['left'])
                    if rl > 0.3:
                        matches.append((p1, r1, 'right', p2, r2, 'left', rl))

                    bt = edge_score(pieces[p1].edge_lines[r1]['bottom'],
                                    pieces[p2].edge_lines[r2]['top'])
                    if bt > 0.3:
                        matches.append((p1, r1, 'bottom', p2, r2, 'top', bt))

    matches.sort(key=lambda x: -x[6])

    if debug:
        print(f"[Edge] {len(matches)} potential matches")
        if matches:
            print(
                f"[Edge] Best: P{matches[0][0]}r{matches[0][1]}.{matches[0][2]} <-> P{matches[0][3]}r{matches[0][4]}.{matches[0][5]}: {matches[0][6]:.3f}")

    if not matches:
        return None

    placed = {}
    m = matches[0]
    p1_size = get_size(pieces[m[0]], m[1])
    placed[m[0]] = (m[1], 0, 0)

    p2_size = get_size(pieces[m[3]], m[4])
    if m[2] == 'right':
        placed[m[3]] = (m[4], p1_size[1], 0)
    else:
        placed[m[3]] = (m[4], 0, p1_size[0])

    def calc_pos(placed_info, placed_piece, placed_rot, edge, new_piece, new_rot):
        pr, px, py = placed_info
        ph, pw = get_size(placed_piece, placed_rot)
        if edge == 'right':
            return (px + pw, py)
        elif edge == 'bottom':
            return (px, py + ph)
        elif edge == 'left':
            nh, nw = get_size(new_piece, new_rot)
            return (px - nw, py)
        else:
            nh, nw = get_size(new_piece, new_rot)
            return (px, py - nh)

    def calc_pos_inv(placed_info, placed_piece, placed_rot, edge, new_piece, new_rot):
        pr, px, py = placed_info
        ph, pw = get_size(placed_piece, placed_rot)
        nh, nw = get_size(new_piece, new_rot)
        if edge == 'left':
            return (px - nw, py)
        elif edge == 'top':
            return (px, py - nh)
        elif edge == 'right':
            return (px + pw, py)
        else:
            return (px, py + ph)

    def overlaps(pos, piece, rot, placed_dict, all_pieces, tol=5):
        x, y = pos
        h, w = get_size(piece, rot)
        for pid, (pr, px, py) in placed_dict.items():
            ph, pw = get_size(all_pieces[pid], pr)
            if not (x + w <= px + tol or px + pw <= x + tol or
                    y + h <= py + tol or py + ph <= y + tol):
                return True
        return False

    for _ in range(n - 2):
        best_match = None
        best_pos = None
        best_score = 0

        for m in matches:
            p1, r1, e1, p2, r2, e2, score = m

            if p1 in placed and p2 not in placed:
                if placed[p1][0] != r1:
                    continue
                pos = calc_pos(placed[p1], pieces[p1], r1, e1, pieces[p2], r2)
                if pos and not overlaps(pos, pieces[p2], r2, placed, pieces):
                    if score > best_score:
                        best_score = score
                        best_match = (p2, r2)
                        best_pos = pos

            elif p2 in placed and p1 not in placed:
                if placed[p2][0] != r2:
                    continue
                pos = calc_pos_inv(placed[p2], pieces[p2], r2, e2, pieces[p1], r1)
                if pos and not overlaps(pos, pieces[p1], r1, placed, pieces):
                    if score > best_score:
                        best_score = score
                        best_match = (p1, r1)
                        best_pos = pos

        if best_match:
            pid, rot = best_match
            placed[pid] = (rot, best_pos[0], best_pos[1])
        else:
            break

    if debug:
        print(f"[Edge] Placed {len(placed)}/{n} pieces")

    if len(placed) < n:
        return None

    min_x = min(p[1] for p in placed.values())
    min_y = min(p[2] for p in placed.values())

    result = []
    for pid, (rot, x, y) in placed.items():
        h, w = get_size(pieces[pid], rot)
        result.append({
            'id': pid, 'rot': rot,
            'x': x - min_x, 'y': y - min_y,
            'w': w, 'h': h
        })

    return result


# ============= Output =============

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


# ============= Main =============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", help="Puzzle image with scattered pieces")
    ap.add_argument("--original", help="Original complete image for reference")
    ap.add_argument("--out", default="solved.png")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--edge-only", action="store_true")
    args = ap.parse_args()

    print(f"[Main] Loading: {args.puzzle}")
    pieces, cfg = extract_pieces(args.puzzle, debug=args.debug)
    if not pieces:
        print("[Error] No pieces found")
        return 1

    print(f"[Main] {len(pieces)} pieces")

    sol = None

    # Strategy 1: Template matching (if original provided)
    if args.original and not args.edge_only:
        print(f"[Main] Using template matching with: {args.original}")
        original = cv2.imread(args.original)
        if original is not None:
            sol = template_match_solve(pieces, original, debug=args.debug)
            if sol and len(sol) == len(pieces):
                save_result(pieces, sol, args.out)
                print("[Main] Done (template matching)!")
                return 0

    # Strategy 2: Edge-only mode
    if args.edge_only:
        print("[Main] Using edge-based assembly only")
        sol = edge_based_assembly(pieces, debug=args.debug)
        if sol:
            save_result(pieces, sol, args.out)
            print("[Main] Done (edge-based)!")
            return 0
        print("[Error] Edge assembly failed")
        return 1

    # Strategy 3: Hybrid (size grouping + edge ordering)
    print("[Main] Using hybrid solver...")
    sol = hybrid_solve(pieces, debug=args.debug)

    if sol and len(sol) == len(pieces):
        save_result(pieces, sol, args.out)
        print("[Main] Done (hybrid)!")
        return 0

    # Strategy 4: Edge-based fallback
    print("[Main] Falling back to edge-based assembly...")
    sol = edge_based_assembly(pieces, debug=args.debug)
    if sol:
        save_result(pieces, sol, args.out)
        print("[Main] Done (edge-based fallback)!")
        return 0

    print("[Error] No solution found")
    return 1


if __name__ == "__main__":
    exit(main())