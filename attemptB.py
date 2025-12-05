#!/usr/bin/env python3
"""
puzzle_solver.py - Complete Jigsaw Puzzle Solver

Handles regular grids, semi-regular grids, and irregular puzzles with rotations.

Usage:
python puzzle_solver.py input.png --out solved.png --debug
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from itertools import combinations_with_replacement
from collections import Counter
import cv2
import os
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


def get_size(p, rot):
    h, w = p.size
    return (w, h) if rot % 2 else (h, w)


def kmeans1d(vals, k):
    if k >= len(vals):
        return sorted(vals)
    vals = sorted(vals)
    c = [vals[i * len(vals) // k] for i in range(k)]
    for _ in range(10):
        g = [[] for _ in range(k)]
        for v in vals:
            g[min(range(k), key=lambda i: abs(v - c[i]))].append(v)
        nc = [int(np.median(x)) if x else c[i] for i, x in enumerate(g)]
        if nc == c:
            break
        c = nc
    return sorted(c)


def can_fill(rows, cols, pieces, tol=25):
    need = [(r, c) for r in rows for c in cols]
    avail = [(p.size[0], p.size[1]) for p in pieces]
    for th, tw in need:
        best, bd = -1, 1e9
        for i, (ph, pw) in enumerate(avail):
            d = min(abs(ph - th) + abs(pw - tw), abs(pw - th) + abs(ph - tw))
            if d < bd:
                bd, best = d, i
        if best >= 0 and bd <= tol * 2:
            avail.pop(best)
        else:
            return False
    return len(avail) == 0


def infer_grid(pieces, debug=False):
    n = len(pieces)
    area = sum(p.size[0] * p.size[1] for p in pieces)
    target = int(np.sqrt(area))
    if debug:
        print(f"[Grid] {n} pieces, area={area}, target={target}")

    # Regular grid check
    sizes = set((min(p.size), max(p.size)) for p in pieces)
    if len(sizes) == 1:
        w, h = list(sizes)[0]
        if debug:
            print(f"[Grid] Regular: {w}x{h}")
        best = None
        for nr in range(1, n + 1):
            if n % nr:
                continue
            nc = n // nr
            for pw, ph in [(w, h), (h, w)]:
                ratio = max(nr * ph, nc * pw) / max(min(nr * ph, nc * pw), 1)
                if best is None or ratio < best[0]:
                    best = (ratio, [ph] * nr, [pw] * nc)
        if best:
            return best[1], best[2]

    # K-means clustering
    ah = [max(p.size) for p in pieces]
    aw = [min(p.size) for p in pieces]

    for tol in [30, 40, 50]:  # Try increasing tolerances
        for nr in range(2, min(n + 1, 8)):
            if n % nr:
                continue
            nc = n // nr
            if debug and tol == 30:
                print(f"[Grid] Trying {nr}x{nc}...")

            hc = kmeans1d(ah, nr)
            wc = kmeans1d(aw, nc)

            if can_fill(hc, wc, pieces, tol):
                if debug:
                    print(f"[Grid] Found (tol={tol}): {hc} x {wc}")
                return hc, wc
            if can_fill(wc, hc, pieces, tol):
                if debug:
                    print(f"[Grid] Found (swap, tol={tol}): {wc} x {hc}")
                return wc, hc

    # Last resort: use actual piece dimensions grouped by frequency
    if debug:
        print("[Grid] Fallback: trying actual dimensions...")

    all_h = sorted([max(p.size) for p in pieces])
    all_w = sorted([min(p.size) for p in pieces])

    # Try 5x4 and 4x5 specifically with actual dimensions
    for nr, nc in [(5, 4), (4, 5), (2, 10), (10, 2)]:
        if nr * nc != n:
            continue

        # Use k-means but with very wide tolerance
        hc = kmeans1d(all_h, nr)
        wc = kmeans1d(all_w, nc)

        if can_fill(hc, wc, pieces, 60):
            if debug:
                print(f"[Grid] Fallback found: {nr}x{nc}")
            return hc, wc

    return None, None


class Solver:
    def __init__(self, pieces, rows, cols, debug=False):
        self.pieces = pieces
        self.rows, self.cols = rows, cols
        self.nr, self.nc = len(rows), len(cols)
        self.debug = debug
        self.scores = self._pre()
        self.best = None
        self.best_s = -1e9
        self.it = 0

    def _pre(self):
        n = len(self.pieces)
        s = {}
        for p1 in range(n):
            s[p1] = {}
            for r1 in range(4):
                s[p1][r1] = {}
                for p2 in range(n):
                    if p1 == p2:
                        continue
                    s[p1][r1][p2] = {}
                    for r2 in range(4):
                        bt = edge_score(self.pieces[p1].edge_lines[r1]['bottom'],
                                        self.pieces[p2].edge_lines[r2]['top'])
                        rl = edge_score(self.pieces[p1].edge_lines[r1]['right'],
                                        self.pieces[p2].edge_lines[r2]['left'])
                        s[p1][r1][p2][r2] = (bt, rl)
        return s

    def _fits(self, pid, rot, r, c):
        ph, pw = get_size(self.pieces[pid], rot)
        return abs(ph - self.rows[r]) <= 50 and abs(pw - self.cols[c]) <= 50

    def solve(self):
        grid = [[None] * self.nc for _ in range(self.nr)]
        used = [False] * len(self.pieces)
        self._bt(grid, used, 0, 0)
        if self.debug:
            print(f"[Solver] it={self.it}, score={self.best_s:.2f}")
        return self._build() if self.best else None

    def _bt(self, grid, used, idx, sc):
        self.it += 1
        if self.it > 500000:
            return
        if idx >= self.nr * self.nc:
            if sc > self.best_s:
                self.best_s = sc
                self.best = [[grid[r][c] for c in range(self.nc)] for r in range(self.nr)]
            return

        r, c = idx // self.nc, idx % self.nc
        cands = []
        for pid in range(len(self.pieces)):
            if used[pid]:
                continue
            for rot in range(4):
                if not self._fits(pid, rot, r, c):
                    continue
                s = 0
                if r > 0 and grid[r - 1][c]:
                    np_, nr = grid[r - 1][c]
                    s += self.scores[np_][nr][pid][rot][0]
                if c > 0 and grid[r][c - 1]:
                    np_, nr = grid[r][c - 1]
                    s += self.scores[np_][nr][pid][rot][1]
                cands.append((pid, rot, s))

        cands.sort(key=lambda x: -x[2])
        for pid, rot, s in cands[:8]:
            grid[r][c] = (pid, rot)
            used[pid] = True
            self._bt(grid, used, idx + 1, sc + s)
            grid[r][c] = None
            used[pid] = False

    def _build(self):
        res = []
        y = 0
        for r in range(self.nr):
            x = 0
            for c in range(self.nc):
                pid, rot = self.best[r][c]
                res.append({'id': pid, 'rot': rot, 'row': r, 'col': c,
                            'x': x, 'y': y, 'h': self.rows[r], 'w': self.cols[c]})
                x += self.cols[c]
            y += self.rows[r]
        return res


def save_result(pieces, sol, path):
    my = max(p['y'] + p['h'] for p in sol)
    mx = max(p['x'] + p['w'] for p in sol)
    canvas = np.zeros((my, mx, 3), dtype=np.uint8)

    for p in sol:
        img, msk = pieces[p['id']].image.copy(), pieces[p['id']].mask.copy()
        for _ in range(p['rot']):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (p['w'], p['h']), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (p['w'], p['h']), interpolation=cv2.INTER_NEAREST)
        y, x, h, w = p['y'], p['x'], min(p['h'], my - p['y']), min(p['w'], mx - p['x'])
        if h > 0 and w > 0:
            roi = canvas[y:y + h, x:x + w]
            v = msk[:h, :w] > 128
            roi[v] = img[:h, :w][v]

    # Seam blend
    for p in sol:
        for yy in [p['y'], p['y'] + p['h']]:
            if 1 < yy < my - 1:
                canvas[yy - 1:yy + 2, :] = cv2.GaussianBlur(canvas[yy - 1:yy + 2, :], (1, 3), 0)
        for xx in [p['x'], p['x'] + p['w']]:
            if 1 < xx < mx - 1:
                canvas[:, xx - 1:xx + 2] = cv2.GaussianBlur(canvas[:, xx - 1:xx + 2], (3, 1), 0)

    cv2.imwrite(path, canvas)
    print(f"[Output] {path} ({mx}x{my})")
    with open(path.replace('.png', '.json'), 'w') as f:
        json.dump(sol, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="solved.png")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--edge-only", action="store_true", help="Use edge-based assembly only")
    args = ap.parse_args()

    print(f"[Main] Loading: {args.image}")
    pieces, cfg = extract_pieces(args.image, debug=args.debug)
    if not pieces:
        print("[Error] No pieces")
        return 1

    print(f"[Main] {len(pieces)} pieces")

    if args.edge_only:
        # Use edge-based assembly
        sol = edge_based_assembly(pieces, debug=True)
        if sol:
            save_edge_result(pieces, sol, args.out)
            print("[Main] Done (edge-based)!")
            return 0
        print("[Error] Edge assembly failed")
        return 1

    rows, cols = infer_grid(pieces, debug=True)
    if rows is None:
        print("[Warning] Grid inference failed, trying edge-based assembly...")
        sol = edge_based_assembly(pieces, debug=True)
        if sol:
            save_edge_result(pieces, sol, args.out)
            print("[Main] Done (edge-based fallback)!")
            return 0
        print("[Error] Both methods failed")
        return 1

    print(f"[Main] Grid: {len(rows)}x{len(cols)}")
    solver = Solver(pieces, rows, cols, debug=True)
    sol = solver.solve()

    if sol:
        save_result(pieces, sol, args.out)
        print("[Main] Done!")
        return 0

    # Grid solver failed, try edge-based
    print("[Warning] Grid solver failed, trying edge-based...")
    sol = edge_based_assembly(pieces, debug=True)
    if sol:
        save_edge_result(pieces, sol, args.out)
        print("[Main] Done (edge-based fallback)!")
        return 0

    print("[Error] No solution")
    return 1


def edge_based_assembly(pieces, debug=False):
    """Assemble puzzle using pure edge matching"""
    n = len(pieces)

    # Precompute all edge scores
    if debug:
        print("[Edge] Computing matches...")

    matches = []
    for p1 in range(n):
        for p2 in range(n):
            if p1 == p2:
                continue
            for r1 in range(4):
                for r2 in range(4):
                    # Right-Left match
                    rl = edge_score(pieces[p1].edge_lines[r1]['right'],
                                    pieces[p2].edge_lines[r2]['left'])
                    if rl > 0.3:
                        matches.append((p1, r1, 'right', p2, r2, 'left', rl))

                    # Bottom-Top match
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

    # Greedy assembly
    placed = {}  # pid -> (rotation, x, y)

    # Start with best match
    m = matches[0]
    p1_size = get_size(pieces[m[0]], m[1])
    placed[m[0]] = (m[1], 0, 0)

    p2_size = get_size(pieces[m[3]], m[4])
    if m[2] == 'right':
        placed[m[3]] = (m[4], p1_size[1], 0)
    else:
        placed[m[3]] = (m[4], 0, p1_size[0])

    # Add remaining pieces
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

    # Normalize positions
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


def calc_pos(placed_info, placed_piece, placed_rot, edge, new_piece, new_rot):
    """Calculate position for new piece adjacent to placed piece"""
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
    """Calculate position when matching from opposite direction"""
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


def overlaps(pos, piece, rot, placed, pieces, tol=5):
    """Check if position overlaps with any placed piece"""
    x, y = pos
    h, w = get_size(piece, rot)

    for pid, (pr, px, py) in placed.items():
        ph, pw = get_size(pieces[pid], pr)

        # Check overlap
        if not (x + w <= px + tol or px + pw <= x + tol or
                y + h <= py + tol or py + ph <= y + tol):
            return True
    return False


def save_edge_result(pieces, sol, path):
    """Save edge-based assembly result"""
    my = max(p['y'] + p['h'] for p in sol)
    mx = max(p['x'] + p['w'] for p in sol)
    canvas = np.zeros((my, mx, 3), dtype=np.uint8)

    for p in sol:
        img, msk = pieces[p['id']].image.copy(), pieces[p['id']].mask.copy()
        for _ in range(p['rot']):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

        # Use actual piece size
        h, w = img.shape[:2]
        y, x = p['y'], p['x']
        h = min(h, my - y)
        w = min(w, mx - x)

        if h > 0 and w > 0:
            roi = canvas[y:y + h, x:x + w]
            v = msk[:h, :w] > 128
            roi[v] = img[:h, :w][v]

    cv2.imwrite(path, canvas)
    print(f"[Output] {path} ({mx}x{my})")

    with open(path.replace('.png', '.json'), 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    exit(main())