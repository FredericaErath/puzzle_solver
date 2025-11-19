"""
preprocess.py

Preprocessing for the Computational Image Puzzle Solver.

Given a single large "canvas" image that contains N puzzle pieces
on a mostly black background (like the macaw parrot example),
this module:

1. Segments non-black pixels as foreground (puzzle pieces).
2. Finds connected components (contours) for each piece.
3. For each component, computes a minimum-area bounding rectangle,
   and warps it to an axis-aligned rectangular patch.
4. Extracts simple edge features for each patch (mean color + color histogram).

The output is a list of PuzzlePiece objects that you can later use
for edge matching and puzzle assembly.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import os
import argparse


# ----------------------------------------------------------
# Data classes
# ----------------------------------------------------------

@dataclass
class EdgeFeatures:
    """
    Features extracted from a single edge of a puzzle piece.

    Attributes
    ----------
    mean_color : (float, float, float)
        Average BGR color of the edge region.
    color_hist : np.ndarray
        Concatenated color histogram of B, G and R channels.
        (length = 3 * hist_bins)
    color_profile : np.ndarray
        1D color profile along the edge: shape (L, 3), where L is the
        number of pixels along that edge (width for top/bottom, height
        for left/right). Each entry is the mean BGR color at that
        position along the edge. 
        The profile is always stored in clockwise order around the piece:
          top   : TL -> TR  (left to right)
          right : TR -> BR  (top to bottom)
          bottom: BR -> BL  (right to left)
          left  : BL -> TL  (bottom to top)
    """
    mean_color: Tuple[float, float, float]
    color_hist: np.ndarray
    color_profile: np.ndarray


@dataclass
class PuzzlePiece:
    """
    Represents one puzzle piece extracted from the canvas.

    Attributes
    ----------
    id : int
        Integer ID of the piece (0, 1, 2, ...).
    image : np.ndarray
        Rectified image of this piece (BGR).
    mask : np.ndarray
        Binary mask of the same size as `image` (255 where the piece exists).
    canvas_corners : np.ndarray
        (4, 2) array of the four corner points in the original canvas.
    size : (int, int)
        (height, width) of the rectified piece image.
    edges : Dict[str, EdgeFeatures]
        Edge features for "top", "right", "bottom", and "left".
    """
    id: int
    image: np.ndarray
    mask: np.ndarray
    canvas_corners: np.ndarray
    size: Tuple[int, int]
    edges: Dict[str, EdgeFeatures]


# ----------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Robustly order four corner points to [top-left, top-right, bottom-right, bottom-left].

    This version avoids the tie problems of using only (x + y) and (y - x)
    by:
      1) sorting points by their angle around the centroid (CCW),
      2) choosing top-left as the point with minimal (x + y),
      3) ensuring overall CCW order.

    Parameters
    ----------
    pts : np.ndarray
        Input points of shape (4, 2) or (4, 1, 2).

    Returns
    -------
    np.ndarray
        Ordered points of shape (4, 2): [tl, tr, br, bl].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    # 1. Compute centroid
    center = pts.mean(axis=0)  # (cx, cy)

    # 2. Compute angle of each point relative to centroid
    #    atan2(y - cy, x - cx) ∈ (-pi, pi]
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

    # Sort points by angle (CCW order)
    idx = np.argsort(angles)
    pts_ccw = pts[idx]

    # 3. Choose a canonical starting point: top-left (smallest x + y)
    sums = pts_ccw.sum(axis=1)
    tl_idx = np.argmin(sums)

    # Rotate the array so that tl is first
    pts_ccw = np.roll(pts_ccw, -tl_idx, axis=0)
    # Now pts_ccw is something like [tl, ?, ?, ?] in CCW order

    # 4. Ensure orientation is [tl, tr, br, bl] in CCW.
    # Compute cross product of vectors (tl->second) x (tl->third)
    # If negative, points are actually in clockwise order, so we flip.
    v1 = pts_ccw[1] - pts_ccw[0]
    v2 = pts_ccw[2] - pts_ccw[0]
    cross = np.cross(v1, v2)

    if cross < 0:
        pts_ccw = np.array([pts_ccw[0], pts_ccw[3], pts_ccw[2], pts_ccw[1]], dtype=np.float32)

    return pts_ccw


def load_canvas_rgb(image_path: str,
                    width: Optional[int] = None,
                    height: Optional[int] = None) -> np.ndarray:
    """
    Load the puzzle canvas as a BGR image (for use with OpenCV).

    - For normal image formats (png/jpg/etc.), we simply call cv2.imread,
      which returns BGR.
    - For raw .rgb files (CSCI 576 style), we assume planar layout:
        [R plane][G plane][B plane],
      each plane having (width * height) bytes.
      We read the planes and convert them to a BGR image.

    Parameters
    ----------
    image_path : str
        Path to the image or raw .rgb file.
    width : int, optional
        Width of the raw .rgb image. Required when extension is ".rgb".
    height : int, optional
        Height of the raw .rgb image. Required when extension is ".rgb".

    Returns
    -------
    np.ndarray
        Image in BGR format (H, W, 3), ready to be used with OpenCV.
    """
    ext = os.path.splitext(image_path)[1].lower()

    # --- Case 1: raw .rgb (planar R, G, B) ---
    if ext == ".rgb":
        if width is None or height is None:
            raise ValueError(
                "Raw .rgb file requires explicit width and height "
                "(please pass width=..., height=...)."
            )

        # Read all bytes
        data = np.fromfile(image_path, dtype=np.uint8)
        expected = width * height * 3
        if data.size != expected:
            raise ValueError(
                f"Size mismatch for raw .rgb: expected {expected} bytes, "
                f"got {data.size}. Check width/height."
            )

        # Assume planar layout: [R plane][G plane][B plane]
        # Reshape to (3, H, W): 0=R, 1=G, 2=B
        planes = data.reshape((3, height, width))

        # Convert to BGR for consistency with OpenCV
        # B = planes[2], G = planes[1], R = planes[0]
        img_bgr = np.stack([planes[2], planes[1], planes[0]], axis=2)  # (H, W, 3)

        return img_bgr

    # --- Case 2: normal image formats (png/jpg/...) ---
    canvas_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if canvas_bgr is None:
        raise FileNotFoundError(f"Cannot read image from: {image_path}")

    # Already BGR, no conversion needed
    return canvas_bgr



def warp_piece(canvas: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a perspective transform to extract and rectify a puzzle piece.

    Parameters
    ----------
    canvas : np.ndarray
        Original canvas image (BGR).
    corners : np.ndarray
        (4, 2) array of corner coordinates in the canvas.

    Returns
    -------
    piece_img : np.ndarray
        Rectified color image of the piece.
    piece_mask : np.ndarray
        Rectified binary mask of the piece (same size as piece_img).
    """
    ordered = order_corners(corners)
    (tl, tr, br, bl) = ordered

    # Compute width and height of the target rectangle
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # Destination coordinates of the rectified rectangle
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Perspective transform
    M = cv2.getPerspectiveTransform(ordered, dst)
    piece_img = cv2.warpPerspective(canvas, M, (max_width, max_height))

    # Build and warp a mask using the same transform
    mask_canvas = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_canvas, corners.astype(np.int32), 255)
    piece_mask = cv2.warpPerspective(mask_canvas, M, (max_width, max_height))

    return piece_img, piece_mask


# ----------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------

def compute_edge_features(
        piece: np.ndarray,
        border: int = 3,
        hist_bins: int = 8
) -> Dict[str, EdgeFeatures]:
    """
    Compute simple features for each edge (top/right/bottom/left) of a piece.

    For each edge we take a thin strip of pixels (e.g. 3 pixels wide) and compute:
      - mean BGR color
      - normalized color histogram per channel
      - 1D color profile along the edge (mean BGR at each position)
        The profile is always stored in clockwise order around the piece:
          top   : TL -> TR  (left to right)
          right : TR -> BR  (top to bottom)
          bottom: BR -> BL  (right to left)
          left  : BL -> TL  (bottom to top)

    Parameters
    ----------
    piece : np.ndarray
        Rectified piece image (BGR).
    border : int, optional
        Width of the strip in pixels, by default 3.
    hist_bins : int, optional
        Number of bins for each color channel histogram, by default 8.

    Returns
    -------
    Dict[str, EdgeFeatures]
        Maps edge name to EdgeFeatures.
    """
    h, w, _ = piece.shape

    # Extract a strip around each edge
    edge_regions = {
        "top": piece[0:border, :, :],
        "bottom": piece[h - border:h, :, :],
        "left": piece[:, 0:border, :],
        "right": piece[:, w - border:w, :]
    }

    features: Dict[str, EdgeFeatures] = {}

    for edge_name, region in edge_regions.items():
        # Average BGR color
        mean_color = region.mean(axis=(0, 1))  # shape (3,)

        # Color histograms for B, G, R
        hist_list = []
        for ch in range(3):
            hist = cv2.calcHist(
                images=[region],
                channels=[ch],
                mask=None,
                histSize=[hist_bins],
                ranges=[0, 256]
            )
            hist = cv2.normalize(hist, None).flatten()
            hist_list.append(hist)

        color_hist = np.concatenate(hist_list)
        
        # 1D color profile in **clockwise** order:
        if edge_name == "top":
            # region: (border, w, 3) -> (w, 3), left to right
            color_profile = region.mean(axis=0)  # TL -> TR
        elif edge_name == "right":
            # region: (h, border, 3) -> (h, 3), top to bottom
            color_profile = region.mean(axis=1)  # TR -> BR
        elif edge_name == "bottom":
            # region: (border, w, 3) -> (w, 3), left to right is BL -> BR
            # we want BR -> BL, so reverse
            color_profile = region.mean(axis=0)[::-1]  # BR -> BL
        elif edge_name == "left":
            # region: (h, border, 3) -> (h, 3), top to bottom is TL -> BL
            # we want BL -> TL, so reverse
            color_profile = region.mean(axis=1)[::-1]  # BL -> TL
        else:
            # Fallback (should not happen)
            color_profile = region.mean(axis=0)


        features[edge_name] = EdgeFeatures(
            mean_color=(
                float(mean_color[0]),
                float(mean_color[1]),
                float(mean_color[2])
            ),
            color_hist=color_hist,
            color_profile=color_profile.astype(np.float32)
        )

    return features


# ----------------------------------------------------------
# Piece detection (fixed version for black background)
# ----------------------------------------------------------

def find_pieces(
        canvas: np.ndarray,
        min_area_ratio: float = 0.003,
        threshold_value: int = 10,
        debug: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detect puzzle pieces by foreground segmentation instead of edges.

    Assumes the background is (almost) black and puzzle pieces contain
    colorful pixels. Steps:

    1. Convert to grayscale.
    2. Threshold: pixels > threshold_value are considered foreground.
    3. Morphological open/close to clean noise and fill small gaps.
    4. Find external contours (each connected component = one piece).
    5. For each contour:
         - filter by area using min_area_ratio
         - compute a minimum-area bounding rectangle (always 4 corners)
         - warp the piece to a rectified patch.

    Parameters
    ----------
    canvas : np.ndarray
        Original canvas image (BGR).
    min_area_ratio : float, optional
        Minimum area as a fraction of total canvas area. This removes tiny blobs.
    threshold_value : int, optional
        Grayscale threshold to separate foreground from black background.
        10 works well for the parrot example.
    debug : bool, optional
        If True, prints diagnostic information.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of (piece_img, piece_mask, corners) for each detected piece.
    """
    h, w = canvas.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio

    # 1) Grayscale conversion
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # 2) Simple threshold: foreground = non-black pixels
    _, mask = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # 3) Morphological operations to remove small noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4) Find external contours on the binary mask
    contours, _ = cv2.findContours(
        mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    pieces_raw: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            # Ignore very small regions
            continue

        # 5a) Compute a minimum-area bounding rectangle for this contour.
        #     This is robust even if the contour has many vertices.
        rect = cv2.minAreaRect(cnt)  # (center, (w, h), angle)
        box = cv2.boxPoints(rect)  # 4 corner points
        corners = np.array(box, dtype=np.float32)

        # 5b) Warp the piece to a neat rectangular patch.
        piece_img, piece_mask = warp_piece(canvas, corners)
        pieces_raw.append((piece_img, piece_mask, corners))

    if debug:
        print(f"Detected {len(pieces_raw)} pieces.")

    return pieces_raw


# ----------------------------------------------------------
# High-level preprocessing API
# ----------------------------------------------------------

def preprocess_puzzle_image(
        image_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        debug: bool = False
) -> List[PuzzlePiece]:
    """
    Run the full preprocessing pipeline on a puzzle canvas.

    Parameters
    ----------
    image_path : str
        Path to the puzzle canvas image (e.g. 'parrot_puzzle.png').
    width: rgb file width
    height: rgb file height
    debug : bool, optional
        If True, prints diagnostic information.

    Returns
    -------
    """
    canvas = load_canvas_rgb(image_path, width=width, height=height)

    if debug:
        print(f"Loaded canvas from {image_path} with shape {canvas.shape}")

    raw_pieces = find_pieces(canvas, debug=debug)

    pieces: List[PuzzlePiece] = []
    for i, (piece_img, piece_mask, corners) in enumerate(raw_pieces):
        edges = compute_edge_features(piece_img)
        ph, pw = piece_img.shape[:2]

        piece = PuzzlePiece(
            id=i,
            image=piece_img,
            mask=piece_mask,
            canvas_corners=corners,
            size=(ph, pw),
            edges=edges
        )
        pieces.append(piece)

    if debug:
        print(f"Preprocessed {len(pieces)} pieces from {image_path}.")

    return pieces


# ----------------------------------------------------------
# Saving utilities
# ----------------------------------------------------------

def save_pieces(
        pieces: List[PuzzlePiece],
        out_dir: str,
        save_meta: bool = True
) -> None:
    """
    Save each piece as an image file and optionally a JSON metadata file.

    Parameters
    ----------
    pieces : List[PuzzlePiece]
        Pieces to save.
    out_dir : str
        Output directory path.
    save_meta : bool, optional
        If True, save metadata to 'pieces_meta.json'.
    """
    os.makedirs(out_dir, exist_ok=True)

    meta = []

    for piece in pieces:
        fname = f"piece_{piece.id:02d}.png"
        path = os.path.join(out_dir, fname)

        # Save rectified piece image
        cv2.imwrite(path, piece.image)

        # Collect metadata (you can add more fields if needed)
        item = {
            "id": piece.id,
            "file": fname,
            "size": piece.size,
            "canvas_corners": piece.canvas_corners.tolist(),
            "edges": {}
        }

        for edge_name, feat in piece.edges.items():
            item["edges"][edge_name] = {
                "mean_color": feat.mean_color
                # If you want the full histogram, uncomment:
                # "color_hist": feat.color_hist.tolist()
            }

        meta.append(item)

    if save_meta:
        meta_path = os.path.join(out_dir, "pieces_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")


# ----------------------------------------------------------
# Command-line test
# ----------------------------------------------------------

def main():
    """
    Simple CLI for testing.

    Example:
        python preprocess.py parrot_puzzle.png --out_dir parrot_pieces --debug
    """
    parser = argparse.ArgumentParser(
        description="Preprocess a puzzle canvas into rectified pieces + features."
    )
    parser.add_argument("image", help="Path to the input puzzle canvas image.")
    parser.add_argument(
        "--out_dir",
        default="pieces_out",
        help="Output directory to save extracted pieces."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of raw .rgb image (required if image is .rgb)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of raw .rgb image (required if image is .rgb)."
    )

    args = parser.parse_args()

    pieces = preprocess_puzzle_image(args.image, args.width, args.height, debug=args.debug)
    save_pieces(pieces, args.out_dir, save_meta=True)

    print(f"Done. Extracted {len(pieces)} pieces to '{args.out_dir}'.")


if __name__ == "__main__":
    main()
