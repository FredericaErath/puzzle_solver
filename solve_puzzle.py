#!/usr/bin/env python3
"""
solve_puzzle.py

统一入口脚本：

1) 对输入 image 调用 preprocess_puzzle_image 得到 pieces。
2) 使用 is_regular_grid(pieces) 判断规则 / 不规则 grid。
3) 规则 grid -> 调用 attemptA.solve_regular_from_pieces
   不规则 grid -> 调用 attemptB.solve_irregular_from_pieces
4) 根据 outdir 和原文件名输出：
   - image_name_solution.png
   - image_name_solution.json  (由 attemptA/B 自己保存)
   - image_name_solution.mp4   (如果 --animate，则调用 animate.py 生成)

用法：
  python solve_puzzle.py <image_path> --outdir <outdir> --animate
"""

import argparse
import os
import sys
import subprocess

from preprocess import preprocess_puzzle_image, is_regular_grid
from attemptA import solve_regular_from_pieces
from attemptB import solve_irregular_from_pieces


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified puzzle solver (regular via attemptA, irregular via attemptB, optional animation)."
    )
    parser.add_argument("image_path", help="Input puzzle image (PNG/RGB)")
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate animation MP4 using animate.py",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs for solvers",
    )
    return parser.parse_args()


def run_animation(image_path: str, solution_json: str, video_out: str):
    """
    调用 animate.py 生成动画：
      python animate.py <image> <solution_json> --out <video_out>
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    animate_script = os.path.join(script_dir, "animate.py")  # 确保文件名与实际一致

    if not os.path.exists(animate_script):
        print(f"[Driver] animate.py not found at {animate_script}, skip animation.")
        return

    if not os.path.exists(solution_json):
        print(f"[Driver] Solution JSON not found: {solution_json}, skip animation.")
        return

    cmd = [
        sys.executable,
        animate_script,
        image_path,
        solution_json,
        "--out",
        video_out,
    ]

    print("[Driver] Running animation:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    image_path = args.image_path
    outdir = args.outdir or "."

    os.makedirs(outdir, exist_ok=True)

    # 拆出原始文件名
    basename = os.path.basename(image_path)            # e.g. "parrot.png"
    image_name, _ = os.path.splitext(basename)         # e.g. "parrot"

    # 约定输出文件名
    solution_png = os.path.join(outdir, f"{image_name}_solution.png")
    solution_json = os.path.join(outdir, f"{image_name}_solution.json")
    solution_mp4 = os.path.join(outdir, f"{image_name}_solution.mp4")

    print(f"[Driver] Input  : {image_path}")
    print(f"[Driver] Outdir : {outdir}")
    print(f"[Driver] Output : {solution_png}, {solution_json}")
    if args.animate:
        print(f"[Driver] Video  : {solution_mp4}")

    # 1) 预处理
    print("[Driver] Preprocessing...")
    pieces, pre_config = preprocess_puzzle_image(image_path, debug=False)
    if not pieces:
        print("[Driver] No pieces detected, abort.")
        return 1

    # 2) 规则 / 不规则 grid 判断（允许 w/h 互换）
    regular = is_regular_grid(pieces)
    print(f"[Driver] Grid type decision: {'REGULAR (attemptA)' if regular else 'IRREGULAR (attemptB)'}")

    # 3) 调用对应 solver 的 wrapper
    if regular:
        # attemptA: wrapper 已经负责 estimate_canvas / auto_tune_and_solve / save_result
        solve_regular_from_pieces(
            pieces=pieces,
            pre_config=pre_config,
            out_path=solution_png,
            target_w=None,
            target_h=None,
            debug=args.debug,
            image_debug=False,   # 需要 step 图可以改成 True
            debug_dir=None,
        )
    else:
        # attemptB: 使用全局优化求解不规则 grid
        solve_irregular_from_pieces(
            raw_pieces=pieces,
            out_path=solution_png,
            debug=args.debug,
        )

    # 检查是否生成成功
    if not os.path.exists(solution_png) or not os.path.exists(solution_json):
        print("[Driver] Solver did not produce expected PNG/JSON, stop here.")
        return 1

    # 4) 可选：调用 animate.py 生成 mp4
    if args.animate:
        try:
            run_animation(image_path, solution_json, solution_mp4)
        except subprocess.CalledProcessError as e:
            print(f"[Driver] Animation failed: {e}")
            return 1

    print("[Driver] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
