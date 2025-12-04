"""
animate_story_v4.py
最终完美版：
1. [关键] 引入 Hybrid 渲染：运动时使用 Erode 去黑边，归位时使用 cv2.rotate 保证像素级完美。
2. [关键] 修复了拼完后依然有细微缝隙或黑边的问题。
"""
import argparse
import json
import cv2
import numpy as np
import random
from preprocess import preprocess_puzzle_image

# --- 动画配置 ---
FPS = 30
SPEED_SCATTER = 30
SPEED_ROTATE = 12
SPEED_MOVE = 18
SCALE_FACTOR = 1.3
CANVAS_SCALE = 0.5  # 视频缩放比例 (0.5 = 导出 1/2 大小的视频，速度快)

# 颜色
BG_COLOR = (25, 25, 25)


def ease_in_out(t):
    if t < 0: return 0
    if t > 1: return 1
    return t * t * (3 - 2 * t)


def get_rotated_image_perfect(img, angle_idx):
    """
    【静止模式】使用整数旋转，保证无损、无黑边
    angle_idx: 0, 1, 2, 3 (代表 0, 90, 180, 270 度)
    """
    res = img.copy()
    # 修正：attemptA的 rotation=1 是顺时针90度 (ROTATE_90_CLOCKWISE)
    # 传入的 angle_idx 应该是 attemptA 中的 rotation 值
    k = int(angle_idx) % 4
    if k == 0: return res
    if k == 1: return cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
    if k == 2: return cv2.rotate(res, cv2.ROTATE_180)
    if k == 3: return cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return res


def get_rotated_image_smooth(img, angle_deg, scale=1.0):
    """
    【运动模式】使用仿射变换，带去黑边处理 (Erode)
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 使用透明边框
    rotated_rgba = cv2.warpAffine(img, M, (new_w, new_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))

    # --- [关键优化] 去黑边处理 ---
    # 分离通道
    b, g, r, a = cv2.split(rotated_rgba)

    # 1. 硬阈值：把半透明的边缘直接变成全透明或全不透明
    _, mask_hard = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY)

    # 2. 腐蚀 (Erode)：向内收缩 1 个像素，切掉被插值污染的边缘
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.erode(mask_hard, kernel, iterations=1)

    # 合并回去
    return cv2.merge([b, g, r, mask_clean])


class PieceSprite:
    def __init__(self, piece_data, start_cx, start_cy):
        self.id = piece_data.id

        # 预处理 RGBA
        b, g, r = cv2.split(piece_data.image)
        mask = piece_data.mask
        if len(mask.shape) == 2: mask = mask[:, :, None]
        if mask.dtype != np.uint8: mask = (mask * 255).astype(np.uint8)

        self.base_img = cv2.merge([b, g, r, mask])

        self.cx = float(start_cx)
        self.cy = float(start_cy)
        self.angle = 0.0  # 角度 (Float)
        self.scale = 1.0
        self.z_index = 0
        self.locked = False  # 是否已归位

    def draw(self, canvas):
        img = None

        # --- 智能切换渲染模式 ---
        # 如果缩放是 1.0 且角度接近 90 度的倍数，使用完美渲染
        # 注意：angle 是负的顺时针角度。
        # attemptA 的 rotation: 1 -> -90 deg

        is_integer_scale = abs(self.scale - 1.0) < 0.01

        # 归一化角度到 0-360
        norm_angle = (-self.angle) % 360
        # 检查是否接近 0, 90, 180, 270
        is_integer_angle = False
        rot_idx = 0

        for k in range(4):
            target = k * 90
            if abs(norm_angle - target) < 0.1 or abs(norm_angle - target) > 359.9:
                is_integer_angle = True
                rot_idx = k
                break

        if is_integer_scale and is_integer_angle:
            # 【完美模式】
            img = get_rotated_image_perfect(self.base_img, rot_idx)
        else:
            # 【运动模式】(带 Erode 去黑边)
            img = get_rotated_image_smooth(self.base_img, self.angle, self.scale)

        # 下面是通用的贴图逻辑
        h, w = img.shape[:2]
        top_left_x = int(self.cx - w / 2)
        top_left_y = int(self.cy - h / 2)

        canvas_h, canvas_w = canvas.shape[:2]

        y1, y2 = max(0, top_left_y), min(canvas_h, top_left_y + h)
        x1, x2 = max(0, top_left_x), min(canvas_w, top_left_x + w)

        if y1 >= y2 or x1 >= x2: return

        iy1, iy2 = y1 - top_left_y, y2 - top_left_y
        ix1, ix2 = x1 - top_left_x, x2 - top_left_x

        src_patch = img[iy1:iy2, ix1:ix2]
        dst_patch = canvas[y1:y2, x1:x2]

        # 简单的硬覆盖 (模拟 attemptA)
        alpha = src_patch[:, :, 3]
        mask_bool = alpha > 128

        dst_patch[mask_bool] = src_patch[mask_bool, :3]
        canvas[y1:y2, x1:x2] = dst_patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("json")
    parser.add_argument("--out", default="puzzle_perfect.mp4")
    args = parser.parse_args()

    print(">>> Loading...")
    pieces_raw, _ = preprocess_puzzle_image(args.image)

    with open(args.json, 'r') as f:
        solution_steps = json.load(f)

    # 布局计算
    sol_xs = [s['x'] for s in solution_steps]
    sol_ys = [s['y'] for s in solution_steps]
    puzzle_w = max(sol_xs) + 500
    puzzle_h = max(sol_ys) + 500

    area_puzzle_w = int(puzzle_w * 1.0)
    area_scatter_w = int(puzzle_w * 0.8)
    full_w = area_puzzle_w + area_scatter_w
    full_h = int(puzzle_h * 1.2)

    offset_x = 100
    offset_y = 100

    sprites = []
    sprite_map = {}

    scatter_x_min = area_puzzle_w + 50
    scatter_x_max = full_w - 50
    scatter_y_min = 50
    scatter_y_max = full_h - 100

    for p in pieces_raw:
        ox, oy = p.canvas_corners[0]
        h, w = p.image.shape[:2]
        cx = ox + w / 2 + offset_x
        cy = oy + h / 2 + offset_y
        sp = PieceSprite(p, cx, cy)
        sprites.append(sp)
        sprite_map[p.id] = sp

    vid_w = int(full_w * CANVAS_SCALE)
    vid_h = int(full_h * CANVAS_SCALE)
    if vid_w % 2 != 0: vid_w += 1
    if vid_h % 2 != 0: vid_h += 1

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (vid_w, vid_h))

    def render_frame():
        canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        canvas[:, :] = BG_COLOR
        cv2.line(canvas, (area_puzzle_w, 0), (area_puzzle_w, full_h), (50, 50, 50), 2)

        sprites.sort(key=lambda s: s.z_index)
        for sp in sprites:
            sp.draw(canvas)

        frame = cv2.resize(canvas, (vid_w, vid_h))
        vw.write(frame)

    # === Animation Loop ===

    print(">>> Phase 0: Original")
    for _ in range(FPS * 1): render_frame()

    print(">>> Phase 1: Scatter")
    start_pos = {sp.id: (sp.cx, sp.cy) for sp in sprites}
    target_pos = {}
    for sp in sprites:
        target_pos[sp.id] = (
            random.randint(scatter_x_min, scatter_x_max),
            random.randint(scatter_y_min, scatter_y_max)
        )

    for f in range(SPEED_SCATTER):
        t = ease_in_out(f / SPEED_SCATTER)
        for sp in sprites:
            sx, sy = start_pos[sp.id]
            tx, ty = target_pos[sp.id]
            sp.cx = sx + (tx - sx) * t
            sp.cy = sy + (ty - sy) * t
        render_frame()

    print(f">>> Phase 2: Solving {len(solution_steps)} steps")

    for idx, step in enumerate(solution_steps):
        pid = step['id']
        if pid not in sprite_map: continue
        sp = sprite_map[pid]
        sp.z_index = 999

        # 计算完美坐标
        final_rot_idx = step['r']  # 0,1,2,3
        h_raw, w_raw = sp.base_img.shape[:2]

        # 计算旋转后的宽高
        if final_rot_idx % 2 == 1:
            w_final, h_final = h_raw, w_raw
        else:
            w_final, h_final = w_raw, h_raw

        target_cx = step['x'] + w_final / 2 + offset_x
        target_cy = step['y'] + h_final / 2 + offset_y
        target_angle = -90.0 * final_rot_idx

        current_cx, current_cy = sp.cx, sp.cy

        # Lift
        for f in range(8):
            t = f / 8.0
            sp.scale = 1.0 + (SCALE_FACTOR - 1.0) * ease_in_out(t)
            render_frame()

        # Rotate
        if abs(sp.angle - target_angle) > 0.1:
            start_ang = sp.angle
            for f in range(SPEED_ROTATE):
                t = ease_in_out(f / SPEED_ROTATE)
                sp.angle = start_ang + (target_angle - start_ang) * t
                render_frame()
            sp.angle = target_angle

            # Move
        for f in range(SPEED_MOVE):
            t = ease_in_out(f / SPEED_MOVE)
            sp.cx = current_cx + (target_cx - current_cx) * t
            sp.cy = current_cy + (target_cy - current_cy) * t
            render_frame()

        # Drop
        for f in range(5):
            t = f / 5.0
            sp.scale = SCALE_FACTOR - (SCALE_FACTOR - 1.0) * ease_in_out(t)
            render_frame()

        # 【关键】强制锁定到完美状态
        sp.scale = 1.0
        sp.cx = target_cx
        sp.cy = target_cy
        sp.angle = target_angle
        sp.locked = True
        sp.z_index = 1

        # 绘制这一帧 (此时 draw 内部会切换到 Perfect Mode)
        render_frame()

        if idx % 10 == 0:
            print(f"  Placed {idx + 1}/{len(solution_steps)}")

    print(">>> Phase 3: Final")
    for _ in range(FPS * 2): render_frame()

    vw.release()
    print(f"Done! Saved to {args.out}")


if __name__ == "__main__":
    main()
