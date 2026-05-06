"""
preview_layouts.py — 对 layout_outputs/ 里的 JSON 生成 2D 俯视预览图

用法:
  python preview_layouts.py                        # 预览 layout_outputs/ 所有文件
  python preview_layouts.py --input layout_outputs --max 20
  python preview_layouts.py --input layout_outputs --open   # 生成后自动打开文件夹
"""

import argparse
import json
import math
import os
import subprocess

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms
except ImportError:
    print("[ERROR] 请安装 matplotlib: pip install matplotlib")
    raise

# ── 配置 ──────────────────────────────────────────────────────────────
INPUT_DIR  = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\layout_outputs"
OUTPUT_DIR = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\layout_previews"

CAT_COLORS = {
    "Sofa":    "#4e79a7",
    "Chair":   "#f28e2b",
    "Table":   "#59a14f",
    "Bed":     "#e15759",
    "Cabinet": "#76b7b2",
    "Lamp":    "#edc948",
    "Other":   "#bab0ac",
}
# ──────────────────────────────────────────────────────────────────────


def draw_preview(layout: dict, out_path: str):
    room = layout.get('room', {})
    room_l = room.get('length', 4000.0) / 1000.0   # mm → m
    room_w = room.get('width',  3000.0) / 1000.0
    room_type = layout.get('room_type', '?')
    furn_list = layout.get('furniture', [])

    aspect = room_w / max(room_l, 0.01)
    fig_w = 9
    fig_h = max(fig_w * aspect + 1.0, 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 房间轮廓
    ax.add_patch(patches.Rectangle(
        (0, 0), room_l, room_w,
        fill=False, edgecolor='black', linewidth=2.0, zorder=1
    ))
    ax.set_xlim(-0.3, room_l + 0.3)
    ax.set_ylim(-0.3, room_w + 0.3)
    ax.set_aspect('equal')

    cat_counts = {}
    for item in furn_list:
        cat = item.get('source_category', 'Other')
        pos = item.get('position', [0, 0, 0])
        x = pos[0] / 1000.0
        y = pos[1] / 1000.0
        rz_deg = item.get('rotation_z', 0.0)

        # target_size = [L, W, H]（和 rotation_z 配合后已是平面尺寸）
        tsz = item.get('target_size', [500, 500, 500])
        # target_size 顺序与 rotation_z 已由 front_to_cad 对齐，直接取前两维
        tl = max(abs(tsz[0]) / 1000.0, 0.05)
        tw = max(abs(tsz[1]) / 1000.0, 0.05)

        color = CAT_COLORS.get(cat, '#bab0ac')
        fine_cat = item.get('_front_fine_cat', '')
        label = fine_cat if fine_cat else cat

        rect = patches.Rectangle(
            (-tl / 2, -tw / 2), tl, tw,
            linewidth=1.2,
            edgecolor='#333333',
            facecolor=color,
            alpha=0.75,
            zorder=2
        )
        tr = (transforms.Affine2D()
              .rotate_deg(-rz_deg)
              .translate(x, y)
              + ax.transData)
        rect.set_transform(tr)
        ax.add_patch(rect)

        # 正面方向箭头
        arrow_len = min(tl, tw) * 0.45
        dx = math.cos(math.radians(-rz_deg)) * arrow_len
        dy = math.sin(math.radians(-rz_deg)) * arrow_len
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='#222', lw=1.0),
                    zorder=3)

        # 类别文字标注
        ax.text(x, y, label, fontsize=5, ha='center', va='center',
                color='#111', zorder=4, clip_on=True)

        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # 图例（右上角文字）
    legend_text = "  ".join(f"{c}×{n}" for c, n in sorted(cat_counts.items()))
    ax.set_title(
        f"{room_type}   {room_l:.1f}m × {room_w:.1f}m   [{len(furn_list)} items]\n"
        f"{legend_text}",
        fontsize=8, pad=4
    )
    ax.set_xlabel("X (m)", fontsize=7)
    ax.set_ylabel("Y (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linestyle='--', zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="layout_outputs 2D 俯视预览")
    parser.add_argument('--input', default=INPUT_DIR, help='layout JSON 目录')
    parser.add_argument('--output', default=OUTPUT_DIR, help='预览图输出目录')
    parser.add_argument('--max', type=int, default=0, help='最多处理文件数（0=全部）')
    parser.add_argument('--open', action='store_true', help='生成后打开文件夹')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(f for f in os.listdir(args.input) if f.endswith('.json'))
    if args.max > 0:
        files = files[:args.max]

    print(f"共 {len(files)} 个文件，输出到: {args.output}")

    ok = 0
    for i, fname in enumerate(files):
        json_path = os.path.join(args.input, fname)
        try:
            with open(json_path, encoding='utf-8') as f:
                layout = json.load(f)
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")
            continue

        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(args.output, out_name)
        try:
            draw_preview(layout, out_path)
            ok += 1
            print(f"  [{i+1}/{len(files)}] {fname} -> {out_name}")
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")

    print(f"\n完成：{ok}/{len(files)} 张预览图")

    if args.open:
        subprocess.Popen(f'explorer "{os.path.abspath(args.output)}"')


if __name__ == '__main__':
    main()
