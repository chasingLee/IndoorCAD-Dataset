"""
select_templates.py — 扫描 3D-FRONT → 打分 → 生成预览图 → 手动选择模板

流程：
  1. 扫描 3D-FRONT JSON，提取所有候选房间，用多维度指标打分
  2. 按分数排序，生成 2D 俯视预览图（matplotlib，无需 STEP/渲染）
  3. 打开预览图目录，用户输入想要的模板编号
  4. 保存选定模板到 TEMPLATE_DIR，供 layout_variants.py 使用

用法:
  python select_templates.py                        # 扫描并生成预览
  python select_templates.py --max-files 300        # 扫描更多文件
  python select_templates.py --select 0,2,5,7       # 直接选定编号（跳过交互）
  python select_templates.py --top 30               # 自动选 top-N，不交互
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

# ================== 配置区 ==================
FRONT_JSON_DIR  = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\front_scene_layout"
MODEL_INFO_PATH = "D:\\Lzm_Temp_Data\\3D-FUTURE-model\\model_info.json"
TEMPLATE_DIR    = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\layout_templates"
PREVIEW_DIR     = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\template_previews"

MAX_FRONT_FILES  = 1000
MIN_USEFUL       = 3       # 最少有效家具数（过滤门槛）
MAX_CANDIDATES   = 500     # 最多保留多少候选（打分后取 top-N 画图）
RANDOM_SEED      = 42

# 所有有家具的房间类型（全部开放）
TARGET_ROOM_TYPES = {
    "Bedroom", "MasterBedroom", "SecondBedroom", "BedRoom",
    "LivingRoom", "LivingDiningRoom", "DiningRoom",
    "Library", "KidsRoom", "ElderlyRoom", "Lounge",
    "NannyRoom",
}

# 各房间类型期望有的主要家具类别（完整性打分用）
ROOM_EXPECTED = {
    "Bedroom":         {"Bed": 2.0, "Cabinet": 1.0, "Chair": 0.5},
    "MasterBedroom":   {"Bed": 2.0, "Cabinet": 1.0, "Chair": 0.5},
    "SecondBedroom":   {"Bed": 2.0, "Cabinet": 0.5},
    "BedRoom":         {"Bed": 2.0, "Cabinet": 0.5},
    "KidsRoom":        {"Bed": 1.5, "Chair": 0.5, "Table": 0.5},
    "ElderlyRoom":     {"Bed": 1.5, "Chair": 0.5},
    "NannyRoom":       {"Bed": 1.5},
    "LivingRoom":      {"Sofa": 2.0, "Table": 1.0, "Cabinet": 0.5},
    "LivingDiningRoom":{"Sofa": 1.5, "Table": 1.5, "Chair": 0.5, "Cabinet": 0.5},
    "DiningRoom":      {"Table": 2.0, "Chair": 1.5},
    "Library":         {"Chair": 1.0, "Table": 1.0, "Cabinet": 1.0},
    "Lounge":          {"Sofa": 1.5, "Chair": 0.5, "Table": 0.5},
}

# 房间面积合理范围 m²（用于密度计算）
ROOM_AREA_RANGE = {
    "Bedroom": (8, 30), "MasterBedroom": (12, 40), "SecondBedroom": (8, 25),
    "BedRoom": (8, 30), "KidsRoom": (6, 20), "ElderlyRoom": (8, 25),
    "NannyRoom": (6, 16),
    "LivingRoom": (15, 50), "LivingDiningRoom": (18, 60),
    "DiningRoom": (10, 35), "Library": (8, 25), "Lounge": (12, 40),
}
# ============================================


# ── 分类逻辑（与 front_to_cad.py 保持一致）────────────────────────────

SUPER_CAT_MAP = {
    "Sofa": "Sofa", "Chair": "Chair", "Table": "Table", "Bed": "Bed",
    "Cabinet/Shelf/Desk": "Cabinet", "Pier/Stool": "Chair",
    "Lighting": "Lamp", "Others": None,
}
FINE_CAT_OVERRIDE = {
    "Desk": "Table", "Dining Table": "Table", "Coffee Table": "Table",
    "Tea Table": "Table", "Corner/Side Table": "Table", "Round End Table": "Table",
    "Dressing Table": "Table",
    "armchair": "Chair", "Dining Chair": "Chair",
    "Lounge Chair / Cafe Chair / Office Chair": "Chair",
    "Lounge Chair / Book-chair / Computer Chair": "Chair",
    "Dressing Chair": "Chair", "Barstool": "Chair",
    "Footstool / Sofastool / Bed End Stool / Stool": "Chair",
    "Three-Seat / Multi-seat Sofa": "Sofa",
    "Three-Seat / Multi-person sofa": "Sofa",
    "Loveseat Sofa": "Sofa", "L-shaped Sofa": "Sofa",
    "U-shaped Sofa": "Sofa", "Lazy Sofa": "Sofa",
    "Chaise Longue Sofa": "Sofa", "Two-seat Sofa": "Sofa", "Couch Bed": "Sofa",
    "King-size Bed": "Bed", "Double Bed": "Bed", "Single bed": "Bed",
    "Kids Bed": "Bed", "Bed Frame": "Bed",
    "Wardrobe": "Cabinet", "Nightstand": "Cabinet",
    "Drawer Chest / Corner cabinet": "Cabinet",
    "Bookcase / jewelry Armoire": "Cabinet", "TV Stand": "Cabinet",
    "Sideboard / Side Cabinet / Console Table": "Cabinet",
    "Sideboard / Side Cabinet / Console": "Cabinet",
    "Shoe Cabinet": "Cabinet", "Wine Cabinet": "Cabinet",
    "Children Cabinet": "Cabinet", "Shelf": "Cabinet", "Floor Lamp": "Lamp",
}
SKIP_FINE_CATS    = frozenset(["Pendant Lamp", "Ceiling Lamp", "Wall Lamp",
                                "Wine Cooler", "Bar", "Bunk Bed",
                                "Folding chair", "Hanging Chair"])
CEILING_FINE_CATS = frozenset(["Pendant Lamp", "Ceiling Lamp"])
FLOOR_ATTACH_MAX_Z   = 800.0
CEILING_ATTACH_RATIO = 0.70


def _get_local_cat(jid: str, model_info: dict) -> tuple[str, str]:
    entry = model_info.get(jid)
    if not entry:
        return "Other", ""
    fine = entry.get("category") or ""
    if fine in FINE_CAT_OVERRIDE:
        return FINE_CAT_OVERRIDE[fine], fine
    super_cat = entry.get("super-category", "")
    return SUPER_CAT_MAP.get(super_cat) or "Other", fine


def _should_skip(local_cat, fine_cat, cad_z, room_h, target_h):
    if not local_cat or local_cat == "Other":
        return True
    if fine_cat in SKIP_FINE_CATS:
        return True
    if fine_cat in CEILING_FINE_CATS:
        return cad_z < room_h * CEILING_ATTACH_RATIO
    return (cad_z - target_h / 2.0) > FLOOR_ATTACH_MAX_Z


# ── 打分系统 ─────────────────────────────────────────────────────────────

def score_room(room_type: str, furniture: list, room_len_m: float, room_wid_m: float) -> dict:
    """
    对一个候选房间打分，返回各维度分数和总分（均归一化到 [0,1]）。

    维度：
      completeness  0.35  — 房间类型期望的主要家具是否存在
      diversity     0.20  — 有效家具类别种数
      density       0.20  — 家具占地面积 / 房间面积，在合理区间内得满分
      distribution  0.15  — 家具位置在房间内的空间分散程度
      count         0.10  — 有效家具总数（对数尺度，上限 15 件）
    """
    cats = [it['local_cat'] for it in furniture]
    cat_set = set(cats)
    n = len(furniture)

    # ── completeness ──
    expected = ROOM_EXPECTED.get(room_type, {})
    if expected:
        comp = sum(w for cat, w in expected.items() if cat in cat_set)
        comp_max = sum(expected.values())
        completeness = min(comp / comp_max, 1.0)
    else:
        completeness = 0.5  # 未知类型给中间分

    # ── diversity ──
    useful_cats = [c for c in cat_set if c not in ('Other', 'Lamp')]
    diversity = min(len(useful_cats) / 5.0, 1.0)   # 5种类别得满分

    # ── density ──
    room_area = room_len_m * room_wid_m
    furn_area = 0.0
    for it in furniture:
        t = it['target_size']  # [length_mm, height_mm, width_mm]
        furn_area += (t[0] / 1000.0) * (t[2] / 1000.0)
    if room_area > 0:
        ratio = furn_area / room_area
        lo, hi = 0.10, 0.65
        if lo <= ratio <= hi:
            density = 1.0
        elif ratio < lo:
            density = ratio / lo
        else:
            density = max(0.0, 1.0 - (ratio - hi) / hi)
    else:
        density = 0.0

    # ── distribution：家具XY位置的归一化标准差 ──
    if n >= 2:
        xs = [it['position'][0] for it in furniture]
        ys = [it['position'][1] for it in furniture]
        mx, my = sum(xs)/n, sum(ys)/n
        vx = sum((x-mx)**2 for x in xs) / n
        vy = sum((y-my)**2 for y in ys) / n
        std = math.sqrt((vx + vy) / 2)
        # 归一化：期望标准差约为房间对角线的 20%
        diag = math.sqrt((room_len_m*1000)**2 + (room_wid_m*1000)**2)
        distribution = min(std / (diag * 0.20 + 1e-6), 1.0)
    else:
        distribution = 0.0

    # ── count ──
    count_score = min(math.log(n + 1) / math.log(16), 1.0)

    total = (0.35 * completeness +
             0.20 * diversity    +
             0.20 * density      +
             0.15 * distribution +
             0.10 * count_score)

    return {
        'total':        round(total, 3),
        'completeness': round(completeness, 3),
        'diversity':    round(diversity, 3),
        'density':      round(density, 3),
        'distribution': round(distribution, 3),
        'count':        round(count_score, 3),
        'n_furniture':  n,
        'categories':   sorted(cat_set),
    }


# ── 2D 俯视图预览 ─────────────────────────────────────────────────────────

# 类别配色
CAT_COLORS = {
    'Bed':     '#4e79a7',
    'Sofa':    '#f28e2b',
    'Chair':   '#76b7b2',
    'Table':   '#59a14f',
    'Cabinet': '#edc948',
    'Lamp':    '#b07aa1',
    'Other':   '#cccccc',
}


def _draw_top_view(ax, furn, room_L, room_W, legend_handles):
    """俯视图（XY平面，主视图）。"""
    import matplotlib.patches as mpatches
    from matplotlib.transforms import Affine2D

    margin = 0.3
    ax.set_xlim(-room_L/2 - margin, room_L/2 + margin)
    ax.set_ylim(-room_W/2 - margin, room_W/2 + margin)
    ax.set_aspect('equal')
    ax.add_patch(mpatches.Rectangle(
        (-room_L/2, -room_W/2), room_L, room_W,
        linewidth=2, edgecolor='#333333', facecolor='#f5f5f0', zorder=0))

    for it in furn:
        cat   = it['local_cat']
        pos   = it['position']
        t     = it['target_size']   # [L_mm, H_mm, W_mm]
        angle = it['rotation_z']
        cx, cy = pos[0]/1000, pos[1]/1000
        fl = max(t[0]/1000, 0.05)   # CAD X = length
        fw = max(t[2]/1000, 0.05)   # CAD Y = width
        color = CAT_COLORS.get(cat, '#cccccc')

        rect = mpatches.Rectangle((-fl/2, -fw/2), fl, fw,
                                   linewidth=0.8, edgecolor='#444',
                                   facecolor=color, alpha=0.75, zorder=2)
        rect.set_transform(Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData)
        ax.add_patch(rect)

        arrow_len = min(fl, fw) * 0.4
        dx = arrow_len * math.cos(math.radians(angle))
        dy = arrow_len * math.sin(math.radians(angle))
        ax.annotate('', xy=(cx+dx, cy+dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='#222', lw=0.8), zorder=3)
        ax.text(cx, cy, cat[:3], ha='center', va='center',
                fontsize=5, color='#111', zorder=4)

        if cat not in legend_handles:
            legend_handles[cat] = mpatches.Patch(color=color, label=cat)

    ax.set_xlabel('X (m)', fontsize=7)
    ax.set_ylabel('Y (m)', fontsize=7)
    ax.set_title('Top View (XY)', fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def _draw_front_view(ax, furn, room_L, room_H):
    """前视图（XZ平面：X横轴，Z=高度纵轴）。"""
    import matplotlib.patches as mpatches

    margin_x, margin_z = 0.3, 0.2
    ax.set_xlim(-room_L/2 - margin_x, room_L/2 + margin_x)
    ax.set_ylim(-margin_z, room_H + margin_z)
    ax.set_aspect('equal')
    # 地板线 + 天花线
    ax.axhline(0,      color='#555', lw=1.5, zorder=0)
    ax.axhline(room_H, color='#999', lw=1.0, ls='--', zorder=0)
    ax.add_patch(mpatches.Rectangle(
        (-room_L/2, 0), room_L, room_H,
        linewidth=1.5, edgecolor='#555', facecolor='#fafaf8', zorder=0))

    for it in furn:
        cat = it['local_cat']
        pos = it['position']
        t   = it['target_size']   # [L_mm, H_mm, W_mm]
        cx  = pos[0] / 1000
        cz  = pos[2] / 1000       # 高度方向
        fl  = max(t[0]/1000, 0.05)
        fh  = max(t[1]/1000, 0.05)
        color = CAT_COLORS.get(cat, '#cccccc')
        # 前视图不旋转（忽略 rotation_z，展示轮廓高度）
        ax.add_patch(mpatches.Rectangle(
            (cx - fl/2, cz), fl, fh,
            linewidth=0.7, edgecolor='#555', facecolor=color, alpha=0.6, zorder=2))

    ax.set_xlabel('X (m)', fontsize=7)
    ax.set_ylabel('Z / Height (m)', fontsize=7)
    ax.set_title('Front View (XZ)', fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def _draw_side_view(ax, furn, room_W, room_H):
    """侧视图（YZ平面：Y横轴，Z=高度纵轴）。"""
    import matplotlib.patches as mpatches

    margin_y, margin_z = 0.3, 0.2
    ax.set_xlim(-room_W/2 - margin_y, room_W/2 + margin_y)
    ax.set_ylim(-margin_z, room_H + margin_z)
    ax.set_aspect('equal')
    ax.axhline(0,      color='#555', lw=1.5, zorder=0)
    ax.axhline(room_H, color='#999', lw=1.0, ls='--', zorder=0)
    ax.add_patch(mpatches.Rectangle(
        (-room_W/2, 0), room_W, room_H,
        linewidth=1.5, edgecolor='#555', facecolor='#fafaf8', zorder=0))

    for it in furn:
        cat = it['local_cat']
        pos = it['position']
        t   = it['target_size']
        cy  = pos[1] / 1000
        cz  = pos[2] / 1000
        fw  = max(t[2]/1000, 0.05)
        fh  = max(t[1]/1000, 0.05)
        color = CAT_COLORS.get(cat, '#cccccc')
        ax.add_patch(mpatches.Rectangle(
            (cy - fw/2, cz), fw, fh,
            linewidth=0.7, edgecolor='#555', facecolor=color, alpha=0.6, zorder=2))

    ax.set_xlabel('Y (m)', fontsize=7)
    ax.set_ylabel('Z / Height (m)', fontsize=7)
    ax.set_title('Side View (YZ)', fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def draw_preview(candidate: dict, out_path: str, idx: int):
    """生成三视图预览图：俯视图（主）+ 前视图 + 侧视图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    room      = candidate['room']
    furn      = candidate['furniture']
    scores    = candidate['scores']
    room_type = candidate['room_type']

    L = room['length'] / 1000   # m
    W = room['width']  / 1000
    H = room['height'] / 1000

    # 布局：左大（俯视），右上（前视），右下（侧视）
    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1],
                           hspace=0.35, wspace=0.30)
    ax_top   = fig.add_subplot(gs[:, 0])   # 俯视占满左列
    ax_front = fig.add_subplot(gs[0, 1])   # 前视右上
    ax_side  = fig.add_subplot(gs[1, 1])   # 侧视右下

    legend_handles: dict = {}
    _draw_top_view(ax_top,   furn, L, W, legend_handles)
    _draw_front_view(ax_front, furn, L, H)
    _draw_side_view(ax_side,   furn, W, H)

    # 图例放在俯视图右上角
    if legend_handles:
        ax_top.legend(handles=list(legend_handles.values()),
                      loc='upper right', fontsize=7, framealpha=0.85)

    # 总标题
    sc = scores
    suptitle = (
        f"#{idx:03d}  {room_type}  "
        f"{L:.1f}m × {W:.1f}m × {H:.1f}m  "
        f"n={sc['n_furniture']}  "
        f"Score={sc['total']:.3f}  |  "
        f"comp={sc['completeness']:.2f}  "
        f"div={sc['diversity']:.2f}  "
        f"dens={sc['density']:.2f}  "
        f"dist={sc['distribution']:.2f}"
    )
    fig.suptitle(suptitle, fontsize=9, y=0.995)

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── 交互式多轮选择 ───────────────────────────────────────────────────────

def _interactive_select(top_cands: list, preview_dir: str) -> list | None:
    """
    多轮交互式模板选择。

    支持的命令（每轮可混合使用）：
      数字编号          直接选定，如: 0 3 7 12  或  0,3,7,12
      all               选全部候选
      +<idx>            追加选定某编号
      -<idx>            取消已选某编号
      list / ls         显示当前已选列表
      info <idx>        打印该候选的详细分数
      open              重新打开预览图目录
      clear             清空当前所有选择
      done / ok / q     确认并退出（selected_indices 为 None 表示放弃）
      quit / exit / q!  放弃退出（返回 None）

    直接按回车（空行）= done。
    """
    import subprocess

    selected: set[int] = set()

    # 自动打开预览目录
    try:
        subprocess.Popen(f'explorer "{preview_dir}"')
    except Exception:
        pass

    n = len(top_cands)
    print("\n" + "="*65)
    print(f"预览图目录: {preview_dir}")
    print(f"候选数量: {n}  (编号 0 ~ {n-1})")
    print("─"*65)
    print("输入命令选择模板（多条命令用空格或逗号分隔，直接回车=完成）：")
    print("  <编号>       追加选定（如: 0 3 7  或  0,3,7）")
    print("  all          选全部")
    print("  +<编号>      追加  ｜  -<编号>  取消")
    print("  list/ls      查看已选  ｜  info <编号>  查看详情")
    print("  open         重新打开预览目录")
    print("  clear        清空选择")
    print("  done/ok      确认保存  ｜  quit/q!  放弃退出")
    print("="*65)

    while True:
        try:
            raw = input(f"\n[已选 {len(selected)} 个] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n中断，放弃选择。")
            return None

        if raw == '':
            # 空行 = 完成
            if not selected:
                confirm = input("当前未选任何模板，确认退出？(y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            break

        tokens = raw.replace(',', ' ').split()
        i = 0
        while i < len(tokens):
            tok = tokens[i].lower()

            # ── done / quit ──
            if tok in ('done', 'ok'):
                return sorted(selected)
            if tok in ('quit', 'exit', 'q!'):
                print("放弃选择，不保存。")
                return None

            # ── 特殊命令 ──
            if tok == 'all':
                selected = set(range(n))
                print(f"  已全选 {n} 个")
                i += 1; continue

            if tok in ('list', 'ls'):
                if not selected:
                    print("  （当前无选择）")
                else:
                    for si in sorted(selected):
                        c = top_cands[si]
                        sc = c['scores']
                        print(f"  #{si:03d}  {c['room_type']:20s}  "
                              f"score={sc['total']:.3f}  n={sc['n_furniture']}")
                i += 1; continue

            if tok == 'info':
                i += 1
                if i < len(tokens) and tokens[i].isdigit():
                    si = int(tokens[i])
                    if 0 <= si < n:
                        c  = top_cands[si]
                        sc = c['scores']
                        cats = Counter(it['local_cat'] for it in c['furniture'])
                        print(f"  #{si:03d}  {c['room_type']}  "
                              f"{c['room']['length']/1000:.1f}m×{c['room']['width']/1000:.1f}m  "
                              f"n={sc['n_furniture']}")
                        print(f"    Score={sc['total']:.3f}  comp={sc['completeness']:.2f}  "
                              f"div={sc['diversity']:.2f}  dens={sc['density']:.2f}  "
                              f"dist={sc['distribution']:.2f}")
                        print(f"    cats={dict(cats)}")
                        print(f"    file={c.get('source_file','?')}")
                    else:
                        print(f"  [WARN] 编号 {si} 超出范围 (0~{n-1})")
                    i += 1
                else:
                    print("  用法: info <编号>")
                continue

            if tok == 'open':
                try:
                    subprocess.Popen(f'explorer "{preview_dir}"')
                    print("  已打开预览目录")
                except Exception as e:
                    print(f"  [WARN] 无法打开: {e}")
                i += 1; continue

            if tok == 'clear':
                selected.clear()
                print("  已清空选择")
                i += 1; continue

            # ── +N 追加 ──
            if tok.startswith('+') and tok[1:].isdigit():
                si = int(tok[1:])
                if 0 <= si < n:
                    selected.add(si)
                    print(f"  +{si:03d}  {top_cands[si]['room_type']}  "
                          f"score={top_cands[si]['scores']['total']:.3f}")
                else:
                    print(f"  [WARN] 编号 {si} 超出范围")
                i += 1; continue

            # ── -N 取消 ──
            if tok.startswith('-') and tok[1:].isdigit():
                si = int(tok[1:])
                selected.discard(si)
                print(f"  -{si:03d} 已取消")
                i += 1; continue

            # ── 纯数字：追加到已选集合（不替换） ──
            if tok.isdigit():
                # 收集连续数字 token，批量追加
                batch = []
                while i < len(tokens) and tokens[i].isdigit():
                    batch.append(int(tokens[i]))
                    i += 1
                valid = [si for si in batch if 0 <= si < n]
                invalid = [si for si in batch if not (0 <= si < n)]
                if valid:
                    selected.update(valid)
                    print(f"  追加: {sorted(valid)}  → 已选合计 {len(selected)} 个")
                if invalid:
                    print(f"  [WARN] 超出范围，忽略: {invalid}")
                continue

            print(f"  [?] 未知命令: {tok!r}，输入 list 查看已选，done 完成")
            i += 1

    return sorted(selected)


# ── 模板提取（核心扫描）─────────────────────────────────────────────────

def extract_candidates(front_json_dir: str, model_info: dict,
                       max_files: int) -> list[dict]:
    """扫描所有 3D-FRONT 文件，返回所有候选房间（未筛选）。"""
    from scipy.spatial.transform import Rotation as ScipyR

    def _yaw(rot_quat):
        try:
            fwd = ScipyR.from_quat(rot_quat).apply([0, 0, 1])
            return float((math.degrees(math.atan2(fwd[2], fwd[0])) + 360) % 360)
        except Exception:
            return 0.0

    def _parse_size(f_info, scale):
        abs_s = [abs(float(s)) for s in scale]
        sz = f_info.get('size') or f_info.get('bbox') or [1, 1, 1]
        if not isinstance(sz, list) or len(sz) < 3:
            sz = [1, 1, 1]
        return [float(sz[i]) * abs_s[i] for i in range(3)]

    all_files = sorted(f for f in os.listdir(front_json_dir) if f.endswith('.json'))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_files)

    candidates = []
    for filename in all_files[:max_files]:
        try:
            with open(os.path.join(front_json_dir, filename), encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        furniture_dict = {
            item['uid']: item
            for item in data.get('furniture', []) if item.get('uid')
        }

        for room in data.get('scene', {}).get('room', []):
            room_type = room.get('type', 'Unknown')
            if room_type not in TARGET_ROOM_TYPES:
                continue

            # 房间包围盒
            positions = [
                child['pos'] for child in room.get('children', [])
                if isinstance(child.get('pos'), list) and len(child['pos']) == 3
            ]
            if positions:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                zs = [p[2] for p in positions]
                bounds = dict(min_x=min(xs), max_x=max(xs),
                              min_y=min(ys), max_y=max(ys),
                              min_z=min(zs), max_z=max(zs))
            else:
                sz  = room.get('size', [6.0, 2.8, 5.0])
                pos = room.get('pos',  [0.0, 0.0, 0.0])
                bounds = dict(min_x=pos[0]-sz[0]/2, max_x=pos[0]+sz[0]/2,
                              min_y=pos[1],          max_y=pos[1]+sz[1],
                              min_z=pos[2]-sz[2]/2,  max_z=pos[2]+sz[2]/2)

            center_x = (bounds['min_x'] + bounds['max_x']) / 2
            center_z = (bounds['min_z'] + bounds['max_z']) / 2
            floor_y  = bounds['min_y']
            room_len = max((bounds['max_x'] - bounds['min_x']) * 1000 + 1000, 4000.0)
            room_wid = max((bounds['max_z'] - bounds['min_z']) * 1000 + 1000, 3000.0)
            room_h   = max((bounds['max_y'] - bounds['min_y']) * 1000 + 300,  2600.0)

            furn_items = []
            for child in room.get('children', []):
                ref_id = child.get('ref')
                if not ref_id or ref_id not in furniture_dict:
                    continue
                f_info = furniture_dict[ref_id]
                jid    = f_info.get('jid', '')
                local_cat, fine_cat = _get_local_cat(jid, model_info)

                scale = child.get('scale', [1, 1, 1])
                if not isinstance(scale, list) or len(scale) != 3:
                    scale = [1, 1, 1]
                raw_sz = _parse_size(f_info, scale)
                t_len  = raw_sz[2] * 1000
                t_h    = raw_sz[1] * 1000
                t_wid  = raw_sz[0] * 1000

                pos   = child.get('pos', [0, 0, 0])
                cad_x = (pos[0] - center_x) * 1000
                cad_y = (pos[2] - center_z) * 1000
                cad_z = (pos[1] - floor_y)  * 1000

                if _should_skip(local_cat, fine_cat, cad_z, room_h, t_h):
                    continue

                rot = child.get('rot', [0, 0, 0, 1])
                if not isinstance(rot, list) or len(rot) != 4:
                    rot = [0, 0, 0, 1]

                furn_items.append({
                    'local_cat':   local_cat,
                    'fine_cat':    fine_cat,
                    'position':    [round(cad_x, 1), round(cad_y, 1), round(cad_z, 1)],
                    'rotation_z':  round(_yaw(rot), 1),
                    'target_size': [round(t_len, 1), round(t_h, 1), round(t_wid, 1)],
                    'source_uid':  ref_id,
                })

            useful = [it for it in furn_items if it['local_cat'] not in ('Other', 'Lamp')]
            if len(useful) < MIN_USEFUL:
                continue

            candidates.append({
                'source_file': filename,
                'room_id':     room.get('instanceid', 'room'),
                'room_type':   room_type,
                'room': {
                    'length': round(room_len, 1),
                    'width':  round(room_wid, 1),
                    'height': round(room_h, 1),
                    'wall_thickness': 200,
                },
                'furniture': furn_items,
            })

    return candidates


# ── 主流程 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-files', type=int, default=MAX_FRONT_FILES,
                        help='最多扫描多少个 3D-FRONT 文件')
    parser.add_argument('--max-candidates', type=int, default=MAX_CANDIDATES,
                        help='打分后保留 top-N 候选（画预览图）')
    parser.add_argument('--select', type=str, default='',
                        help='直接选定编号，逗号分隔（如 0,2,5）')
    parser.add_argument('--top', type=int, default=0,
                        help='自动选 top-N 模板，不交互')
    parser.add_argument('--preview-only', action='store_true',
                        help='只生成预览图，不保存模板')
    args = parser.parse_args()

    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR,  exist_ok=True)

    # 1. 加载 model_info
    print("Loading model_info ...")
    with open(MODEL_INFO_PATH, encoding='utf-8') as f:
        entries = json.load(f)
    model_info = {e['model_id']: e for e in entries if e.get('model_id')}
    print(f"  {len(model_info)} entries")

    # 2. 扫描提取候选
    print(f"\nScanning 3D-FRONT (max {args.max_files} files) ...")
    candidates = extract_candidates(FRONT_JSON_DIR, model_info, args.max_files)
    print(f"  {len(candidates)} candidate rooms found")

    if not candidates:
        print("No candidates. Try --max-files with a larger number.")
        sys.exit(1)

    # 3. 打分 + 排序
    print("\nScoring ...")
    for cand in candidates:
        room = cand['room']
        cand['scores'] = score_room(
            cand['room_type'],
            cand['furniture'],
            room['length'] / 1000,
            room['width']  / 1000,
        )

    candidates.sort(key=lambda c: -c['scores']['total'])

    # 保留 top-N 候选画图
    top_cands = candidates[:args.max_candidates]

    # 类型分布统计
    type_cnt = Counter(c['room_type'] for c in top_cands)
    print(f"Top-{len(top_cands)} type distribution: {dict(type_cnt)}")
    print(f"Score range: {top_cands[-1]['scores']['total']:.3f} ~ "
          f"{top_cands[0]['scores']['total']:.3f}")

    # 4. 生成预览图
    print(f"\nGenerating {len(top_cands)} preview images -> {PREVIEW_DIR}")
    for idx, cand in enumerate(top_cands):
        out_path = os.path.join(PREVIEW_DIR, f"preview_{idx:03d}.png")
        try:
            draw_preview(cand, out_path, idx)
        except Exception as e:
            print(f"  [WARN] preview_{idx:03d} failed: {e}")

        sc = cand['scores']
        print(f"  #{idx:03d}  {cand['room_type']:20s}  "
              f"score={sc['total']:.3f}  "
              f"n={sc['n_furniture']:2d}  "
              f"cats={sc['categories']}")

    print(f"\nPreview images saved to: {PREVIEW_DIR}")

    if args.preview_only:
        return

    # 5. 选定模板
    if args.top > 0:
        selected_indices = list(range(min(args.top, len(top_cands))))
        print(f"\nAuto-selecting top {len(selected_indices)} templates.")
    elif args.select:
        selected_indices = [int(x.strip()) for x in args.select.split(',')
                            if x.strip().isdigit()]
        print(f"\nUsing --select: {selected_indices}")
    else:
        selected_indices = _interactive_select(top_cands, PREVIEW_DIR)
        if selected_indices is None:
            print("Aborted.")
            return

    # 6. 清空旧模板目录，保存选定模板
    import shutil
    for fn in os.listdir(TEMPLATE_DIR):
        if fn.endswith('.json'):
            os.remove(os.path.join(TEMPLATE_DIR, fn))

    saved = 0
    for idx in selected_indices:
        if idx >= len(top_cands):
            print(f"  [WARN] index {idx} out of range, skipped")
            continue
        cand = top_cands[idx]
        src_img = os.path.join(PREVIEW_DIR, f"preview_{idx:03d}.png")
        name = f"template_{saved:03d}_{cand['room_type']}_score{cand['scores']['total']:.2f}"
        out_json = os.path.join(TEMPLATE_DIR, name + ".json")
        out_img  = os.path.join(TEMPLATE_DIR, name + ".png")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(cand, f, indent=2, ensure_ascii=False)
        if os.path.exists(src_img):
            shutil.copy2(src_img, out_img)
        saved += 1
        sc = cand['scores']
        print(f"  Saved #{idx:03d} -> {name}  "
              f"(score={sc['total']:.3f}, n={sc['n_furniture']})")

    print(f"\nDone: {saved} templates saved to {TEMPLATE_DIR}")
    print("Next: run layout_variants.py to generate scene variants.")


if __name__ == '__main__':
    main()
