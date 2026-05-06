"""
layout_variants.py — 固定布局 × 替换家具，批量生成变体场景

流程：
  1. 从 3D-FRONT 布局文件中筛选高质量模板（有 Bed 或 Sofa，有效家具 >= MIN_USEFUL）
  2. 将每个模板固定布局，按类别从资产库中随机采样不同模型替换
  3. 每个模板生成 VARIANTS_PER_TEMPLATE 个变体，直接输出 layout JSON
  4. 最终的 layout JSON 送入 Scene_sys_v2.py 组装 STEP 文件

用法:
  python layout_variants.py               # 使用默认配置
  python layout_variants.py --templates 30 --variants 5
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

# ================== 配置区 ==================
FRONT_JSON_DIR   = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\front_scene_layout"
INDEX_JSON_PATH  = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\cad_assets_index.json"
MODEL_INFO_PATH  = "D:\\Lzm_Temp_Data\\3D-FUTURE-model\\model_info.json"
TEMPLATE_DIR     = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\layout_templates"
OUTPUT_DIR       = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\layout_variants_out"

MAX_FRONT_FILES      = 1000     # 最多扫描多少个 3D-FRONT 文件找模板
MIN_USEFUL_FURNITURE = 4       # 模板最少有效家具数（不含 Other）
MAX_TEMPLATES        = 100      # 最多保留多少个模板
VARIANTS_PER_TEMPLATE = 10      # 每个模板生成多少个变体
RANDOM_SEED          = 42

# 只选这些房间类型作为模板（None = 全部）
TARGET_ROOM_TYPES = {}

# 尺寸兼容容差：候选模型最大尺寸在目标尺寸的 [1/SCALE_TOL, SCALE_TOL] 倍内
SCALE_TOL = 2.5
# ============================================


def _load_model_info(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        entries = json.load(f)
    return {e['model_id']: e for e in entries if e.get('model_id')}


def _load_assets(path: str) -> dict[str, list]:
    """加载资产库，按 category 分组。"""
    with open(path, encoding='utf-8') as f:
        assets = json.load(f)
    by_cat: dict[str, list] = {}
    for a in assets:
        cat = a.get('category', 'Other')
        by_cat.setdefault(cat, []).append(a)
    return by_cat


# ── 复用 front_to_cad 的分类逻辑 ────────────────────────────────────────

SUPER_CAT_MAP = {
    "Sofa":               "Sofa",
    "Chair":              "Chair",
    "Table":              "Table",
    "Bed":                "Bed",
    "Cabinet/Shelf/Desk": "Cabinet",
    "Pier/Stool":         "Chair",
    "Lighting":           "Lamp",
    "Others":             None,
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
    "Chaise Longue Sofa": "Sofa", "Two-seat Sofa": "Sofa",
    "Couch Bed": "Sofa",
    "King-size Bed": "Bed", "Double Bed": "Bed", "Single bed": "Bed",
    "Kids Bed": "Bed", "Bed Frame": "Bed",
    "Wardrobe": "Cabinet", "Nightstand": "Cabinet",
    "Drawer Chest / Corner cabinet": "Cabinet",
    "Bookcase / jewelry Armoire": "Cabinet", "TV Stand": "Cabinet",
    "Sideboard / Side Cabinet / Console Table": "Cabinet",
    "Sideboard / Side Cabinet / Console": "Cabinet",
    "Shoe Cabinet": "Cabinet", "Wine Cabinet": "Cabinet",
    "Children Cabinet": "Cabinet", "Shelf": "Cabinet",
    "Floor Lamp": "Lamp",
}

SKIP_FINE_CATS = frozenset([
    "Pendant Lamp", "Ceiling Lamp", "Wall Lamp",
    "Wine Cooler", "Bar", "Bunk Bed", "Folding chair",
    "Hanging Chair",
])

CEILING_FINE_CATS = frozenset(["Pendant Lamp", "Ceiling Lamp"])

FLOOR_ATTACH_MAX_Z   = 800.0
CEILING_ATTACH_RATIO = 0.70


def _get_local_cat(jid: str, model_info: dict) -> tuple[str, str]:
    """返回 (local_cat, fine_cat)。"""
    entry = model_info.get(jid)
    if not entry:
        return "Other", ""
    fine = entry.get("category") or ""
    if fine in FINE_CAT_OVERRIDE:
        return FINE_CAT_OVERRIDE[fine], fine
    super_cat = entry.get("super-category", "")
    local = SUPER_CAT_MAP.get(super_cat) or "Other"
    return local, fine


def _should_skip(local_cat: str, fine_cat: str, cad_z: float,
                 room_h: float, target_h: float) -> bool:
    if not local_cat or local_cat == "Other":
        return True
    if fine_cat in SKIP_FINE_CATS:
        return True
    if fine_cat in CEILING_FINE_CATS:
        return cad_z < room_h * CEILING_ATTACH_RATIO
    return (cad_z - target_h / 2.0) > FLOOR_ATTACH_MAX_Z


# ── 模板提取 ─────────────────────────────────────────────────────────────

def extract_templates(front_json_dir: str, model_info: dict,
                      max_files: int, max_templates: int) -> list[dict]:
    """
    扫描 3D-FRONT JSON 文件，提取高质量房间布局作为模板。
    每个模板是一个 dict：
      room_type, room_dims, furniture[{local_cat, fine_cat, position, rotation_z,
                                        target_size, source_uid}]
    """
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

    templates = []
    for filename in all_files[:max_files]:
        if len(templates) >= max_templates:
            break
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
            if len(templates) >= max_templates:
                break

            room_type = room.get('type', 'Unknown')
            if TARGET_ROOM_TYPES and room_type not in TARGET_ROOM_TYPES:
                continue

            # 计算房间包围盒
            positions = [
                child['pos'] for child in room.get('children', [])
                if isinstance(child.get('pos'), list) and len(child['pos']) == 3
            ]
            if positions:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                zs = [p[2] for p in positions]
                bounds = {
                    'min_x': min(xs), 'max_x': max(xs),
                    'min_y': min(ys), 'max_y': max(ys),
                    'min_z': min(zs), 'max_z': max(zs),
                }
            else:
                sz = room.get('size', [6.0, 2.8, 5.0])
                pos = room.get('pos', [0.0, 0.0, 0.0])
                bounds = {
                    'min_x': pos[0]-sz[0]/2, 'max_x': pos[0]+sz[0]/2,
                    'min_y': pos[1],          'max_y': pos[1]+sz[1],
                    'min_z': pos[2]-sz[2]/2,  'max_z': pos[2]+sz[2]/2,
                }

            center_x = (bounds['min_x'] + bounds['max_x']) / 2
            center_z = (bounds['min_z'] + bounds['max_z']) / 2
            floor_y  = bounds['min_y']
            room_len = max((bounds['max_x'] - bounds['min_x']) * 1000 + 1000, 4000.0)
            room_wid = max((bounds['max_z'] - bounds['min_z']) * 1000 + 1000, 3000.0)
            room_h   = max((bounds['max_y'] - bounds['min_y']) * 1000 + 300,  2600.0)

            furniture_items = []
            for child in room.get('children', []):
                ref_id = child.get('ref')
                if not ref_id or ref_id not in furniture_dict:
                    continue
                f_info = furniture_dict[ref_id]
                jid    = f_info.get('jid', '')
                local_cat, fine_cat = _get_local_cat(jid, model_info)

                scale   = child.get('scale', [1, 1, 1])
                if not isinstance(scale, list) or len(scale) != 3:
                    scale = [1, 1, 1]
                raw_sz  = _parse_size(f_info, scale)
                t_len   = raw_sz[2] * 1000   # Unity Z → CAD X
                t_h     = raw_sz[1] * 1000   # Unity Y → CAD Z
                t_wid   = raw_sz[0] * 1000   # Unity X → CAD Y

                pos   = child.get('pos', [0, 0, 0])
                cad_x = (pos[0] - center_x) * 1000
                cad_y = (pos[2] - center_z) * 1000
                cad_z = (pos[1] - floor_y)  * 1000

                if _should_skip(local_cat, fine_cat, cad_z, room_h, t_h):
                    continue

                rot = child.get('rot', [0, 0, 0, 1])
                if not isinstance(rot, list) or len(rot) != 4:
                    rot = [0, 0, 0, 1]

                furniture_items.append({
                    'local_cat':   local_cat,
                    'fine_cat':    fine_cat,
                    'position':    [round(cad_x, 1), round(cad_y, 1), round(cad_z, 1)],
                    'rotation_z':  round(_yaw(rot), 1),
                    'target_size': [round(t_len, 1), round(t_h, 1), round(t_wid, 1)],
                    'source_uid':  ref_id,
                })

            useful = [it for it in furniture_items if it['local_cat'] != 'Other']
            if len(useful) < MIN_USEFUL_FURNITURE:
                continue

            templates.append({
                'source_file': filename,
                'room_id':     room.get('instanceid', 'room'),
                'room_type':   room_type,
                'room': {
                    'length': round(room_len, 1),
                    'width':  round(room_wid, 1),
                    'height': round(room_h, 1),
                    'wall_thickness': 200,
                },
                'furniture': furniture_items,
            })
            print(f"  [OK] {room_type} [{room.get('instanceid','')}]  "
                  f"useful={len(useful)}/{len(furniture_items)}")

    return templates


# ── 变体生成 ─────────────────────────────────────────────────────────────

def _size_score(asset: dict, t_len: float, t_wid: float, t_h: float) -> float:
    """尺寸相似度 [0,1]，高斯衰减（log 尺度）。"""
    t_foot = max(t_len, t_wid, 1.0)
    m_foot = max(asset.get('length', 1), asset.get('width', 1), 1.0)
    m_h    = max(asset.get('height', 1), 1.0)
    foot_s = math.exp(-1.5 * abs(math.log(m_foot / t_foot)))
    h_s    = math.exp(-1.0 * abs(math.log(max(m_h / max(t_h, 1), 0.01))))
    return 0.6 * foot_s + 0.4 * h_s


def generate_variant(template: dict, assets_by_cat: dict,
                     rng: random.Random, variant_idx: int) -> dict:
    """
    从模板生成一个变体：对每件家具，从资产库同类别中按尺寸加权随机采样。
    """
    variant_furniture = []
    used_folders: set[str] = set()

    for item in template['furniture']:
        cat = item['local_cat']
        if cat == 'Other':
            continue

        candidates = assets_by_cat.get(cat, [])
        if not candidates:
            continue

        t_len, t_h, t_wid = item['target_size']
        t_foot = max(t_len, t_wid, 1.0)

        # 过滤尺寸差异过大的
        filtered = [
            a for a in candidates
            if max(a.get('length', 1), a.get('width', 1), 1.0) / t_foot <= SCALE_TOL
            and t_foot / max(a.get('length', 1), a.get('width', 1), 1.0) <= SCALE_TOL
        ]
        if not filtered:
            filtered = candidates  # 无候选时放开限制

        # 尺寸加权随机采样（变体之间用不同随机偏移保证多样性）
        scores = [_size_score(a, t_len, t_wid, t_h) for a in filtered]
        total  = sum(scores) or 1.0
        weights = [s / total for s in scores]

        # 尝试选一个在本变体中未使用过的模型
        chosen = None
        for _ in range(10):
            chosen = rng.choices(filtered, weights=weights, k=1)[0]
            if chosen['folder_path'] not in used_folders:
                break
        used_folders.add(chosen['folder_path'])

        variant_furniture.append({
            'id':          f"{cat}_{len(variant_furniture)}",
            'folder':      chosen['folder_path'],
            'position':    item['position'],
            'rotation_z':  item['rotation_z'],
            'target_size': item['target_size'],
            'source_category':  cat,
            'matched_category': chosen.get('category', cat),
            '_fine_cat':        item['fine_cat'],
        })

    return {
        'room_type':  template['room_type'],
        'room':       template['room'],
        'furniture':  variant_furniture,
        '_source':    f"{template['source_file']} | {template['room_id']} | v{variant_idx}",
    }


# ── 主流程 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=int, default=VARIANTS_PER_TEMPLATE,
                        help='每个模板生成多少个变体')
    parser.add_argument('--seed',     type=int, default=RANDOM_SEED)
    parser.add_argument('--template-dir', type=str, default=TEMPLATE_DIR,
                        help='模板目录（由 select_templates.py 生成）')
    args = parser.parse_args()

    template_dir = args.template_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取已选模板
    template_files = sorted(
        f for f in os.listdir(template_dir) if f.endswith('.json')
    )
    if not template_files:
        print(f"No template JSON files found in {template_dir}")
        print("Run select_templates.py first to pick templates.")
        sys.exit(1)

    templates = []
    for fn in template_files:
        with open(os.path.join(template_dir, fn), encoding='utf-8') as f:
            templates.append(json.load(f))
    print(f"Loaded {len(templates)} templates from {template_dir}")

    print("Loading assets ...")
    assets_by_cat = _load_assets(INDEX_JSON_PATH)
    for cat, lst in sorted(assets_by_cat.items()):
        print(f"  {cat}: {len(lst)} models")

    type_counts = Counter(t['room_type'] for t in templates)
    print("Template type distribution:", dict(type_counts))

    # 生成变体
    rng = random.Random(args.seed)
    total = 0
    for i, tmpl in enumerate(templates):
        for v in range(args.variants):
            variant = generate_variant(tmpl, assets_by_cat, rng, v)
            if len(variant['furniture']) < 2:
                continue
            out_name = f"variant_{i:03d}_{tmpl['room_type']}_v{v}.json"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(variant, f, indent=2, ensure_ascii=False)
            total += 1

    print(f"\nGenerated {total} variants -> {OUTPUT_DIR}")

    cat_counter: Counter = Counter()
    for fn in os.listdir(OUTPUT_DIR):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(OUTPUT_DIR, fn), encoding='utf-8') as f:
            d = json.load(f)
        for item in d['furniture']:
            cat_counter[item['source_category']] += 1
    print("Category distribution (all variants):")
    for cat, cnt in cat_counter.most_common():
        print(f"  {cat}: {cnt}")


if __name__ == '__main__':
    main()
