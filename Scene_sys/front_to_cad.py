import json
import os
import math
import random
from collections import Counter
from scipy.spatial.transform import Rotation as R

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# ================== 配置区 ==================
FRONT_JSON_DIR = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\front_scene_layout"
INDEX_JSON_PATH = "cad_assets_index.json"
OUTPUT_LAYOUT_DIR = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\layout_outputs"
MODEL_INFO_PATH = "D:\\Lzm_Temp_Data\\3D-FUTURE-model\\model_info.json"
MAX_SCENES = 99999           # 实际上限由数据集大小决定，默认全量处理
MAX_FURNITURE_PER_SCENE = 50
MIN_USEFUL_FURNITURE = 3     # 输出房间最少有效家具数（低于此值跳过）
SHUFFLE_FILES = True          # 随机打乱文件顺序，获取更多样化的场景
RANDOM_SEED   = 42            # 固定随机种子保证可复现

# 房间类型过滤：None 表示不过滤（处理所有类型）
ROOM_TYPE_FILTER = None
# 每个房间文件独立输出（True=按房间切分；False=整栋房子合并，旧行为）
SPLIT_BY_ROOM = True

# model_info super-category → 本地资产类别
SUPER_CAT_MAP = {
    "Sofa":              "Sofa",
    "Chair":             "Chair",
    "Table":             "Table",
    "Bed":               "Bed",
    "Cabinet/Shelf/Desk": "Cabinet",
    "Pier/Stool":        "Chair",
    "Lighting":          "Lamp",
    "Others":            None,   # None = 跳过
}

# model_info category 细分 → 跳过（我们资产库中没有）
SKIP_FINE_CATEGORIES = frozenset([
    "Pendant Lamp", "Ceiling Lamp", "Wall Lamp",   # 灯具，位置特殊，单独处理
    "Wine Cooler", "Bar",                           # 酒柜/吧台，无对应资产
    "Bunk Bed",                                     # 上下铺，几何太复杂
    "Folding chair",                                # 折叠椅，无对应
    "Hanging Chair",                                # 吊椅，无对应
])

# 细分 → 覆盖本地类别（super-category 给的不够准确时修正）
FINE_CAT_OVERRIDE = {
    "Desk":                         "Table",
    "Dining Table":                 "Table",
    "Coffee Table":                 "Table",
    "Tea Table":                    "Table",
    "Corner/Side Table":            "Table",
    "Round End Table":              "Table",
    "Dressing Table":               "Table",
    "armchair":                     "Chair",
    "Dining Chair":                 "Chair",
    "Lounge Chair / Cafe Chair / Office Chair": "Chair",
    "Lounge Chair / Book-chair / Computer Chair": "Chair",
    "Dressing Chair":               "Chair",
    "Barstool":                     "Chair",
    "Footstool / Sofastool / Bed End Stool / Stool": "Chair",
    "Three-Seat / Multi-seat Sofa": "Sofa",
    "Three-Seat / Multi-person sofa": "Sofa",
    "Loveseat Sofa":                "Sofa",
    "L-shaped Sofa":                "Sofa",
    "U-shaped Sofa":                "Sofa",
    "Lazy Sofa":                    "Sofa",
    "Chaise Longue Sofa":           "Sofa",
    "Two-seat Sofa":                "Sofa",
    "Couch Bed":                    "Sofa",
    "King-size Bed":                "Bed",
    "Double Bed":                   "Bed",
    "Single bed":                   "Bed",
    "Kids Bed":                     "Bed",
    "Bed Frame":                    "Bed",
    "Wardrobe":                     "Cabinet",
    "Nightstand":                   "Cabinet",
    "Drawer Chest / Corner cabinet": "Cabinet",
    "Bookcase / jewelry Armoire":   "Cabinet",
    "TV Stand":                     "Cabinet",
    "Sideboard / Side Cabinet / Console Table": "Cabinet",
    "Sideboard / Side Cabinet / Console": "Cabinet",
    "Shoe Cabinet":                 "Cabinet",
    "Wine Cabinet":                 "Cabinet",
    "Children Cabinet":             "Cabinet",
    "Shelf":                        "Cabinet",
    "Floor Lamp":                   "Lamp",
}

# 天花板灯具的细分类别（位置判断用）
CEILING_FINE_CATS = frozenset(["Pendant Lamp", "Ceiling Lamp"])

# 主要家具类别（在布局文件中靠前排列，碰撞时优先保留）
PRIMARY_CATEGORIES = frozenset({"Bed", "Table", "Chair", "Sofa"})

# 地板附着物：底面 Z 不超过此值（mm）即认为贴地；放宽到 800mm 容纳搁板/电视柜等
FLOOR_ATTACH_MAX_Z = 800.0
# 天花板附着物（吊灯等）要求 cad_z ≥ 房间高度 × 此比例
CEILING_ATTACH_MIN_RATIO = 0.70

# 主要家具类别（在布局文件中靠前排列，碰撞时优先保留）
PRIMARY_CATEGORIES = frozenset({"Bed", "Table", "Chair", "Sofa"})

# 地板附着物：底面 Z 不超过此值（mm）即认为贴地；放宽到 800mm 容纳搁板/电视柜等
FLOOR_ATTACH_MAX_Z = 800.0
# 天花板附着物（吊灯等）要求 cad_z ≥ 房间高度 × 此比例
CEILING_ATTACH_MIN_RATIO = 0.70
# ============================================


# 双语风格/材质标签，用于跨语言风格一致性评分
_STYLE_TAGS: dict[str, str] = {
    # 英文
    "modern": "modern", "contemporary": "modern", "minimalist": "modern",
    "traditional": "traditional", "classic": "traditional",
    "industrial": "industrial",
    "scandinavian": "nordic", "nordic": "nordic",
    "rustic": "rustic", "farmhouse": "rustic",
    "luxury": "luxury", "premium": "luxury",
    "fabric": "fabric", "textile": "fabric", "upholstered": "fabric",
    "leather": "leather",
    "wood": "wood", "wooden": "wood", "walnut": "wood", "oak": "wood",
    "metal": "metal", "steel": "metal", "iron": "metal",
    # 中文
    "现代": "modern", "简约": "modern", "极简": "modern", "当代": "modern",
    "传统": "traditional", "古典": "traditional", "中式": "traditional",
    "工业": "industrial",
    "北欧": "nordic", "斯堪的纳维亚": "nordic",
    "乡村": "rustic", "田园": "rustic",
    "奢华": "luxury", "高端": "luxury",
    "布艺": "fabric", "织物": "fabric", "软包": "fabric",
    "皮革": "leather", "真皮": "leather",
    "木质": "wood", "实木": "wood", "胡桃木": "wood", "橡木": "wood",
    "金属": "metal", "钢": "metal", "铁": "metal",
}


def _extract_style_tags(text: str) -> set:
    """从文本中提取风格/材质标签集合，支持中英文。"""
    tags = set()
    tl = text.lower()
    for kw, tag in _STYLE_TAGS.items():
        if kw in tl or kw in text:
            tags.add(tag)
    return tags


def normalize_text(text):
    return " ".join(str(text or "").strip().lower().replace("\n", " ").split())


def merge_front_description(f_info, model_info_entry: dict | None = None):
    """合并 3D-FRONT furniture 字段 + model_info 风格/材质，作为 TF-IDF 查询文本。"""
    parts = []
    for v in (f_info.get("title", ""), f_info.get("category", ""),
              f_info.get("style", ""), f_info.get("material", ""),
              f_info.get("description", "")):
        if v:
            parts.append(str(v))
    if model_info_entry:
        for key in ("category", "style", "material", "theme"):
            v = model_info_entry.get(key)
            if v:
                parts.append(str(v))
    return " ".join(parts)


class FrontTranslator:
    def __init__(self, index_path, model_info_path: str = MODEL_INFO_PATH):
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                self.local_assets = json.load(f)
        else:
            print("⚠️ 找不到本地资产索引库，匹配功能将受限！")
            self.local_assets = []

        # 加载 3D-FUTURE model_info：jid → {super-category, category, style, material}
        self._model_info: dict[str, dict] = {}
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r', encoding='utf-8') as f:
                for entry in json.load(f):
                    mid = entry.get('model_id')
                    if mid:
                        self._model_info[mid] = entry
            print(f"✅ model_info 加载完成：{len(self._model_info)} 条记录")
        else:
            print(f"⚠️ 找不到 model_info.json: {model_info_path}")

        self._build_tfidf_index()

    # ── TF-IDF 索引（每类别一个向量化器）──────────────────────────────

    def _build_tfidf_index(self):
        """
        对资产库的中文描述建立逐类别 TF-IDF 索引。
        使用字符级 2-3 gram，无需中文分词器即可正确处理中文。
        """
        self._tfidf_by_cat: dict = {}
        if not _SKLEARN_OK or not self.local_assets:
            if not _SKLEARN_OK:
                print("⚠️ sklearn 未安装，降级为基础尺寸匹配（pip install scikit-learn 可启用 TF-IDF）")
            return

        by_cat: dict[str, list] = {}
        for m in self.local_assets:
            cat = m.get('category', 'Other')
            by_cat.setdefault(cat, []).append(m)

        for cat, models in by_cat.items():
            # 将描述 + 模型名合并为查询文档
            texts = [
                (m.get('description', '') + ' ' + m.get('model_name', '')).strip()
                for m in models
            ]
            if not any(texts):
                continue
            try:
                vec = TfidfVectorizer(
                    analyzer='char',        # 字符 n-gram，无需分词，适合中文
                    ngram_range=(2, 3),
                    min_df=1,
                    sublinear_tf=True,
                )
                mat = vec.fit_transform(texts)
                self._tfidf_by_cat[cat] = (vec, mat, models)
            except Exception as exc:
                print(f"⚠️ 类别 {cat} TF-IDF 构建失败: {exc}")

        n = sum(len(v[2]) for v in self._tfidf_by_cat.values())
        print(f"✅ TF-IDF 索引构建完成：{len(self._tfidf_by_cat)} 个类别，共 {n} 个模型")

    # ── 评分辅助 ──────────────────────────────────────────────────────

    def _size_score(self, model: dict, t_length: float, t_width: float, t_height: float) -> float:
        """
        尺寸相似度得分 [0, 1]。
        t_length/width/height 均为 mm，对应布局中 target_size[0/2/1]。
        评分维度：
          • 占地面积（footprint max dim）：权 0.50
          • 平面长宽比：权 0.30
          • 高度：权 0.20
        均使用高斯衰减（log 尺度），使 2× 与 0.5× 惩罚对称。
        """
        t_foot = max(t_length, t_width, 1.0)
        t_ratio = t_length / t_width if t_width > 1 else 1.0

        m_foot = max(model.get('length', 1), model.get('width', 1), 1.0)
        m_ratio = model.get('aspect_ratio', 1.0)
        m_h = max(model.get('height', 1), 1.0)

        foot_score  = math.exp(-1.5 * abs(math.log(m_foot / t_foot)))
        ratio_score = math.exp(-1.0 * abs(m_ratio - t_ratio))
        h_score     = math.exp(-1.0 * abs(math.log(max(m_h / max(t_height, 1), 0.01))))

        return 0.50 * foot_score + 0.30 * ratio_score + 0.20 * h_score

    def _tfidf_scores(self, query_text: str, category: str) -> dict:
        """返回该类别中每个模型的 TF-IDF 余弦相似度 {folder_path: score}。"""
        result: dict[str, float] = {}
        entry = self._tfidf_by_cat.get(category)
        if entry is None:
            return result
        vec, mat, models = entry
        try:
            q_vec = vec.transform([query_text])
            sims = _cos_sim(q_vec, mat)[0]
            for sim, m in zip(sims, models):
                result[m.get('folder_path', '')] = float(sim)
        except Exception:
            pass
        return result

    def get_local_category(self, jid: str) -> str:
        """用 jid 查 model_info，返回本地资产类别；找不到返回 'Other'。"""
        entry = self._model_info.get(jid)
        if not entry:
            return "Other"
        fine = entry.get("category", "")
        # 细分类别覆盖优先
        if fine in FINE_CAT_OVERRIDE:
            return FINE_CAT_OVERRIDE[fine]
        # 粗分类别兜底
        super_cat = entry.get("super-category", "")
        return SUPER_CAT_MAP.get(super_cat) or "Other"

    def get_model_info(self, jid: str) -> dict | None:
        return self._model_info.get(jid)

    def match_best_model(
        self,
        target_category: str,
        target_length: float,
        target_width: float,
        target_height: float,
        front_text: str,
        scene_style_tags: set | None = None,
    ):
        """
        综合评分匹配最优模型。
          • 尺寸相似度   0.55  —— 防碰撞的主要因素
          • TF-IDF 语义  0.30  —— 利用中文详细描述（字符 n-gram）
          • 风格一致性   0.10  —— 场景内风格连贯
          • 风格加成     0.05  —— 与当前场景主风格一致时的小额奖励
        """
        candidates = [m for m in self.local_assets if m.get('category') == target_category]
        if not candidates:
            candidates = [m for m in self.local_assets if m.get('category') == 'Other']
            if not candidates:
                return None

        # 预取 TF-IDF 相似度
        tfidf_map = self._tfidf_scores(front_text, target_category)

        scene_tags = scene_style_tags or set()
        scored = []
        for m in candidates:
            size_s  = self._size_score(m, target_length, target_width, target_height)
            text_s  = tfidf_map.get(m.get('folder_path', ''), 0.0)
            m_tags  = _extract_style_tags(m.get('description', ''))
            style_s = (len(m_tags & scene_tags) / len(m_tags | scene_tags)
                       if scene_tags and (m_tags | scene_tags) else 0.0)

            combined = 0.55 * size_s + 0.30 * text_s + 0.10 * style_s
            if scene_tags and (m_tags & scene_tags):
                combined += 0.05   # 风格一致加成

            scored.append((combined, m))

        scored.sort(key=lambda x: -x[0])
        best_score, best_model = scored[0]
        if best_score < 0.15:
            print(f"   ⚠️ 匹配得分偏低({best_score:.2f})，建议扩充该类别资产: {target_category}")
        return best_model

    # ── 过滤 ──────────────────────────────────────────────────────────

    def _should_skip_item(self, local_cat: str, fine_cat: str, cad_z: float,
                          room_height_mm: float, target_height_mm: float = 0.0) -> bool:
        """
        判断是否跳过该物品。
          • local_cat = "Other" 或 None → 跳过（无法匹配资产）
          • fine_cat 在 SKIP_FINE_CATEGORIES → 跳过（资产库无此类）
          • 天花板灯具：fine_cat 在 CEILING_FINE_CATS，要求 cad_z ≥ 房间高度 × CEILING_ATTACH_MIN_RATIO
          • 其余：底面 Z > FLOOR_ATTACH_MAX_Z → 视为悬浮物，跳过
        """
        if not local_cat or local_cat == "Other":
            return True

        if fine_cat in SKIP_FINE_CATEGORIES:
            return True

        if fine_cat in CEILING_FINE_CATS:
            return cad_z < room_height_mm * CEILING_ATTACH_MIN_RATIO

        bottom_z = cad_z - target_height_mm / 2.0
        return bottom_z > FLOOR_ATTACH_MAX_Z

    # ── 尺寸解析 ──────────────────────────────────────────────────────

    def parse_front_size(self, f_info, scale):
        """解析 3D-FRONT 家具尺寸；scale 可能含负值（镜像），取绝对值。"""
        abs_scale = [abs(s) for s in scale]
        size = f_info.get('size')
        if isinstance(size, list) and len(size) == 3:
            return [size[0] * abs_scale[0], size[1] * abs_scale[1], size[2] * abs_scale[2]]
        bbox = f_info.get('bbox')
        if isinstance(bbox, list) and len(bbox) == 3:
            return [bbox[0] * abs_scale[0], bbox[1] * abs_scale[1], bbox[2] * abs_scale[2]]
        return [abs_scale[0], abs_scale[1], abs_scale[2]]

    def front_to_cad_position(self, front_pos, floor_y, center_x, center_z):
        x, y, z = front_pos
        cad_x = (x - center_x) * 1000.0
        cad_y = (z - center_z) * 1000.0
        cad_z = (y - floor_y) * 1000.0
        return [cad_x, cad_y, cad_z]

    def extract_yaw_from_quat(self, rot_quat):
        try:
            rotation = R.from_quat(rot_quat)
            # 3D-FRONT/Unity canonical forward = +Z axis (not +X)
            forward = rotation.apply([0.0, 0.0, 1.0])
            # Coordinate mapping: FRONT_X→CAD_X, FRONT_Z→CAD_Y (Z-up conversion)
            cad_fwd_x = forward[0]
            cad_fwd_y = forward[2]
            yaw = math.degrees(math.atan2(cad_fwd_y, cad_fwd_x))
            return float((yaw + 360.0) % 360.0)
        except Exception:
            return 0.0

    def compute_scene_bounds(self, front_data):
        scene = front_data.get('scene', {})
        bbox = scene.get('boundingBox')
        if isinstance(bbox, dict):
            min_xyz = bbox.get('min', [])
            max_xyz = bbox.get('max', [])
            if len(min_xyz) == 3 and len(max_xyz) == 3:
                return {
                    'min_x': min_xyz[0], 'max_x': max_xyz[0],
                    'min_y': min_xyz[1], 'max_y': max_xyz[1],
                    'min_z': min_xyz[2], 'max_z': max_xyz[2],
                }

        positions = []
        for room in scene.get('room', []):
            for child in room.get('children', []):
                pos = child.get('pos')
                if isinstance(pos, list) and len(pos) == 3:
                    positions.append(pos)

        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]
            return {
                'min_x': min(xs), 'max_x': max(xs),
                'min_y': min(ys), 'max_y': max(ys),
                'min_z': min(zs), 'max_z': max(zs),
            }

        return {
            'min_x': -3.0, 'max_x': 3.0,
            'min_y': 0.0,  'max_y': 2.8,
            'min_z': -3.0, 'max_z': 3.0,
        }

    # ── 单房间布局计算 ────────────────────────────────────────────────

    def _compute_room_bounds(self, room: dict) -> dict:
        """从单个 room 的 children 位置计算包围盒（fallback 时使用 room.size）。"""
        positions = [
            child['pos'] for child in room.get('children', [])
            if isinstance(child.get('pos'), list) and len(child['pos']) == 3
        ]
        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]
            return {
                'min_x': min(xs), 'max_x': max(xs),
                'min_y': min(ys), 'max_y': max(ys),
                'min_z': min(zs), 'max_z': max(zs),
            }
        # 使用 room.size（3D-FRONT 单位为米）
        size = room.get('size', [6.0, 2.8, 5.0])
        pos  = room.get('pos',  [0.0, 0.0, 0.0])
        return {
            'min_x': pos[0] - size[0] / 2, 'max_x': pos[0] + size[0] / 2,
            'min_y': pos[1],               'max_y': pos[1] + size[1],
            'min_z': pos[2] - size[2] / 2, 'max_z': pos[2] + size[2] / 2,
        }

    def _process_room(
        self,
        room: dict,
        furniture_dict: dict,
        max_furniture_per_scene: int,
    ) -> dict | None:
        """将单个 3D-FRONT room 转换为 CAD 布局字典。返回 None 表示跳过（家具太少）。

        处理流程：
          1. 第一遍：收集全部候选家具，计算 cad_position、过滤悬浮物
          2. 按优先级排序：主要家具（Bed/Table/Chair/Sofa）靠前，碰撞时优先保留
          3. 第二遍：按序匹配模型，构建布局条目（rotation_z 统一置 0）
        """
        bounds   = self._compute_room_bounds(room)
        center_x = (bounds['min_x'] + bounds['max_x']) / 2.0
        center_z = (bounds['min_z'] + bounds['max_z']) / 2.0
        floor_y  = bounds['min_y']

        room_length = max((bounds['max_x'] - bounds['min_x']) * 1000.0 + 1000.0, 4000.0)
        room_width  = max((bounds['max_z'] - bounds['min_z']) * 1000.0 + 1000.0, 3000.0)
        room_height = max((bounds['max_y'] - bounds['min_y']) * 1000.0 + 300.0,  2600.0)

        cad_layout = {
            'room_type': room.get('type', 'Unknown'),
            'room': {
                'length': round(room_length, 1),
                'width':  round(room_width,  1),
                'height': round(room_height, 1),
                'wall_thickness': 200,
            },
            'furniture': [],
        }

        # ── 第一遍：收集 & 过滤 ───────────────────────────────────────
        candidates = []
        for child in room.get('children', []):
            ref_id = child.get('ref')
            if not ref_id or ref_id not in furniture_dict:
                continue

            f_info   = furniture_dict[ref_id]
            jid      = f_info.get('jid', '')
            mi_entry = self.get_model_info(jid)
            fine_cat = (mi_entry.get('category', '') if mi_entry else '')
            scale    = child.get('scale', [1.0, 1.0, 1.0])
            if not isinstance(scale, list) or len(scale) != 3:
                scale = [1.0, 1.0, 1.0]

            raw_size      = self.parse_front_size(f_info, scale)
            # 3D-FRONT/Unity 坐标：size[0]=X(左右宽), size[1]=Y(高), size[2]=Z(前后深)
            # CAD 坐标（Z-up，模型正面→+X）：X=前后深(t_length), Y=左右宽(t_width), Z=高
            target_length = raw_size[2] * 1000.0  # Unity Z(depth) → CAD X
            target_height = raw_size[1] * 1000.0  # Unity Y(height) → CAD Z
            target_width  = raw_size[0] * 1000.0  # Unity X(width)  → CAD Y

            local_cat    = self.get_local_category(jid)
            pos          = child.get('pos', [0.0, 0.0, 0.0])
            cad_position = self.front_to_cad_position(pos, floor_y, center_x, center_z)
            cad_z        = cad_position[2]

            if self._should_skip_item(local_cat, fine_cat, cad_z, room_height, target_height):
                print(f"   ⏭  跳过: jid={jid!r} fine={fine_cat!r} cat={local_cat}  z={cad_z:.0f}mm")
                continue

            candidates.append({
                'child':         child,
                'f_info':        f_info,
                'jid':           jid,
                'mi_entry':      mi_entry,
                'fine_cat':      fine_cat,
                'local_cat':     local_cat,
                'scale':         scale,
                'target_length': target_length,
                'target_height': target_height,
                'target_width':  target_width,
                'cad_position':  cad_position,
            })

        # ── 第二遍：按优先级排序（主要家具靠前）──────────────────────
        # 稳定排序，同优先级内保留原始顺序
        candidates.sort(key=lambda c: 0 if c['local_cat'] in PRIMARY_CATEGORIES else 1)

        # ── 第三遍：匹配模型，构建布局条目 ───────────────────────────
        scene_style_counter: Counter = Counter()

        for cand in candidates:
            if len(cad_layout['furniture']) >= max_furniture_per_scene:
                break

            child     = cand['child']
            f_info    = cand['f_info']
            local_cat = cand['local_cat']
            fine_cat  = cand['fine_cat']
            mi_entry  = cand['mi_entry']
            scale     = cand['scale']
            rot_quat  = child.get('rot', [0.0, 0.0, 0.0, 1.0])
            if not isinstance(rot_quat, list) or len(rot_quat) != 4:
                rot_quat = [0.0, 0.0, 0.0, 1.0]
            rotation_z = self.extract_yaw_from_quat(rot_quat)

            front_text     = merge_front_description(f_info, mi_entry)
            preferred_tags = {tag for tag, _ in scene_style_counter.most_common(3)}
            best_model = self.match_best_model(
                local_cat,
                cand['target_length'], cand['target_width'], cand['target_height'],
                front_text, scene_style_tags=preferred_tags,
            )

            if not best_model:
                print(f"   ⚠️ 未能匹配模型: jid={cand['jid']!r} fine={fine_cat!r} cat=[{local_cat}]")
                continue

            scene_style_counter.update(_extract_style_tags(best_model.get('description', '')))

            cad_layout['furniture'].append({
                'id':          f"{local_cat}_{child.get('instanceid', '').split('/')[-1]}",
                'source_uid':  child.get('ref'),
                'instanceid':  child.get('instanceid', ''),
                'folder':      best_model.get('folder_path', ''),
                'position':    [round(cand['cad_position'][0], 1),
                                round(cand['cad_position'][1], 1),
                                round(cand['cad_position'][2], 1)],
                'rotation_z':  round(rotation_z, 1),
                'scale':       [round(scale[0], 3), round(scale[1], 3), round(scale[2], 3)],
                'target_size': [round(cand['target_length'], 1),
                                round(cand['target_height'], 1),
                                round(cand['target_width'],  1)],
                'target_max_size': round(max(cand['target_length'],
                                             cand['target_height'],
                                             cand['target_width']), 1),
                'source_category':      local_cat,
                'matched_category':     best_model.get('category', 'Unknown'),
                'matched_aspect_ratio': best_model.get('aspect_ratio'),
                'matched_max_dim':      best_model.get('max_dim'),
                '_front_jid':      cand['jid'],
                '_front_fine_cat': fine_cat,
            })

        if len(cad_layout['furniture']) < MIN_USEFUL_FURNITURE:
            return None  # 有效家具太少，跳过
        return cad_layout

    # ── 主翻译流程 ────────────────────────────────────────────────────

    def translate(
        self,
        front_json_dir,
        output_dir,
        max_scenes=50,
        max_furniture_per_scene=50,
        room_type_filter=None,
        split_by_room=True,
    ):
        """
        room_type_filter: set[str] | None — 只处理指定类型的房间（None=全部）
        split_by_room:    True  — 每个房间单独输出一个 layout JSON（推荐）
                          False — 整栋房子合并输出（旧行为，room_type_filter 无效）
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_scenes = 0

        all_files = [f for f in os.listdir(front_json_dir) if f.endswith('.json')]
        if SHUFFLE_FILES:
            rng = random.Random(RANDOM_SEED)
            rng.shuffle(all_files)
        else:
            all_files.sort()

        for filename in all_files:
            if processed_scenes >= max_scenes:
                break

            front_json_path = os.path.join(front_json_dir, filename)
            try:
                with open(front_json_path, 'r', encoding='utf-8') as f:
                    front_data = json.load(f)
            except Exception as exc:
                print(f"⚠️ 跳过 {filename}：读取失败 ({exc})")
                continue

            furniture_dict = {
                item['uid']: item
                for item in front_data.get('furniture', [])
                if item.get('uid')
            }

            base_name = os.path.splitext(filename)[0]
            all_rooms = front_data.get('scene', {}).get('room', [])

            if split_by_room:
                for room in all_rooms:
                    if processed_scenes >= max_scenes:
                        break

                    room_type = room.get('type', 'Unknown')
                    room_id   = room.get('instanceid', 'room').replace('/', '_')

                    # ── 房间类型过滤 ─────────────────────────────────
                    if room_type_filter and room_type not in room_type_filter:
                        print(f"   ⏭  跳过房间类型: {room_type!r}  [{room_id}]")
                        continue

                    print(f"\n📂 {filename} | 房间: {room_type} [{room_id}]")
                    cad_layout = self._process_room(room, furniture_dict, max_furniture_per_scene)
                    if cad_layout is None:
                        print(f"   ⚠️ 有效家具不足，跳过此房间。")
                        continue

                    out_name = f"layout_{base_name}_{room_id}.json"
                    output_path = os.path.join(output_dir, out_name)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(cad_layout, f, indent=4, ensure_ascii=False)

                    print(f"   ✅ 保存: {out_name}  (家具数: {len(cad_layout['furniture'])})")
                    processed_scenes += 1

            else:
                # ── 旧行为：整栋房子合并处理 ──────────────────────────
                bounds   = self.compute_scene_bounds(front_data)
                center_x = (bounds['min_x'] + bounds['max_x']) / 2.0
                center_z = (bounds['min_z'] + bounds['max_z']) / 2.0
                floor_y  = bounds['min_y']
                room_length = max((bounds['max_x'] - bounds['min_x']) * 1000.0 + 2000.0, 6000.0)
                room_width  = max((bounds['max_z'] - bounds['min_z']) * 1000.0 + 2000.0, 5000.0)
                room_height = max((bounds['max_y'] - bounds['min_y']) * 1000.0 + 400.0,  2800.0)

                cad_layout = {
                    'room': {
                        'length': round(room_length, 1),
                        'width':  round(room_width,  1),
                        'height': round(room_height, 1),
                        'wall_thickness': 200,
                    },
                    'furniture': [],
                }
                scene_style_counter: Counter = Counter()
                scene_furniture_count = 0

                for room in all_rooms:
                    for child in room.get('children', []):
                        if scene_furniture_count >= max_furniture_per_scene:
                            break
                        ref_id = child.get('ref')
                        if not ref_id or ref_id not in furniture_dict:
                            continue
                        f_info   = furniture_dict[ref_id]
                        jid      = f_info.get('jid', '')
                        mi_entry = self.get_model_info(jid)
                        fine_cat = (mi_entry.get('category', '') if mi_entry else '')
                        scale    = child.get('scale', [1.0, 1.0, 1.0])
                        if not isinstance(scale, list) or len(scale) != 3:
                            scale = [1.0, 1.0, 1.0]
                        raw_size      = self.parse_front_size(f_info, scale)
                        target_length = raw_size[2] * 1000.0
                        target_height = raw_size[1] * 1000.0
                        target_width  = raw_size[0] * 1000.0
                        local_cat     = self.get_local_category(jid)
                        pos           = child.get('pos', [0.0, 0.0, 0.0])
                        cad_position  = self.front_to_cad_position(pos, floor_y, center_x, center_z)
                        cad_z         = cad_position[2]
                        if self._should_skip_item(local_cat, fine_cat, cad_z, room_height, target_height):
                            continue
                        rot_quat = child.get('rot', [0.0, 0.0, 0.0, 1.0])
                        if not isinstance(rot_quat, list) or len(rot_quat) != 4:
                            rot_quat = [0.0, 0.0, 0.0, 1.0]
                        cad_rot_z  = self.extract_yaw_from_quat(rot_quat)
                        front_text = merge_front_description(f_info, mi_entry)
                        preferred_tags = {tag for tag, _ in scene_style_counter.most_common(3)}
                        best_model = self.match_best_model(
                            local_cat, target_length, target_width, target_height,
                            front_text, scene_style_tags=preferred_tags,
                        )
                        if not best_model:
                            continue
                        scene_style_counter.update(_extract_style_tags(best_model.get('description', '')))
                        cad_layout['furniture'].append({
                            'id':          f"{local_cat}_{child.get('instanceid', '').split('/')[-1]}",
                            'source_uid':  ref_id,
                            'instanceid':  child.get('instanceid', ''),
                            'folder':      best_model.get('folder_path', ''),
                            'position':    [round(cad_position[0], 1), round(cad_position[1], 1), round(cad_position[2], 1)],
                            'rotation_z':  round(cad_rot_z, 1),
                            'scale':       [round(scale[0], 3), round(scale[1], 3), round(scale[2], 3)],
                            'target_size': [round(target_length, 1), round(target_height, 1), round(target_width, 1)],
                            'target_max_size': round(max(target_length, target_height, target_width), 1),
                            'source_category':      local_cat,
                            'matched_category':     best_model.get('category', 'Unknown'),
                            'matched_aspect_ratio': best_model.get('aspect_ratio'),
                            'matched_max_dim':      best_model.get('max_dim'),
                            '_front_jid':      jid,
                            '_front_fine_cat': fine_cat,
                        })
                        scene_furniture_count += 1
                    if scene_furniture_count >= max_furniture_per_scene:
                        break

                output_path = os.path.join(output_dir, f"layout_{base_name}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cad_layout, f, indent=4, ensure_ascii=False)
                print(f"\n✅ {filename} → {output_path}  (家具数: {len(cad_layout['furniture'])})")
                processed_scenes += 1

        print(f"\n🎉 完成: 共处理 {processed_scenes} 个场景，输出目录: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="3D-FRONT → layout JSON 转换")
    parser.add_argument('--input',   default=FRONT_JSON_DIR,    help='3D-FRONT JSON 目录')
    parser.add_argument('--output',  default=OUTPUT_LAYOUT_DIR, help='输出目录')
    parser.add_argument('--max',     type=int, default=MAX_SCENES, help='最多输出房间数（默认全量）')
    parser.add_argument('--min-furniture', type=int, default=MIN_USEFUL_FURNITURE,
                        help='输出房间最少有效家具数')
    args = parser.parse_args()

    translator = FrontTranslator(INDEX_JSON_PATH)
    translator.translate(
        args.input,
        args.output,
        max_scenes=args.max,
        max_furniture_per_scene=MAX_FURNITURE_PER_SCENE,
        room_type_filter=ROOM_TYPE_FILTER,
        split_by_room=SPLIT_BY_ROOM,
    )
