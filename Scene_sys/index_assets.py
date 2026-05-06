import os
import json
import math
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Pnt, gp_Trsf, gp_Ax1, gp_Dir

# ==========================================
# 路径配置区
# ==========================================
DATASET_PATHS = [
    ("Sofa",    r"D:/Lzm_Temp_Data/Li_temp_project/Dataset_Categoried/1_1000/Sofa"),
    ("Table",   r"D:/Lzm_Temp_Data/Li_temp_project/Dataset_Categoried/1_1000/Desk"),
    ("Chair",   r"D:/Lzm_Temp_Data/Li_temp_project/Dataset_Categoried/1_1000/Chair"),
    ("Cabinet", r"D:/Lzm_Temp_Data/Li_temp_project/Dataset_Categoried/1_1000/Cabinet(storage_furniture)"),
    ("Bed",     r"D:/Lzm_Temp_Data/Li_temp_project/Dataset_Categoried/1_1000/Bed"),
    ("Other",   r"D:/Lzm_Temp_Data/Li_temp_project/Dataset_Categoried/1_1000/other_furniture"),
    ("Other",   r"D:/Lzm_Temp_Data/Li_temp_project/Dataset_Categoried/1_1000/Desktop_object"),
]

INDEX_FILE = "cad_assets_index.json"

# 每一类最多扫描数量（0 = 不限制）
TEST_LIMIT = 1200
# ==========================================


def read_description(root):
    for fname in os.listdir(root):
        if fname.lower().endswith('.txt') and 'description' in fname.lower():
            try:
                with open(os.path.join(root, fname), 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read().strip()
            except Exception:
                pass
    return ""


def collect_sample_images(root, max_images=8):
    images = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(dirpath, f).replace('\\', '/'))
                if len(images) >= max_images:
                    return images
    return images


def get_normalized_dims(step_path, align_rot=None, extra_rz=0.0):
    """读取 STEP，应用完整对齐旋转，精确计算包围盒尺寸。"""
    if align_rot is None:
        align_rot = [0, 0, 0]
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        return None

    # 1. 应用站立旋转
    if any(align_rot):
        rx, ry, rz = [math.radians(a) for a in align_rot]
        tx, ty, tz = gp_Trsf(), gp_Trsf(), gp_Trsf()
        tx.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), rx)
        ty.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), ry)
        tz.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), rz)
        builder1 = BRepBuilderAPI_Transform(shape, tz.Multiplied(ty).Multiplied(tx), True)
        if not builder1.IsDone():
            return None
        shape = builder1.Shape()
        if shape is None or shape.IsNull():
            return None

    # 2. 应用正面修正旋转
    total_rz = extra_rz + 90.0
    if abs(total_rz % 360.0) > 0.01:
        tz2 = gp_Trsf()
        tz2.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(total_rz))
        builder2 = BRepBuilderAPI_Transform(shape, tz2, True)
        if not builder2.IsDone():
            return None
        shape = builder2.Shape()
        if shape is None or shape.IsNull():
            return None

    # 3. 计算包围盒
    BRepMesh_IncrementalMesh(shape, 1.0, False)
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    if bbox.IsVoid():
        return None

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    l = xmax - xmin
    w = ymax - ymin
    h = zmax - zmin

    return {
        "length": round(l, 2),
        "width":  round(w, 2),
        "height": round(h, 2),
        "aspect_ratio": round(l / w, 3) if w > 0.001 else 1.0,
    }


def load_existing_index(index_file):
    """加载已有索引，返回 (entries_list, already_indexed_paths_set)。"""
    if not os.path.exists(index_file):
        return [], set()
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        already = {entry['folder_path'] for entry in data if 'folder_path' in entry}
        print(f"[断点续扫] 已有索引 {len(data)} 条，跳过这些目录继续扫描。")
        return data, already
    except Exception as e:
        print(f"[警告] 读取已有索引失败，将从头扫描: {e}")
        return [], set()


def save_index(index_file, index_data):
    """写入索引：先写临时文件，再替换目标（Windows 下 os.replace 可能被文件锁阻止，
    fallback 到直接覆写）。"""
    tmp = index_file + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4, ensure_ascii=False)
    try:
        os.replace(tmp, index_file)
    except PermissionError:
        # Windows: 目标文件被占用时 os.replace 失败，直接覆写
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=4, ensure_ascii=False)
        try:
            os.remove(tmp)
        except OSError:
            pass


def scan_dataset():
    index_data, already_indexed = load_existing_index(INDEX_FILE)
    total_before = len(index_data)

    limit_str = str(TEST_LIMIT) if TEST_LIMIT > 0 else "无限制"
    print(f"[开始] 定向分类扫描，每类上限 {limit_str} 个\n")

    for category, base_path in DATASET_PATHS:
        print(f"[扫描] 类别 [{category}] -> {base_path}")

        if not os.path.exists(base_path):
            print(f"  [跳过] 目录不存在: {base_path}")
            continue

        count_for_category = 0

        for root, dirs, files in os.walk(base_path):
            if TEST_LIMIT > 0 and count_for_category >= TEST_LIMIT:
                print(f"  [停止] [{category}] 已达上限 {TEST_LIMIT}")
                break

            step_files = [f for f in files if f.lower().endswith(('.step', '.stp'))]
            if not step_files:
                continue

            folder_path_normalized = root.replace("\\", "/")

            # 断点续扫：已索引的直接跳过
            if folder_path_normalized in already_indexed:
                count_for_category += 1
                continue

            folder_name = os.path.basename(root)
            step_path = os.path.join(root, step_files[0])
            meta_path = os.path.join(root, "alignment_meta.json")

            align_rot = [0, 0, 0]
            extra_rz = 0.0
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if meta.get('status') == 'verified':
                        align_rot = meta.get("align_rotation", [0, 0, 0])
                        extra_rz = float(meta.get("extra_rz", 0.0))
                except Exception:
                    pass

            dims = get_normalized_dims(step_path, align_rot, extra_rz)
            if dims:
                entry = {
                    "folder_path":   folder_path_normalized,
                    "category":      category,
                    "model_name":    step_files[0],
                    "description":   read_description(root),
                    "sample_images": collect_sample_images(root, max_images=8),
                    "max_dim":       max(dims['length'], dims['width'], dims['height']),
                    **dims,
                }
                index_data.append(entry)
                already_indexed.add(folder_path_normalized)
                count_for_category += 1

                # 每成功一条立即写盘（断点保护）
                save_index(INDEX_FILE, index_data)
                print(f"  [OK {count_for_category}] {folder_name} "
                      f"(L={dims['length']:.0f} W={dims['width']:.0f} H={dims['height']:.0f} "
                      f"ratio={dims['aspect_ratio']})")
            else:
                print(f"  [跳过] {folder_name} — 无效几何体")

        print(f"  [{category}] 本次新增 {count_for_category} 个\n")

    new_count = len(index_data) - total_before
    print(f"[完成] 总索引 {len(index_data)} 条（本次新增 {new_count} 条），保存至: {INDEX_FILE}")


if __name__ == "__main__":
    scan_dataset()
