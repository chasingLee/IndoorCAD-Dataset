import concurrent.futures
import os
import json
import math
from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Dir, gp_GTrsf
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

# ================== 配置区 ==================
LAYOUT_DIR = "layout_outputs"    # front_to_cad.py 的输出目录
OUTPUT_DIR = "scene_outputs"
MAX_SCENES = 99999           # 默认全量处理，用命令行 --max 控制
MAX_FURNITURE_PER_SCENE = 50

# 碰撞检测（2D 平面投影）
COLLISION_GAP    = 20.0   # mm，家具最小间距
# 移动解决碰撞
PUSH_STEP        = 100.0  # mm，每次推移步长
MAX_PUSH_ATTEMPTS = 20    # 每个方向最多尝试次数

ROOM_MARGIN = 50          # mm，房间边界内边距，避免家具贴墙

# 超时保护（Windows 线程超时，单位秒）
FURNITURE_TIMEOUT = 60    # 单件家具处理超时（STEP读取+标准化+变换）
SCENE_TIMEOUT     = 400   # 整个场景超时（保护极端大场景）
# ============================================

SUPPORTED_STEP_EXTS = ('.step', '.stp')


def find_step_file(folder_path):
    if not os.path.isdir(folder_path):
        return None
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(SUPPORTED_STEP_EXTS):
            return os.path.join(folder_path, filename)
    return None


def load_step_shape(step_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP 读取失败: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP 形状为空: {step_path}")
    return shape


def compute_bounding_box(shape):
    # BRepBndLib 直接从 BRep 解析几何体计算包围盒，不需要预先三角化。
    # 避免 BRepMesh_IncrementalMesh 的开销（对复杂 STEP 可能花费数十秒）。
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    if bbox.IsVoid():
        return None
    return bbox.Get()


def make_room_geometry(room):
    """生成六面封闭房间（地板、天花板、四面墙）。
    使用薄实体板（BRepPrimAPI_MakeBox）：
      - CAD 软件中显示为完整实体，双侧可见 ✓
      - 各内表面法线天然朝向室内，渲染光照正确 ✓
    各板定位原则：内表面恰好位于房间边界，厚板向外侧延伸。
    """
    L = room.get('length', 6000)
    W = room.get('width', 5000)
    H = room.get('height', 2800)
    T = room.get('wall_thickness', 200)
    hl, hw = L / 2.0, W / 2.0

    return [
        # 地板：内表面 Z=0，向下延伸 T
        BRepPrimAPI_MakeBox(gp_Pnt(-hl, -hw, -T),  gp_Pnt(hl,    hw,    0  )).Shape(),
        # 天花板：内表面 Z=H，向上延伸 T
        BRepPrimAPI_MakeBox(gp_Pnt(-hl, -hw,  H),  gp_Pnt(hl,    hw,    H+T)).Shape(),
        # 前墙（+Y侧）：内表面 Y=+hw，向外延伸 T
        BRepPrimAPI_MakeBox(gp_Pnt(-hl,  hw,  0),  gp_Pnt(hl,    hw+T,  H  )).Shape(),
        # 后墙（-Y侧）：内表面 Y=-hw，向外延伸 T
        BRepPrimAPI_MakeBox(gp_Pnt(-hl, -hw-T, 0), gp_Pnt(hl,   -hw,   H  )).Shape(),
        # 右墙（+X侧）：内表面 X=+hl，向外延伸 T
        BRepPrimAPI_MakeBox(gp_Pnt( hl, -hw,   0), gp_Pnt(hl+T,  hw,   H  )).Shape(),
        # 左墙（-X侧）：内表面 X=-hl，向外延伸 T
        BRepPrimAPI_MakeBox(gp_Pnt(-hl-T, -hw, 0), gp_Pnt(-hl,   hw,   H  )).Shape(),
    ]


def normalize_shape(shape, target_size=None, align_rot=(0, 0, 0), forward_angle=0.0, flip_x=False):
    """
    对模型做标准化：
    1. 应用 align_rot（纠正 Z 轴朝上），可选 flip_x（修正上下颠倒的模型）
    2. 平移使底部落在 Z=0，XY 居中
    3. 应用 -forward_angle 旋转，使模型正面朝向 +X 轴（OCC 坐标系）
    4. 可选按 target_size 缩放

    参数:
        forward_angle: 模型在 align_rot 之后，正面在 OCC XY 平面上的角度（从+X轴CCW，度）
        flip_x: 若为 True，在 align_rot 之后绕 X 轴旋转 180°，纠正上下颠倒的模型
    """
    # 步骤1: 朝向校正
    if any(align_rot):
        rx, ry, rz = [math.radians(a) for a in align_rot]
        rot_x = gp_Trsf()
        rot_x.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), rx)
        rot_y = gp_Trsf()
        rot_y.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), ry)
        rot_z = gp_Trsf()
        rot_z.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), rz)
        shape = BRepBuilderAPI_Transform(shape, rot_z.Multiplied(rot_y).Multiplied(rot_x), True).Shape()

    # 步骤1b: 翻转修正（修正上下颠倒的模型）
    if flip_x:
        flip_trsf = gp_Trsf()
        flip_trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.pi)
        shape = BRepBuilderAPI_Transform(shape, flip_trsf, True).Shape()

    # 步骤2: 计算包围盒，居中并落底
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    if bbox.IsVoid():
        return shape, 0.0

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    center = gp_Trsf()
    center.SetTranslation(gp_Vec(-(xmin + xmax) / 2.0, -(ymin + ymax) / 2.0, -zmin))
    shape = BRepBuilderAPI_Transform(shape, center, True).Shape()

    # 步骤3: 正面朝向对齐到 +X 轴
    # 将 forward_angle（模型当前正面角度）旋转回 0°（即 +X 方向）
    if abs(forward_angle) > 0.01:
        fwd_trsf = gp_Trsf()
        fwd_trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                             math.radians(-forward_angle))
        shape = BRepBuilderAPI_Transform(shape, fwd_trsf, True).Shape()

    # 步骤4: 强制缩放到 target_size
    # 步骤3 已将模型正面对齐到 +X：X=长度, Y=宽度, Z=高度
    # 重新计算包围盒（步骤3可能改变了尺寸）
    bbox_cur = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox_cur)
    if not bbox_cur.IsVoid():
        xc0, yc0, zc0, xc1, yc1, zc1 = bbox_cur.Get()
        dx, dy, dz = xc1 - xc0, yc1 - yc0, zc1 - zc0
    max_dim = max(dx, dy, dz)

    if target_size is not None and isinstance(target_size, (list, tuple)) and len(target_size) == 3:
        t_length, t_height, t_width = target_size
        s_x = t_length / max(dx, 1e-6)
        s_y = t_width  / max(dy, 1e-6)
        s_z = t_height / max(dz, 1e-6)

        # 非均匀缩放用 gp_GTrsf（三轴独立）
        gtrsf = gp_GTrsf()
        gtrsf.SetValue(1, 1, s_x)
        gtrsf.SetValue(2, 2, s_y)
        gtrsf.SetValue(3, 3, s_z)
        shape = BRepBuilderAPI_GTransform(shape, gtrsf, True).Shape()

        # 缩放后重新落底（GTrsf 不保证 Z=0 仍在底部）
        bbox3 = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox3)
        if not bbox3.IsVoid():
            _, _, z3min, _, _, _ = bbox3.Get()
            if abs(z3min) > 1e-6:
                tr = gp_Trsf()
                tr.SetTranslation(gp_Vec(0, 0, -z3min))
                shape = BRepBuilderAPI_Transform(shape, tr, True).Shape()
        max_dim = max(t_length, t_width)

    elif target_size is not None and isinstance(target_size, (int, float)) and max_dim > 1e-6:
        st = gp_Trsf()
        st.SetScale(gp_Pnt(0, 0, 0), target_size / max_dim)
        shape = BRepBuilderAPI_Transform(shape, st, True).Shape()
        max_dim = float(target_size)

    return shape, max_dim


def transform_shape(shape, rotation_z, position):
    """在场景坐标系中放置模型：先绕 Z 轴旋转，再平移。
    rotation_z 为目标朝向角度（从 +X 轴 CCW，度）；normalize_shape 已将正面对齐到 +X，
    所以此处直接应用场景所需朝向即可。
    """
    rot = gp_Trsf()
    rot.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(rotation_z))
    trans = gp_Trsf()
    trans.SetTranslation(gp_Vec(position[0], position[1], position[2]))
    return BRepBuilderAPI_Transform(shape, trans.Multiplied(rot), True).Shape()


def bbox_to_tuple(bbox):
    if bbox is None:
        return None
    return tuple(float(v) for v in bbox)


def footprint_overlap(a, b, gap=COLLISION_GAP):
    """2D XY 平面投影碰撞检测（忽略 Z 轴）。
    gap: 要求的最小间距（mm），确保家具之间留有空隙。
    """
    axmin, aymin, _, axmax, aymax, _ = a
    bxmin, bymin, _, bxmax, bymax, _ = b
    return not (
        axmax + gap < bxmin or axmin - gap > bxmax or
        aymax + gap < bymin or aymin - gap > bymax
    )


def try_move_furniture(shape, bbox, placed_boxes, room):
    """
    多策略碰撞解决：
    1. 8个方向（每45°），每方向按步长递增推移 MAX_PUSH_ATTEMPTS 步
    2. 若所有方向推移失败，尝试将家具旋转90°后再推移
    成功返回 (new_shape, new_bbox)；无法解决返回 (None, None)。
    """
    def _try_directions(s, bb):
        # 优先方向：先试朝向远离所有碰撞体重心的合方向，再试8个固定方向
        cx = (bb[0] + bb[3]) / 2.0
        cy = (bb[1] + bb[4]) / 2.0
        dx_sum, dy_sum = 0.0, 0.0
        for pb in placed_boxes:
            if not footprint_overlap(bb, pb, COLLISION_GAP):
                continue
            pcx = (pb[0] + pb[3]) / 2.0
            pcy = (pb[1] + pb[4]) / 2.0
            ddx, ddy = cx - pcx, cy - pcy
            dist = math.sqrt(ddx**2 + ddy**2) or 1.0
            dx_sum += ddx / dist
            dy_sum += ddy / dist
        norm = math.sqrt(dx_sum**2 + dy_sum**2)
        if norm > 1e-6:
            primary = [(dx_sum / norm, dy_sum / norm)]
        else:
            primary = []

        # 8个等间距固定方向
        fixed = [(math.cos(math.radians(a)), math.sin(math.radians(a)))
                 for a in range(0, 360, 45)]
        directions = primary + fixed

        for pdx, pdy in directions:
            for attempt in range(1, MAX_PUSH_ATTEMPTS + 1):
                step = PUSH_STEP * attempt
                moved = (
                    bb[0] + pdx * step, bb[1] + pdy * step, bb[2],
                    bb[3] + pdx * step, bb[4] + pdy * step, bb[5],
                )
                sx, sy, _ = clamp_bbox_within_room(moved, room)
                if abs(sx) > 1.0 or abs(sy) > 1.0:
                    break  # 越界，不再向此方向推更远
                if not any(footprint_overlap(moved, pb, COLLISION_GAP) for pb in placed_boxes):
                    new_s = apply_translation(s, pdx * step, pdy * step, 0.0)
                    return new_s, moved
        return None, None

    # 策略1：原始朝向推移
    result = _try_directions(shape, bbox)
    if result[0] is not None:
        return result

    # 策略2：旋转90°后推移
    rotated = rotate_shape_90z(shape)
    rot_bbox = bbox_to_tuple(compute_bounding_box(rotated))
    if rot_bbox is not None:
        result = _try_directions(rotated, rot_bbox)
        if result[0] is not None:
            return result

    return None, None


def clamp_bbox_within_room(bbox, room):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    L, W, H = room.get('length', 6000), room.get('width', 5000), room.get('height', 2800)
    min_x, max_x = -L/2 + ROOM_MARGIN, L/2 - ROOM_MARGIN
    min_y, max_y = -W/2 + ROOM_MARGIN, W/2 - ROOM_MARGIN

    shift_x = 0.0
    shift_y = 0.0
    shift_z = 0.0
    if xmin < min_x:
        shift_x = min_x - xmin
    elif xmax > max_x:
        shift_x = max_x - xmax
    if ymin < min_y:
        shift_y = min_y - ymin
    elif ymax > max_y:
        shift_y = max_y - ymax
    if zmin < 0:
        shift_z = -zmin
    elif zmax > H:
        shift_z = H - zmax
    return shift_x, shift_y, shift_z


def apply_translation(shape, dx, dy, dz):
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def rotate_shape_90z(shape):
    """绕家具自身中心竖直轴旋转 90°（水平旋转，保持 Z 朝上，位置不变）。
    旋转轴方向 = 房间 Z 轴 [0,0,1]，旋转中心 = 家具占地面中心（而非世界原点）。
    """
    # 先算包围盒取 XY 中心，确保旋转后家具位置不漂移
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    if bbox.IsVoid():
        cx, cy = 0.0, 0.0
    else:
        xmin, ymin, _, xmax, ymax, _ = bbox.Get()
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(cx, cy, 0.0), gp_Dir(0, 0, 1)), math.pi / 2.0)
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def _read_meta(folder):
    """读取 alignment_meta.json，返回 (align_rot, forward_angle, flip_x)。

    Align_model_v2.py 写入的字段：
      align_rotation   : [rx, ry, rz]  — 使模型站立的 Euler XYZ 旋转
      extra_rz         : float         — 绕 Z 轴的正面修正角度（度）
        约定：extra_rz 使正面朝向 -Y；-Y 在 OCC XY 平面对应 270°（或 -90°）
        因此 forward_angle = 270.0 + extra_rz，归一化到 [0, 360)
      flip_correction  : bool          — 可选，模型上下颠倒时手动置 true
    """
    meta_path = os.path.join(folder, 'alignment_meta.json')
    align_rot = [0, 0, 0]
    forward_angle = 0.0
    flip_x = False
    if not os.path.exists(meta_path):
        return align_rot, forward_angle, flip_x
    try:
        meta = json.load(open(meta_path, 'r', encoding='utf-8'))
        if meta.get('status') not in ('verified',):
            # 未成功对齐（manual_review 等）：用默认值，不施加任何朝向旋转
            return align_rot, forward_angle, flip_x
        align_rot = meta.get('align_rotation', align_rot)
        flip_x = bool(meta.get('flip_correction', False))
        extra_rz = float(meta.get('extra_rz', 0.0))
        # extra_rz 使正面朝 -Y（OCC 角度 270°），正面在 OCC XY 平面的最终角度：
        forward_angle = (270.0 + extra_rz) % 360.0
    except Exception:
        pass
    return align_rot, forward_angle, flip_x


def _process_one_furniture(item, placed_boxes, room):
    """处理单件家具：STEP读取 → 标准化 → 变换 → 边界校正 → 碰撞检测。
    返回 (placed_shape, bbox, skip_reason)；skip_reason 为 None 表示成功。
    此函数在独立线程中运行以支持超时保护。
    """
    folder = item.get('folder', '').strip()
    if not folder:
        return None, None, '缺少 folder 路径'

    step_path = find_step_file(folder)
    if not step_path:
        return None, None, '未找到 STEP 文件'

    try:
        shape = load_step_shape(step_path)
    except Exception as exc:
        return None, None, f'STEP 读取失败: {exc}'

    align_rot, forward_angle, flip_x = _read_meta(folder)
    target_size = item.get('target_size') or item.get('target_max_size', None)
    try:
        shape, _ = normalize_shape(
            shape,
            target_size=target_size,
            align_rot=tuple(align_rot),
            forward_angle=forward_angle,
            flip_x=flip_x,
        )
    except Exception as exc:
        return None, None, f'标准化失败: {exc}'

    position = item.get('position', [0.0, 0.0, 0.0])
    try:
        rotation_z = item.get('rotation_z', 0.0)
        placed_shape = transform_shape(shape, rotation_z, position)
        bbox = bbox_to_tuple(compute_bounding_box(placed_shape))
    except Exception as exc:
        return None, None, f'变换/包围盒失败: {exc}'

    if bbox is None:
        return None, None, '无法计算边界盒'

    try:
        shift_x, shift_y, shift_z = clamp_bbox_within_room(bbox, room)
        if abs(shift_x) > 0.1 or abs(shift_y) > 0.1 or abs(shift_z) > 0.1:
            placed_shape = apply_translation(placed_shape, shift_x, shift_y, shift_z)
            bbox = bbox_to_tuple(compute_bounding_box(placed_shape))
    except Exception as exc:
        print(f"   ⚠️  {item.get('id','?')}: 边界调整失败，使用原始位置 ({exc})")

    # 碰撞检测（2D XY 平面投影）
    if any(footprint_overlap(bbox, pb, COLLISION_GAP) for pb in placed_boxes):
        new_shape, new_bbox = try_move_furniture(placed_shape, bbox, placed_boxes, room)
        if new_shape is not None:
            return new_shape, new_bbox, None   # 碰撞推移成功
        return None, None, '碰撞且平移无法解决，已跳过'

    return placed_shape, bbox, None


def build_scene(layout_path, output_path, max_furniture=MAX_FURNITURE_PER_SCENE,
                furniture_timeout=FURNITURE_TIMEOUT, scene_timeout=SCENE_TIMEOUT):
    import time as _time
    with open(layout_path, 'r', encoding='utf-8') as f:
        layout = json.load(f)

    room = layout.get('room', {})
    furniture = layout.get('furniture', [])[:max_furniture]
    placed = []
    placed_boxes = []
    skipped = []

    shapes = make_room_geometry(room)

    scene_deadline = _time.monotonic() + scene_timeout

    # 单线程 executor — 仅用于给每件家具施加超时，OCP 不支持真并行
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
        for item in furniture:
            if _time.monotonic() > scene_deadline:
                remaining = len(furniture) - len(placed) - len(skipped)
                print(f"   ⏰ 场景超时（>{scene_timeout}s），剩余 {remaining} 件家具跳过")
                break

            fid = item.get('id', '?')
            future = exe.submit(_process_one_furniture, item, list(placed_boxes), room)
            try:
                placed_shape, bbox, reason = future.result(timeout=furniture_timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                skipped.append((item, f'超时（>{furniture_timeout}s），跳过'))
                print(f"   ⏰ {fid}: 超时跳过")
                continue
            except Exception as exc:
                skipped.append((item, f'未知异常: {exc}'))
                print(f"   ❌ {fid}: 未知异常: {exc}")
                continue

            if reason is not None:
                skipped.append((item, reason))
                if '碰撞' in reason:
                    print(f"   ⏭  {fid}: {reason}")
                continue

            # 碰撞推移成功时打印提示（原始 bbox 不同于 new_bbox）
            if any(footprint_overlap(bbox, pb, COLLISION_GAP) for pb in placed_boxes):
                print(f"   ➡  {fid}: 碰撞已推移解决")

            placed.append(item)
            placed_boxes.append(bbox)
            shapes.append(placed_shape)

    try:
        writer = STEPControl_Writer()
        for shape in shapes:
            writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(output_path)
    except Exception as exc:
        raise RuntimeError(f"STEP 写出失败: {exc}") from exc

    return {
        'layout': os.path.basename(layout_path),
        'output': os.path.basename(output_path),
        'placed_count': len(placed),
        'skipped_count': len(skipped),
        'skipped': skipped,
    }


def build_all_scenes(layout_dir=LAYOUT_DIR, output_dir=OUTPUT_DIR, max_scenes=MAX_SCENES, max_furniture=MAX_FURNITURE_PER_SCENE):
    os.makedirs(output_dir, exist_ok=True)
    layout_files = sorted([f for f in os.listdir(layout_dir) if f.endswith('.json')])
    summary = []
    for idx, layout_file in enumerate(layout_files):
        if idx >= max_scenes:
            break
        layout_path = os.path.join(layout_dir, layout_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(layout_file)[0]}.step")
        print(f"处理场景 {idx+1}/{min(len(layout_files), max_scenes)}: {layout_file}")
        try:
            result = build_scene(layout_path, output_path, max_furniture=max_furniture)
            print(f"  已放置: {result['placed_count']}，跳过: {result['skipped_count']}")
            for item, reason in result['skipped']:
                print(f"    跳过 {item.get('id','unknown')}: {reason}")
            summary.append(result)
        except Exception as exc:
            print(f"  ❌ 场景构建失败，跳过此场景继续: {exc}")
    summary_path = os.path.join(output_dir, 'scene_build_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, indent=4, ensure_ascii=False)
    print(f"完成。生成场景数量: {len(summary)}，汇总保存到 {summary_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="layout JSON → STEP 场景组装")
    parser.add_argument('--layout-dir', default=LAYOUT_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--max', type=int, default=MAX_SCENES, help='最多处理场景数（默认全量）')
    parser.add_argument('--max-furniture', type=int, default=MAX_FURNITURE_PER_SCENE)
    args = parser.parse_args()
    build_all_scenes(
        layout_dir=args.layout_dir,
        output_dir=args.output_dir,
        max_scenes=args.max,
        max_furniture=args.max_furniture,
    )
