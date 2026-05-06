"""
Align_model_v2.py  —  六方向站立判断 + 水平朝向判断（两阶段 VLM 对齐）

流程：
  阶段一（站立判断）：
    将模型分别旋转到 6 种朝向（6 面朝上），每种朝向渲染 3 张带俯视角的环绕图，
    拼成 6组×3图 的大图，让 VLM 判断哪一组模型是正确站立在地面上的。

  阶段二（朝向判断）：
    在正确站立姿态下，渲染 4 个水平朝向（每隔 90°），带透视的正视图，
    让 VLM 判断哪个角度是模型的正前方。

  计算标签：
    程序根据 VLM 返回的组号+角度号自动计算最终对齐旋转参数，不依赖 VLM 计算。

  断点续传：alignment_meta.json 存在且 status="verified"/"manual_review" 时跳过。

用法：
  python Align_model_v2.py                  # 使用默认区间
  python Align_model_v2.py 1 500            # 处理第 1~500 个模型
"""

import os, sys, json, re, time, math, base64, requests, tempfile
import numpy as np
from scipy.spatial.transform import Rotation as SciR
from PIL import Image, ImageDraw, ImageFont

from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Dir
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

import trimesh

# ── 强制 pyrender 使用 NVIDIA GPU（必须在 import pyrender 之前设置）──────────
# Windows 上 pyglet/OpenGL 默认可能选中核显；通过 pyglet 平台提示强制 NVIDIA。
import pyglet
pyglet.options["shadow_window"] = False          # 禁止 pyglet 创建隐藏 shadow window
pyglet.options["debug_gl"] = False
# 枚举可用 GPU，优先选 NVIDIA
try:
    _displays = pyglet.canvas.get_display().get_screens()
    _nvidia = None
    for _s in _displays:
        if hasattr(_s, "name") and "nvidia" in str(getattr(_s, "name", "")).lower():
            _nvidia = _s
    if _nvidia:
        pyglet.options["display"] = _nvidia
except Exception:
    pass

import os as _os
# pyglet 的 EGL headless 后端在 Windows 不可用，但可以通过 NV_OPTIMUS 环境变量
# 告知驱动强制走 NVIDIA（对 Optimus 笔记本有效；台式机双卡需在 NVIDIA 控制面板设置）
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # 指定第一块 GPU（RTX 4090）

import pyrender

# ══════════════════════════════════════════════════════════════ 配置区 ══════
DATASET_ROOT   = r"D:\Lzm_Temp_Data\Li_temp_project\Dataset_Categoried\1_1000"
API_KEY        = ""
URL            = ""
MODEL_NAME     = "qwen3vl"

TEST_LIMIT     = 5000      # 本次最多调用 API 次数
SAFE_WAIT_TIME = 1         # 每个模型处理完后等待秒数
START_FROM     = 1         # 默认处理区间起始（1-indexed）
END_AT         = 4044      # 默认处理区间结束
OVERWRITE_ALL  = False     # True = 覆盖所有已验证通过的模型

VIEW_SIZE      = 448       # 单视角分辨率（px）；提升后 VLM 可见更多细节
SSAA_FACTOR    = 2         # 超采样倍数：离屏以 VIEW_SIZE*SSAA_FACTOR 渲染再缩小，4090 轻松承担
MESH_DEFLECT   = 1.0       # STL 三角化弦差（mm），越小越精细；64GB 内存可以更小
MESH_MAX_FACES = 1_000_000 # trimesh 渲染前最大面数；64GB 内存可放宽到百万级
STEP_SIZE_LARGE = 30 * 1024 * 1024   # STEP 文件超过此字节数视为"大文件"
HEADER_H       = 30        # 视角标注栏高度（px）
# ═══════════════════════════════════════════════════════════════════════════

SUMMARY_PATH = os.path.join(DATASET_ROOT, "manual_review_summary.json")

# ── 阶段一：6种朝向定义 ──────────────────────────────────────────────────
# 每种朝向 = 将模型某个面转到朝上（+Z方向）
# 格式：(组号, 朝上面描述, 旋转矩阵使该面朝上)
# 6个朝向：原始的 +Z/-Z/+X/-X/+Y/-Y 面朝上
# 旋转用 Euler XYZ 角度表示（rx, ry, rz），应用到原始模型使该面朝上

# 6种朝向：将模型的各个面分别转到 +Z 方向（朝上）
# 定义：(组号1-6, 标签, (rx,ry,rz)度数)
# 组1: 原始 +Z 面朝上（不旋转）
# 组2: 原始 -Z 面朝上（绕X转180°）
# 组3: 原始 +X 面朝上（绕Y转-90°）
# 组4: 原始 -X 面朝上（绕Y转+90°）
# 组5: 原始 +Y 面朝上（绕X转+90°）
# 组6: 原始 -Y 面朝上（绕X转-90°）
ORIENTATION_GROUPS = [
    (1, "+Z up (original)",  (  0.0,   0.0,  0.0)),
    (2, "-Z up (flipped)",   (180.0,   0.0,  0.0)),
    (3, "+X up",             (  0.0,  90.0,  0.0)),
    (4, "-X up",             (  0.0, -90.0,  0.0)),
    (5, "+Y up",             (-90.0,   0.0,  0.0)),
    (6, "-Y up",             ( 90.0,   0.0,  0.0)),
]

# 阶段一：每组3个相机方向（稍微俯视约25°，水平三个角度：0°/120°/240°）
# 俯视角25°：eye_dir = (cos(0°)*cos(25°), sin(0°)*cos(25°), sin(25°))
def _phase1_eye_dirs():
    tilt = math.radians(25)
    dirs = []
    for az_deg in [0, 120, 240]:
        az = math.radians(az_deg)
        x = math.cos(az) * math.cos(tilt)
        y = math.sin(az) * math.cos(tilt)
        z = math.sin(tilt)
        dirs.append(np.array([x, y, z]))
    return dirs

PHASE1_EYE_DIRS = _phase1_eye_dirs()

# 阶段二：4个水平朝向，每隔90°，带25°俯视（固定从正前方看）
# 正视角度：从 -Y 方向看（模型正面朝向 -Y），加25°俯视
# 4个角度对应旋转模型0/90/180/270°，相机方向固定
PHASE2_ANGLES = [0, 90, 180, 270]  # 绕Z轴旋转模型的角度

# ── 类别视觉特征提示 ──────────────────────────────────────────────────────
_HINTS: dict[str, dict] = {
    "chair":          {"top": "flat seat or cushion surface",           "front": "face with backrest visible from seated user's side"},
    "sofa":           {"top": "row of seat cushions along the top",     "front": "face showing armrests on both sides with backrest"},
    "table":          {"top": "large flat horizontal tabletop surface", "front": "side with symmetrical leg structure or drawers"},
    "desk":           {"top": "flat work surface",                      "front": "side with drawers, modesty panel, or keyboard tray"},
    "bed":            {"top": "mattress or sleeping surface area",      "front": "headboard side (taller, more decorative structure)"},
    "cabinet":        {"top": "flat horizontal top panel",             "front": "side with doors, drawers, or handles"},
    "lamp":           {"top": "shade top or light-emitting opening",   "front": "side with switch or decorative element"},
    "storagefurniture":{"top":"flat horizontal top panel",             "front": "side with doors, drawers, or handles"},
}

def _get_hints(category: str) -> tuple[str, str]:
    key = category.lower().replace(" ", "").replace("_", "")
    for k, v in _HINTS.items():
        if k.replace("_", "") in key:
            return v["top"], v["front"]
    return "functional top surface of the object", "face toward the user during normal use"


# ══════════════════════════════════════════════════════════════ 工具函数 ══════

def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _add_header(img: Image.Image, text: str, bg_hex: str, h: int = HEADER_H) -> Image.Image:
    W, H = img.size
    canvas = Image.new("RGB", (W, H + h), _hex_rgb(bg_hex))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = None
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((W - tw) // 2, (h - th) // 2), text, fill="white", font=font)
    canvas.paste(img, (0, h))
    return canvas

def _add_group_label(img: Image.Image, text: str) -> Image.Image:
    """在图像左侧添加竖向组标签（用于区分6组）。"""
    W, H = img.size
    label_w = 28
    canvas = Image.new("RGB", (W + label_w, H), (40, 40, 40))
    canvas.paste(img, (label_w, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = None
    # 竖向文字用旋转
    tmp = Image.new("RGB", (H, label_w), (40, 40, 40))
    d2 = ImageDraw.Draw(tmp)
    d2.text((10, 6), text, fill=(255, 220, 80), font=font)
    rotated = tmp.rotate(90, expand=True)
    canvas.paste(rotated, (0, 0))
    return canvas


# ══════════════════════════════════════════════════════════ STEP / STL 处理 ══════

def _load_step(step_path: str):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    shape = reader.OneShape()
    return None if shape.IsNull() else shape

def _occ_translate(shape, dx, dy, dz):
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()

def _occ_rotate(shape, axis_dir: gp_Dir, angle_deg: float):
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), axis_dir), math.radians(angle_deg))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()

def _occ_apply_euler_xyz(shape, rx_deg, ry_deg, rz_deg):
    """应用 XYZ Euler 旋转（Rz @ Ry @ Rx）。"""
    Rx = gp_Trsf()
    Rx.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(1,0,0)), math.radians(rx_deg))
    Ry = gp_Trsf()
    Ry.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(0,1,0)), math.radians(ry_deg))
    Rz = gp_Trsf()
    Rz.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(0,0,1)), math.radians(rz_deg))
    return BRepBuilderAPI_Transform(shape, Rz.Multiplied(Ry).Multiplied(Rx), True).Shape()

def _occ_center_bottom(shape):
    """将包围盒 XY 中心移到原点，底部(Zmin)落到 Z=0。"""
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    if bbox.IsVoid():
        return shape
    xmin, ymin, zmin, xmax, ymax, _ = bbox.Get()
    return _occ_translate(shape, -(xmin+xmax)/2, -(ymin+ymax)/2, -zmin)

def _adaptive_deflect(step_path: str) -> float:
    """根据 STEP 文件大小自适应计算三角化弦差。64GB 内存可用更精细的网格。"""
    try:
        size = os.path.getsize(step_path)
    except OSError:
        return MESH_DEFLECT
    if size > STEP_SIZE_LARGE * 4:   # >120 MB
        return 3.5
    if size > STEP_SIZE_LARGE * 2:   # >60 MB
        return 2.0
    if size > STEP_SIZE_LARGE:       # >30 MB
        return 1.5
    return MESH_DEFLECT


def _write_stl(shape, path: str, deflect: float = MESH_DEFLECT) -> bool:
    mesh = BRepMesh_IncrementalMesh(shape, deflect, False)
    mesh.Perform()
    StlAPI_Writer().Write(shape, path)
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_oriented_stl(step_path: str, rx: float, ry: float, rz: float,
                      extra_rz: float, tmp_dir: str, suffix: str,
                      preloaded_shape=None) -> str | None:
    """
    加载 STEP（或复用 preloaded_shape），应用朝向旋转(rx,ry,rz)，居中落底，
    再绕Z轴旋转 extra_rz 度，写出 STL。suffix 用于区分不同临时文件。
    preloaded_shape: 已读取的 OCC shape，避免重复加载大文件。
    """
    shape = preloaded_shape if preloaded_shape is not None else _load_step(step_path)
    if shape is None:
        return None
    deflect = _adaptive_deflect(step_path)
    shape = _occ_apply_euler_xyz(shape, rx, ry, rz)
    shape = _occ_center_bottom(shape)
    if abs(extra_rz) > 0.01:
        shape = _occ_rotate(shape, gp_Dir(0, 0, 1), extra_rz)
        shape = _occ_center_bottom(shape)
    out = os.path.join(tmp_dir, f"oriented_{suffix}.stl")
    return out if _write_stl(shape, out, deflect) else None


# ══════════════════════════════════════════════════════════════ 场景渲染 ══════

_RENDERER: pyrender.OffscreenRenderer | None = None
_RENDER_RES = VIEW_SIZE * SSAA_FACTOR   # 实际离屏分辨率，渲染后缩小到 VIEW_SIZE

def _get_renderer() -> pyrender.OffscreenRenderer:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = pyrender.OffscreenRenderer(_RENDER_RES, _RENDER_RES)
    return _RENDERER


def _load_trimesh(stl_path: str) -> trimesh.Trimesh | None:
    try:
        m = trimesh.load(stl_path, force="mesh")
        if isinstance(m, trimesh.Scene):
            geoms = list(m.geometry.values())
            if not geoms:
                return None
            m = trimesh.util.concatenate(geoms)
        if not isinstance(m, trimesh.Trimesh) or len(m.vertices) == 0:
            return None
        # 对超大网格做面数简化，防止渲染器OOM或显存不足
        if len(m.faces) > MESH_MAX_FACES:
            ratio = MESH_MAX_FACES / len(m.faces)
            try:
                m = m.simplify_quadric_decimation(int(len(m.faces) * ratio))
                print(f"   ℹ️  网格简化至 {len(m.faces)} 面（原过大）")
            except Exception:
                pass  # 简化失败则继续用原始网格
        return m
    except Exception:
        return None


def _make_pyrender_mesh(tm: trimesh.Trimesh) -> pyrender.Mesh:
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.36, 0.55, 0.72, 1.0],
        metallicFactor=0.25,
        roughnessFactor=0.55,
        doubleSided=True,
    )
    return pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True)


def _camera_distance(tm: trimesh.Trimesh, eye_dir: np.ndarray, fov_deg: float) -> float:
    eye_n = eye_dir / np.linalg.norm(eye_dir)
    verts = tm.vertices
    proj = verts @ eye_n
    perp1 = np.cross(eye_n, [0, 0, 1] if abs(eye_n[2]) < 0.9 else [1, 0, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(eye_n, perp1)
    p1_proj = verts @ perp1; s1 = (p1_proj.max() - p1_proj.min()) / 2
    p2_proj = verts @ perp2; s2 = (p2_proj.max() - p2_proj.min()) / 2
    half_span = max(s1, s2, 1e-3)
    depth_offset = (proj.max() - proj.min()) / 2
    half_fov = math.radians(fov_deg / 2)
    dist = half_span / (math.tan(half_fov) * 0.68) + depth_offset
    return float(dist)


def _make_camera_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = eye - target
    z = z / np.linalg.norm(z)
    up = np.array([0., 0., 1.])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0., 1., 0.])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x; pose[:3, 1] = y; pose[:3, 2] = z; pose[:3, 3] = eye
    return pose


def _build_scene_with_ground(tm: trimesh.Trimesh) -> pyrender.Scene:
    scene = pyrender.Scene(
        ambient_light=[0.25, 0.25, 0.25],
        bg_color=[0.93, 0.93, 0.93, 1.0],
    )
    scene.add(_make_pyrender_mesh(tm))
    _add_ground_plane(scene, tm)
    return scene


def _add_ground_plane(scene: pyrender.Scene, tm: trimesh.Trimesh) -> None:
    bb = tm.bounding_box.bounds
    zmin = bb[0, 2]
    span = max(bb[1, 0] - bb[0, 0], bb[1, 1] - bb[0, 1]) * 2.0
    cx = (bb[0, 0] + bb[1, 0]) / 2
    cy = (bb[0, 1] + bb[1, 1]) / 2
    half = span / 2

    verts = np.array([
        [cx - half, cy - half, zmin],
        [cx + half, cy - half, zmin],
        [cx + half, cy + half, zmin],
        [cx - half, cy + half, zmin],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    ground_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    ground_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.55, 0.55, 0.55, 0.50],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
        alphaMode="BLEND",
    )
    scene.add(pyrender.Mesh.from_trimesh(ground_tm, material=ground_mat, smooth=False))


def _render_view(scene: pyrender.Scene, tm: trimesh.Trimesh,
                 eye_dir: np.ndarray, fov_deg: float = 42.0) -> Image.Image:
    r = _get_renderer()
    center = tm.bounding_box.centroid

    eye_n = np.array(eye_dir, float)
    eye_n = eye_n / np.linalg.norm(eye_n)
    dist  = _camera_distance(tm, eye_n, fov_deg)
    eye   = center + eye_n * dist
    pose  = _make_camera_pose(eye, center)

    proj       = tm.vertices @ eye_n
    depth_span = proj.max() - proj.min()
    znear = max(dist - depth_span * 0.6 - depth_span * 0.1, dist * 0.01, 0.1)
    zfar  = dist + depth_span * 0.6 + depth_span * 0.1

    cam = pyrender.PerspectiveCamera(yfov=math.radians(fov_deg), aspectRatio=1.0,
                                     znear=znear, zfar=zfar)
    cam_node = scene.add(cam, pose=pose)

    light1 = pyrender.DirectionalLight(color=[1., 1., 1.], intensity=4.5)
    ln1    = scene.add(light1, pose=pose)

    fill_eye  = center - (eye - center)
    fill_pose = _make_camera_pose(fill_eye, center)
    light2 = pyrender.DirectionalLight(color=[0.8, 0.85, 1.], intensity=1.8)
    ln2    = scene.add(light2, pose=fill_pose)

    top_pose = _make_camera_pose(center + np.array([0., 0., dist]), center)
    light3 = pyrender.DirectionalLight(color=[1., 0.95, 0.9], intensity=1.2)
    ln3    = scene.add(light3, pose=top_pose)

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    img = Image.fromarray(color[:, :, :3])
    if SSAA_FACTOR > 1:
        img = img.resize((VIEW_SIZE, VIEW_SIZE), Image.LANCZOS)

    scene.remove_node(cam_node)
    scene.remove_node(ln1)
    scene.remove_node(ln2)
    scene.remove_node(ln3)
    return img


# ══════════════════════════════════════════════════════ 阶段一渲染：6组×3图 ══════

GROUP_COLORS = ["#1A6B9A", "#28A745", "#C0392B", "#8E44AD", "#E67E22", "#16A085"]

def render_phase1_grid(step_path: str, tmp_dir: str) -> str | None:
    """
    渲染阶段一大图：6组×3视角 = 18张子图。
    布局：每行=一组（6行），每列=一个视角（3列）。
    每组左侧有组号标签，视角标题在图片顶部。
    返回合并图路径。
    """
    try:
        cell_w = VIEW_SIZE
        cell_h = VIEW_SIZE + HEADER_H
        n_groups = len(ORIENTATION_GROUPS)
        n_views  = len(PHASE1_EYE_DIRS)

        # 总图尺寸：3列 × 6行，每列额外加左侧组标签宽度
        label_w = 32
        total_w = label_w + cell_w * n_views
        total_h = cell_h * n_groups
        grid = Image.new("RGB", (total_w, total_h), (240, 240, 240))

        # 大文件只加载一次 STEP，6组复用同一个 shape
        print(f"   读取 STEP 文件...")
        base_shape = _load_step(step_path)
        if base_shape is None:
            print(f"   ❌ STEP 加载失败")
            return None

        for gi, (gid, glabel, (rx, ry, rz)) in enumerate(ORIENTATION_GROUPS):
            print(f"   渲染组 G{gid}/G{len(ORIENTATION_GROUPS)}...")
            stl_path = load_oriented_stl(step_path, rx, ry, rz, 0.0, tmp_dir, f"g{gid}",
                                          preloaded_shape=base_shape)
            if stl_path is None:
                print(f"   ⚠️ 组{gid} STL生成失败，跳过")
                continue

            tm = _load_trimesh(stl_path)
            if tm is None:
                print(f"   ⚠️ 组{gid} STL加载失败，跳过")
                continue

            scene = _build_scene_with_ground(tm)
            color_hex = GROUP_COLORS[gi % len(GROUP_COLORS)]

            # 绘制左侧组标签
            label_img = Image.new("RGB", (label_w, cell_h), _hex_rgb(color_hex))
            ld = ImageDraw.Draw(label_img)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                font = None
            grp_text = f"G{gid}"
            ld.text((4, cell_h // 2 - 8), grp_text, fill=(255, 255, 255), font=font)
            grid.paste(label_img, (0, gi * cell_h))

            for vi, eye_dir in enumerate(PHASE1_EYE_DIRS):
                img = _render_view(scene, tm, eye_dir, fov_deg=40.0)
                view_angle = [0, 120, 240][vi]
                header_text = f"G{gid} · View {vi+1} ({view_angle}°)"
                img = _add_header(img.resize((cell_w, VIEW_SIZE)), header_text, color_hex)
                x = label_w + vi * cell_w
                y = gi * cell_h
                grid.paste(img, (x, y))

        out = os.path.join(tmp_dir, "phase1_grid.jpg")
        grid.save(out, "JPEG", quality=88)
        return out
    except Exception as e:
        print(f"   ❌ Phase-1 render failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════ 阶段二渲染：4方向正视图 ══════

def render_phase2_grid(step_path: str, upright_rx: float, upright_ry: float,
                       upright_rz: float, tmp_dir: str) -> str | None:
    """
    在已确定站立姿态下，渲染4个水平朝向的正视图（相机在-Y方向+25°俯视）。
    每个视角对应将模型绕Z轴旋转 0/90/180/270 度后从 -Y 方向观察。
    """
    try:
        cell_w = VIEW_SIZE
        cell_h = VIEW_SIZE + HEADER_H
        total_w = cell_w * 4
        total_h = cell_h
        grid = Image.new("RGB", (total_w, total_h), (240, 240, 240))

        # 相机方向：从 -Y 方向带25°俯视
        tilt = math.radians(25)
        cam_eye_dir = np.array([0.0, -math.cos(tilt), math.sin(tilt)])

        colors = ["#2471A3", "#1E8449", "#922B21", "#7D3C98"]
        angle_labels = ["0°", "90°", "180°", "270°"]

        # 大文件只加载一次 STEP，4个角度复用
        print(f"   读取 STEP 文件（阶段二）...")
        base_shape = _load_step(step_path)
        if base_shape is None:
            print(f"   ❌ STEP 加载失败")
            return None

        for i, angle in enumerate(PHASE2_ANGLES):
            print(f"   渲染角度 {angle}°...")
            stl_path = load_oriented_stl(step_path, upright_rx, upright_ry, upright_rz,
                                          float(angle), tmp_dir, f"p2_a{angle}",
                                          preloaded_shape=base_shape)
            if stl_path is None:
                print(f"   ⚠️ 阶段二 {angle}° STL生成失败")
                continue

            tm = _load_trimesh(stl_path)
            if tm is None:
                continue

            scene = _build_scene_with_ground(tm)
            img = _render_view(scene, tm, cam_eye_dir, fov_deg=40.0)
            header = f"Angle {i+1} ({angle}°)"
            img = _add_header(img.resize((cell_w, VIEW_SIZE)), header, colors[i])
            grid.paste(img, (i * cell_w, 0))

        out = os.path.join(tmp_dir, "phase2_grid.jpg")
        grid.save(out, "JPEG", quality=90)
        return out
    except Exception as e:
        print(f"   ❌ Phase-2 render failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════ VLM 交互 ══════

def _query_vlm(image_path: str, prompt: str) -> dict | None:
    import random
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{_encode(image_path)}"}},
        ]}],
        "temperature": 0.0,
        "top_p": 0.9,
    }
    max_attempts = 6
    base_wait = 3.0
    for attempt in range(max_attempts):
        try:
            resp = requests.post(URL, headers=headers, json=data, timeout=120)
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
                print(f"   ⚠️ 无法解析 JSON: {raw[:300]}")
                # JSON解析失败不属于限速，直接视为无效响应
                return None
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 0))
                wait = max(retry_after, int(base_wait * (2 ** attempt))) + random.uniform(1, 5)
                print(f"   ⚠️ HTTP 429 限速，等待 {wait:.0f}s 后重试({attempt+1}/{max_attempts})...")
                time.sleep(wait)
            else:
                wait = base_wait * (2 ** attempt) + random.uniform(0, 2)
                print(f"   ⚠️ HTTP {resp.status_code}，等待 {wait:.0f}s 后重试({attempt+1}/{max_attempts})...")
                time.sleep(wait)
        except Exception as e:
            wait = base_wait * (2 ** attempt) + random.uniform(0, 2)
            print(f"   ⚠️ 请求异常: {e}，等待 {wait:.0f}s 后重试({attempt+1}/{max_attempts})...")
            time.sleep(wait)
    return None


def _build_phase1_prompt(category: str) -> str:
    top_hint, front_hint = _get_hints(category)
    return f"""You are a 3D furniture model orientation expert.

The image shows a [{category}] CAD model placed in 6 different orientations (Groups G1–G6).
Each group occupies one row. Within each group, 3 views are shown from different angles around the model (0°, 120°, 240° horizontal rotation, all with ~25° downward tilt).

The model has a semi-transparent grey ground plane. Each group places the model with a different face pointing upward.

Your task: Identify which group (G1–G6) shows the model standing correctly upright — meaning:
- The model is NOT lying on its side and NOT upside-down
- The model's functional top surface faces upward
- The model's base rests naturally on the ground plane

Visual features of a [{category}]:
- Correct top surface: {top_hint}
- Correct front/usage face: {front_hint}

Instructions:
1. Look at each group's 3 views. Judge the overall 3D shape.
2. Identify which group shows the model in its natural upright standing position.

--- Output ONLY raw JSON, no markdown fences ---
{{
  "reasoning": "brief explanation of which group looks upright and why",
  "upright_group": <integer 1-6>,
  "confidence": "high/medium/low"
}}"""


def _build_phase2_prompt(category: str) -> str:
    _, front_hint = _get_hints(category)
    return f"""You are a 3D furniture model orientation expert.

The image shows a [{category}] model standing upright (correct orientation confirmed), rendered from 4 horizontal viewing directions (Angle 1 to Angle 4, each 90° apart around the vertical axis).

Your task: Identify which angle shows the model's FRONT face directly facing the camera.

The front face of a [{category}]: {front_hint}

Instructions:
1. Look at all 4 angles.
2. Identify which one shows the natural front face of the object pointing toward the viewer.
3. The front face is typically the most "user-facing" surface during normal use.

--- Output ONLY raw JSON, no markdown fences ---
{{
  "reasoning": "brief explanation of which angle shows the front face",
  "front_angle_index": <integer 1-4>,
  "confidence": "high/medium/low"
}}"""


# ══════════════════════════════════════════════════════ 旋转参数计算 ══════

def compute_alignment(upright_group_id: int, front_angle_index: int) -> dict:
    """
    根据阶段一选出的站立组和阶段二选出的正面角度，计算最终对齐旋转参数。

    站立组决定了将哪个面转到 +Z 朝上（即 align_rotation = 该组的 euler 角）。
    正面角度决定了额外绕 Z 轴旋转多少度，使正面朝向 -Y（即正前方）。

    约定：正面朝向 -Y 方向（角度为 front_angle_index 时正面对着相机，
          相机在 -Y 方向，因此正面朝 -Y）。
    四个角度对应的额外 Z 旋转：
      - front_angle_index=1 → 模型旋转 0° 时正面朝向相机 → 无需额外旋转
      - front_angle_index=2 → 模型旋转 90° 时正面朝向相机 → 需要再旋 -90° 回正
      - front_angle_index=3 → 模型旋转 180° 时正面朝向 → 需要再旋 -180°（或+180°）
      - front_angle_index=4 → 模型旋转 270° 时正面朝向 → 需要再旋 -270°（或+90°）
    即最终额外 Z 旋转 = -(front_angle_index - 1) * 90°，归一化到 (-180, 180]
    """
    _, _, (rx, ry, rz) = ORIENTATION_GROUPS[upright_group_id - 1]
    extra_rz_raw = -(front_angle_index - 1) * 90.0
    extra_rz = ((extra_rz_raw + 180.0) % 360.0) - 180.0
    return {
        "align_rotation": [rx, ry, rz],
        "extra_rz": extra_rz,
        "upright_group": upright_group_id,
        "front_angle_index": front_angle_index,
    }


def _safe_int(val, fallback: int = 0) -> int:
    """Convert VLM field to int; fall back to scanning reasoning text if val is empty/invalid."""
    if val is None or val == "":
        return fallback
    try:
        return int(val)
    except (ValueError, TypeError):
        return fallback


def _extract_int_from_reasoning(text: str, candidates: list[int]) -> int:
    """
    When the structured field is missing, scan the reasoning text for the first
    occurrence of a candidate integer (e.g. 'G6' → 6, 'Angle 3' → 3).
    Returns 0 if nothing found.
    """
    import re as _re
    for c in candidates:
        # match standalone digit, possibly preceded by 'G', 'group', 'angle', etc.
        if _re.search(rf'\b[Gg]?\s*{c}\b', text):
            return c
    return 0


# ══════════════════════════════════════════════════════════ 单模型处理 ══════

def _run_phase1(step_path: str, category: str, tmp_dir: str, api_ref: list) -> dict | None:
    grid_path = render_phase1_grid(step_path, tmp_dir)
    if grid_path is None:
        return None

    print(f"   [Phase1] VLM 判断站立组 ({category})...")
    res = _query_vlm(grid_path, _build_phase1_prompt(category))
    api_ref[0] += 1
    if res is None:
        print("   [Phase1] 无 VLM 响应")
        return None

    reasoning = res.get("reasoning", "")
    gid = _safe_int(res.get("upright_group", 0))
    if gid < 1 or gid > 6:
        gid = _extract_int_from_reasoning(reasoning, list(range(1, 7)))
    conf = res.get("confidence", "low")
    print(f"   [Phase1] {reasoning}")
    print(f"   [Phase1] 站立组={gid}, 置信={conf}")

    if gid < 1 or gid > 6:
        print(f"   [Phase1] 无效组号 {gid}，VLM原始响应: {res}")
        return None

    return {"upright_group": gid, "confidence": conf,
            "vlm_decision": res, "grid_path": grid_path}


def _run_phase2(step_path: str, category: str, upright_group: int,
                tmp_dir: str, api_ref: list) -> dict | None:
    _, _, (rx, ry, rz) = ORIENTATION_GROUPS[upright_group - 1]
    grid_path = render_phase2_grid(step_path, rx, ry, rz, tmp_dir)
    if grid_path is None:
        return None

    print(f"   [Phase2] VLM 判断正面朝向 ({category})...")
    res = _query_vlm(grid_path, _build_phase2_prompt(category))
    api_ref[0] += 1
    if res is None:
        print("   [Phase2] 无 VLM 响应")
        return None

    reasoning2 = res.get("reasoning", "")
    aidx = _safe_int(res.get("front_angle_index", 0))
    if aidx < 1 or aidx > 4:
        aidx = _extract_int_from_reasoning(reasoning2, [1, 2, 3, 4])
    conf = res.get("confidence", "low")
    print(f"   [Phase2] {reasoning2}")
    print(f"   [Phase2] 正面角度索引={aidx}, 置信={conf}")

    if aidx < 1 or aidx > 4:
        print(f"   [Phase2] 无效角度索引 {aidx}，VLM原始响应: {res}")
        return None

    return {"front_angle_index": aidx, "confidence": conf,
            "vlm_decision": res, "grid_path": grid_path}


def process_folder(folder: str, category: str, api_ref: list) -> str:
    try:
        step_name = next(f for f in os.listdir(folder)
                         if f.lower().endswith(('.step', '.stp')))
    except StopIteration:
        return "SKIP(no STEP)"

    step_path = os.path.join(folder, step_name)
    meta_path = os.path.join(folder, "alignment_meta.json")
    attempts_log = []

    with tempfile.TemporaryDirectory() as tmp:
        for attempt in range(1, 3):
            print(f"   ── 尝试 {attempt}/2 ──")

            p1 = _run_phase1(step_path, category, tmp, api_ref)
            if p1 is None:
                attempts_log.append({"attempt": attempt, "phase": 1, "result": None})
                time.sleep(SAFE_WAIT_TIME)
                continue

            p2 = _run_phase2(step_path, category, p1["upright_group"], tmp, api_ref)
            if p2 is None:
                attempts_log.append({"attempt": attempt, "phase": 1, "p1": p1, "p2": None})
                time.sleep(SAFE_WAIT_TIME)
                continue

            alignment = compute_alignment(p1["upright_group"], p2["front_angle_index"])
            attempts_log.append({"attempt": attempt, "p1": p1, "p2": p2, "alignment": alignment})

            meta = {
                "status":           "verified",
                "align_rotation":   alignment["align_rotation"],
                "extra_rz":         alignment["extra_rz"],
                "upright_group":    alignment["upright_group"],
                "front_angle_index":alignment["front_angle_index"],
                "p1_confidence":    p1["confidence"],
                "p2_confidence":    p2["confidence"],
                "p1_vlm_decision":  p1["vlm_decision"],
                "p2_vlm_decision":  p2["vlm_decision"],
                "category_hint":    category,
                "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

            flag = "⚠️ 建议复查" if p1["confidence"] == "low" or p2["confidence"] == "low" else "✅"
            return (f"{flag}  站立组=G{alignment['upright_group']}  "
                    f"正面角={alignment['extra_rz']:.0f}°  "
                    f"rot={alignment['align_rotation']}")

            time.sleep(SAFE_WAIT_TIME)

    # 两次均失败
    meta = {
        "status":        "manual_review",
        "category_hint": category,
        "attempts":      attempts_log,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    _append_summary(folder, category, "VLM判断失败（2次尝试）", attempts_log)
    return "MANUAL_REVIEW(两次判断均失败)"


# ══════════════════════════════════════════════════════════════ 汇总文件 ══════

def _load_summary() -> list:
    if os.path.exists(SUMMARY_PATH):
        try:
            return json.load(open(SUMMARY_PATH, "r", encoding="utf-8"))
        except Exception:
            pass
    return []

def _append_summary(folder: str, category: str, reason: str, attempts: list):
    summary = _load_summary()
    for entry in summary:
        if entry.get("folder") == folder:
            entry.update({"reason": reason, "attempts": attempts,
                          "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
            break
    else:
        summary.append({"folder": folder, "category": category,
                         "reason": reason, "attempts": attempts,
                         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════ 主循环 ══════

def main(start_from: int, end_at: int):
    target_dirs = []
    for root, _, files in os.walk(DATASET_ROOT):
        if any(f.lower().endswith(('.step', '.stp')) for f in files):
            target_dirs.append(root)
    total = len(target_dirs)
    print(f"🚀 发现 {total} 个模型目录，当前进程处理区间 [{start_from}, {end_at}]")

    # 筛选本次需要处理的目录
    pending = []
    for i, folder in enumerate(target_dirs, start=1):
        if i < start_from or i > end_at:
            continue
        meta_path = os.path.join(folder, "alignment_meta.json")
        if not OVERWRITE_ALL and os.path.exists(meta_path):
            try:
                existing = json.load(open(meta_path, "r", encoding="utf-8"))
                if existing.get("status") in ("verified", "manual_review"):
                    print(f"[{i}/{total}] ⏭️  已处理，跳过: {os.path.basename(folder)}")
                    continue
            except Exception:
                pass
        try:
            step_name = next(f for f in os.listdir(folder)
                             if f.lower().endswith(('.step', '.stp')))
        except StopIteration:
            continue
        pending.append((i, folder, os.path.join(folder, step_name),
                        os.path.basename(os.path.dirname(folder))))

    api_calls = 0

    for i, folder, step_path, category in pending:
        if api_calls >= TEST_LIMIT:
            print(f"\n🛑 已达 API 上限 ({TEST_LIMIT})，停止。")
            break

        meta_path = os.path.join(folder, "alignment_meta.json")
        label = os.path.basename(folder)
        print(f"\n[{i}/{total}] {label}  ({category})")

        with tempfile.TemporaryDirectory() as tmp:
            # 阶段一渲染
            p1_grid = render_phase1_grid(step_path, tmp)
            if p1_grid is None:
                print(f"   ❌ 阶段一渲染失败，跳过")
                _append_summary(folder, category, "阶段一渲染失败", [])
                continue

            # 阶段一 VLM
            print(f"   [Phase1] VLM 判断站立组 ({category})...")
            res1 = _query_vlm(p1_grid, _build_phase1_prompt(category))
            api_calls += 1

            if res1 is None:
                print(f"   ❌ Phase1 VLM 无响应（网络失败），跳过，不写 meta")
                continue

            reasoning1 = res1.get("reasoning", "")
            gid = _safe_int(res1.get("upright_group", 0))
            if gid < 1 or gid > 6:
                gid = _extract_int_from_reasoning(reasoning1, list(range(1, 7)))
            conf1 = res1.get("confidence", "low")
            print(f"   [Phase1] 站立组=G{gid}  置信={conf1}")

            if gid < 1 or gid > 6:
                print(f"   ❌ Phase1 无效组号，跳过，不写 meta")
                continue

            _, _, (rx, ry, rz) = ORIENTATION_GROUPS[gid - 1]

            # 阶段二渲染
            print(f"   [Phase2] 渲染4方向正视图...")
            p2_grid = render_phase2_grid(step_path, rx, ry, rz, tmp)

            if p2_grid is None:
                print(f"   ❌ 阶段二渲染失败，跳过，不写 meta")
                continue

            # 阶段二 VLM
            print(f"   [Phase2] VLM 判断正面朝向 ({category})...")
            res2 = _query_vlm(p2_grid, _build_phase2_prompt(category))
            api_calls += 1

            if res2 is None:
                print(f"   ❌ Phase2 VLM 无响应（网络失败），跳过，不写 meta")
                continue

            reasoning2 = res2.get("reasoning", "")
            aidx = _safe_int(res2.get("front_angle_index", 0))
            if aidx < 1 or aidx > 4:
                aidx = _extract_int_from_reasoning(reasoning2, [1, 2, 3, 4])
            conf2 = res2.get("confidence", "low")
            print(f"   [Phase2] 正面角度索引={aidx}  置信={conf2}")

            if aidx < 1 or aidx > 4:
                print(f"   ❌ Phase2 无效角度索引，跳过，不写 meta")
                continue

            alignment = compute_alignment(gid, aidx)
            meta = {
                "status":            "verified",
                "align_rotation":    alignment["align_rotation"],
                "extra_rz":          alignment["extra_rz"],
                "upright_group":     alignment["upright_group"],
                "front_angle_index": alignment["front_angle_index"],
                "p1_confidence":     conf1,
                "p2_confidence":     conf2,
                "p1_vlm_decision":   res1,
                "p2_vlm_decision":   res2,
                "category_hint":     category,
                "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

            flag = "⚠️" if conf1 == "low" or conf2 == "low" else "✅"
            print(f"   → {flag} G{gid}  extra_rz={alignment['extra_rz']:.0f}°  "
                  f"rot={alignment['align_rotation']}")
            time.sleep(SAFE_WAIT_TIME)

    print(f"\n🎉 进程运行结束。本次共调用 API {api_calls} 次。")
    if os.path.exists(SUMMARY_PATH):
        n = len(_load_summary())
        if n:
            print(f"📋 {n} 个模型需手工复查，见: {SUMMARY_PATH}")


if __name__ == "__main__":
    sf, ea = START_FROM, END_AT
    if len(sys.argv) > 1:
        try: sf = max(1, int(sys.argv[1]))
        except ValueError: pass
    if len(sys.argv) > 2:
        try: ea = int(sys.argv[2])
        except ValueError: pass
    main(sf, ea)
