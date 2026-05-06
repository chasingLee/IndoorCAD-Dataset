"""
render_scenes.py  —  GPU 加速室内场景多视角渲染
================================================================
架构说明
--------
瓶颈分析：
  • STEP→STL 网格化  ← CPU 密集（OCC C++ 内核）
  • STL→屏幕截图渲染 ← GPU（pyvista/VTK OpenGL，RTX 4090 自动使用）

优化策略：
  ┌─────────────────────────────────────────────────────┐
  │ CPU workers (ProcessPoolExecutor, N 个进程)          │
  │   scene_1.step → scene_1.stl ─────────────────────┐ │
  │   scene_2.step → scene_2.stl ───────────────────┐ │ │
  │   scene_3.step → scene_3.stl ─────────────────┐ │ │ │
  │                                                │ │ │ │
  │ GPU main thread (pyvista，复用同一 OpenGL 上下文)│ │ │ │
  │   render(scene_1) → render(scene_2) → ...  ←──┘ ┘ ┘ │
  └─────────────────────────────────────────────────────┘
  CPU 转换和 GPU 渲染流水线重叠，最大化两者利用率。

GPU 特性（RTX 4090）：
  • 8× MSAA 抗锯齿（硬件多重采样，GPU 执行）
  • SSAO 屏幕空间环境光遮蔽（GPU 后处理，增强纵深感）
  • 阴影（GPU shadow map）
  • PBR 材质（GPU 着色器）

注意：
  若渲染时发现使用的是集成显卡而非 4090，请在
  NVIDIA 控制面板 → 管理3D设置 → 程序设置 → python.exe
  → 选择"高性能 NVIDIA 处理器"并保存。

运行：
  python render_scenes.py
  python render_scenes.py --scene_dir scene_outputs --out_dir scene_renders --n_views 8 --workers 8
"""

import os

# Force NVIDIA GPU before VTK/pyvista initializes OpenGL context.
# Must be set before ANY import of vtk or pyvista.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')          # CUDA ops → GPU 0
os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW', 'vtkWin32OpenGLRenderWindow')  # Windows OpenGL
# Tell the NVIDIA driver to use the dGPU for this process (Optimus/hybrid graphics).
# This is the programmatic equivalent of the NVIDIA Control Panel "High-performance" setting.
os.environ.setdefault('NVAPI_OPTIMUS_ENABLEMENT', '1')

import json
import math
import argparse
import tempfile
import time

try:
    import numpy as np
    from PIL import Image, ImageEnhance
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# 顶层保护性导入：Windows multiprocessing spawn 会在子进程中重新执行本模块，
# 子进程只做 STEP→STL 转换，不需要 pyvista；若导入失败则跳过，不影响子进程。
try:
    import pyvista as pv
except Exception:
    pv = None  # type: ignore[assignment]

# ================== 配置区 ==================
SCENE_DIR   = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\test_p1_scenes"  # 输入 STEP 文件
OUTPUT_DIR  = "D:\\Lzm_Temp_Data\\Li_temp_project\\Scene_sy\\test_p1_renders"   # 输出 PNG 图像和相机参数 JSON
IMAGE_SIZE  = (1024, 1024)

# ── 人眼水平游走视角（原有功能）──────────────────────────────
HORIZONTAL_AZIMUTHS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
EYE_HEIGHT_MM    = 1600.0   # 相机离地高度 (mm)
EYE_OFFSET_RATIO = 0.25     # 相机偏离中心比例（避免死角正中）
LOOK_DOWN_ANGLE  = 10.0     # 俯视角（°），轻微向下营造自然感
HORIZONTAL_FOV   = 90.0     # 水平游走视角FOV (°)，更宽的视场（焦距更小）

# ── 全貌展示：八个角落透视图（扩展）─────────────────────────
# 摄像机置于房间八个方向，立体展示室内全貌，是室内设计/地产摄影的标准手法
RENDER_CORNERS      = True    # 是否渲染 8 个角落透视图
CORNER_AZIMUTHS     = [0, 45, 90, 135, 180, 225, 270, 315]  # 8个方向
CORNER_EYE_HEIGHT   = 1400.0  # 角落相机高度 (mm)，略低以拍到更多家具
CORNER_LOOK_H_RATIO = 0.30    # 焦点高度占房间高度的比例（约家具中部位置）
CORNER_MARGIN_RATIO = 0.12    # 相机距墙的比例间距（防止穿墙）
CORNER_FOV          = 105.0    # 广角 FOV (°)，捕获更大视野，焦距更小

# ── 全貌展示：45° 等角斜视图（新增）─────────────────────────
# 摄像机在角落高处俯视整个房间，兼顾布局和空间深度感
RENDER_ISO          = False    # 是否渲染等角斜视图
ISO_HEIGHT_RATIO    = 1.5     # 相机高度/房间高度倍数
ISO_FOV             = 65.0    # 等角视图 FOV (°)，更广的视角

# ── 俯视总览（原有功能）──────────────────────────────────────
RENDER_OVERVIEW  = False       # 是否渲染正射俯视图

MSAA_SAMPLES     = 16       # GPU MSAA 倍数（4/8/16），越高越清晰，耗 VRAM 更多
MESH_DEFLECTION  = 1.0     # STEP→STL 网格化精度（mm），越小越精细越慢
CPU_WORKERS      = min(max(1, (os.cpu_count() or 4) - 1), 6)  # 上限 6：STEP 文件大，太多进程会 OOM

# ── 白色简洁渲染配色（Clean White Aesthetic）────────────────────
TECH_BG_COLOR      = '#e8e8e8'   # 浅灰背景
TECH_FLOOR_COLOR   = '#c8c8c8'   # 地板：中灰，半透明感
TECH_CEILING_COLOR = '#dcdcdc'   # 天花板：浅灰
TECH_WALL_COLOR    = '#d0d0d0'   # 墙壁：浅灰，带透明质感
TECH_FURN_COLORS   = [           # 家具：纯白系
    '#ffffff',
    '#f5f5f5',
    '#efefef',
    '#f8f8f8',
    '#fafafa',
    '#f2f2f2',
    '#f0f0f0',
    '#f6f6f6',
]
TECH_WIRE_COLOR    = '#888888'   # 线框颜色（关闭）
TECH_WIRE_OPACITY  = 0.0         # 线框叠加透明度（0=关闭）
# ============================================


# ──────────────────────────────────────────
# CPU Worker（在子进程中运行，不能使用 lambda/闭包）
# ──────────────────────────────────────────

def _convert_step_worker(args: tuple) -> tuple:
    """子进程 worker：STEP → STL，返回 (step_path, stl_path_or_None, err_msg)。
    必须定义在模块顶层，Windows spawn 模式要求。
    """
    step_path, stl_path, deflection = args
    try:
        # 延迟导入（在子进程内执行，静态分析工具报 unresolved 属误报，运行时正常）
        from OCP.STEPControl import STEPControl_Reader  # noqa: PLC0415
        from OCP.IFSelect import IFSelect_RetDone       # noqa: PLC0415
        from OCP.StlAPI import StlAPI_Writer            # noqa: PLC0415
        from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: PLC0415

        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) != IFSelect_RetDone:
            return step_path, None, "STEP 读取失败"
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return step_path, None, "形状为空"
        BRepMesh_IncrementalMesh(shape, deflection)
        writer = StlAPI_Writer()
        writer.Write(shape, stl_path)
        if not os.path.exists(stl_path) or os.path.getsize(stl_path) == 0:
            return step_path, None, "STL 写入失败"
        return step_path, stl_path, None
    except Exception as exc:
        return step_path, None, str(exc)


# ──────────────────────────────────────────
# GPU 渲染（主进程，独占 OpenGL 上下文）
# ──────────────────────────────────────────

def _print_gpu_info():
    """打印 VTK 渲染后端信息，检测是否使用了正确的 GPU。"""
    try:
        import vtk
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.Initialize()
        info = rw.ReportCapabilities() or ""
        rw.Finalize()
        vendor_line = next((l for l in info.splitlines() if "vendor" in l.lower()), "")
        renderer_line = next((l for l in info.splitlines() if "renderer" in l.lower()), "")
        print(f"  GPU vendor  : {vendor_line.strip()}")
        print(f"  GPU renderer: {renderer_line.strip()}")
        if "intel" in (vendor_line + renderer_line).lower():
            print()
            print("  ⚠️  检测到使用 Intel 集显！RTX 4090 未被调用，渲染速度极慢。")
            print("  ── 修复方法（二选一）────────────────────────────────")
            print("  [方法1] NVIDIA 控制面板 → 管理3D设置 → 程序设置")
            print('          添加 python.exe → 选择"高性能 NVIDIA 处理器"→ 保存')
            print("  [方法2] 在运行脚本前设置环境变量：")
            print("          set CUDA_VISIBLE_DEVICES=0")
            print("          (使 Windows 优先向 NVIDA 驱动请求 OpenGL 上下文)")
            print("  ──────────────────────────────────────────────────")
            print()
    except Exception:
        print("  GPU 渲染后端: (无法查询，VTK 将使用系统默认 GPU)")


def _post_process(img_path: str, is_overview: bool = False):
    """轻度后处理：对比度微增 + 轻暗角，保持白色风格。"""
    if not _PIL_OK:
        return
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # 对比度略增，让阴影更有层次
        img = ImageEnhance.Contrast(img).enhance(1.15)

        if not is_overview:
            arr = np.array(img, dtype=np.float32)
            # 轻微暗角（仅压暗四角，不偏色）
            vy = np.linspace(-1.0, 1.0, h)[:, np.newaxis]
            vx = np.linspace(-1.0, 1.0, w)[np.newaxis, :]
            vignette = np.clip(1.0 - 0.25 * (vx ** 2 + vy ** 2), 0.0, 1.0)[:, :, np.newaxis]
            arr = arr * vignette
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        img.save(img_path, quality=93)
    except Exception as e:
        print(f"    ⚠️  后处理失败（跳过）: {e}")


def _add_tech_lights(plotter, cx, cy, room_z_min, H):
    """白色室内灯光：中性白主光 + 温白补光，适合白色场景。"""
    # 主光：正上方中性白，模拟顶灯
    plotter.add_light(pv.Light(
        position=(cx, cy, room_z_min + H * 1.2),
        focal_point=(cx, cy, room_z_min),
        intensity=1.4, color='#ffffff',
    ))
    # 侧光1：左前方温白，减少阴影
    plotter.add_light(pv.Light(
        position=(cx - H * 0.7, cy + H * 0.5, room_z_min + H * 0.9),
        focal_point=(cx, cy, room_z_min + H * 0.4),
        intensity=0.55, color='#fff8f0',
    ))
    # 侧光2：右后方，对称补光
    plotter.add_light(pv.Light(
        position=(cx + H * 0.7, cy - H * 0.5, room_z_min + H * 0.9),
        focal_point=(cx, cy, room_z_min + H * 0.4),
        intensity=0.45, color='#fff8f0',
    ))
    # 地面补光：消除家具底部死角
    plotter.add_light(pv.Light(
        position=(cx, cy, room_z_min + H * 0.05),
        focal_point=(cx, cy, room_z_min + H * 0.5),
        intensity=0.25, color='#ffffff',
    ))


def _split_and_color_scene(plotter, mesh, bounds):
    """
    将场景网格按连通体拆分，根据几何特征分类为地板/天花板/墙壁/家具，
    分别应用科技感配色。

    分类逻辑（基于各连通体的包围盒）：
      • 地板  : z_max ≤ z_floor_thresh 且 高度很薄 且 XY 面积大
      • 天花板: z_min ≥ z_ceil_thresh  且 高度很薄 且 XY 面积大
      • 墙壁  : 高度 > 70% 房间高度 且 某一水平方向极薄
      • 家具  : 其余所有连通体，循环使用 TECH_FURN_COLORS
    """
    z_min, z_max = bounds[4], bounds[5]
    total_H = z_max - z_min          # 包含地板板/天花板板厚度的总 Z 跨度
    x_span  = bounds[1] - bounds[0]
    y_span  = bounds[3] - bounds[2]

    # 阈值：板厚通常 200mm ≈ total_H * 0.06（房间 3000+400mm）
    SLAB_THIN  = total_H * 0.12      # 地板/天花板判定最大厚度
    SLAB_AREA  = 0.50                # XY 面积覆盖率阈值（必须 > 50% 房间平面）
    WALL_H_MIN = total_H * 0.65      # 墙壁最小高度比例
    WALL_THIN  = 0.22                # 墙壁薄侧与对应房间尺寸之比

    # 共用 Phong 材质参数（高 specular 模拟金属/光滑表面）
    mesh_kw = dict(smooth_shading=True,
                   ambient=0.30, diffuse=0.55, specular=0.60, specular_power=90)

    try:
        bodies = mesh.split_bodies()
        n_bodies = bodies.n_blocks
    except Exception:
        # split_bodies 失败则整体用家具色渲染
        plotter.add_mesh(mesh, color=TECH_FURN_COLORS[0], **mesh_kw)
        return

    furn_idx = 0
    for i in range(n_bodies):
        body = bodies[i]
        if body is None or body.n_points == 0:
            continue

        b = body.bounds           # (xmin, xmax, ymin, ymax, zmin, zmax)
        bz_min, bz_max = b[4], b[5]
        b_h   = bz_max - bz_min
        bx_sp = b[1] - b[0]
        by_sp = b[3] - b[2]

        # 地板判定
        if (bz_max <= z_min + total_H * 0.10
                and b_h < SLAB_THIN
                and bx_sp > x_span * SLAB_AREA):
            color = TECH_FLOOR_COLOR

        # 天花板判定
        elif (bz_min >= z_max - total_H * 0.10
                and b_h < SLAB_THIN
                and bx_sp > x_span * SLAB_AREA):
            color = TECH_CEILING_COLOR

        # 墙壁判定（高度接近房间高度，且在某一水平轴上很薄）
        elif (b_h > WALL_H_MIN
                and (bx_sp < x_span * WALL_THIN
                     or by_sp < y_span * WALL_THIN)):
            color = TECH_WALL_COLOR

        # 家具（其余）
        else:
            color = TECH_FURN_COLORS[furn_idx % len(TECH_FURN_COLORS)]
            furn_idx += 1

        plotter.add_mesh(body, color=color, **mesh_kw)

        # 家具叠加半透明线框，营造全息/HUD 科技感
        if color not in (TECH_FLOOR_COLOR, TECH_CEILING_COLOR, TECH_WALL_COLOR):
            if TECH_WIRE_OPACITY > 0.01:
                plotter.add_mesh(body, style='wireframe',
                                 color=TECH_WIRE_COLOR, opacity=TECH_WIRE_OPACITY,
                                 line_width=0.5)


def _compute_camera_params(
    cam_pos, focal_point, up, fov_deg: float, img_size: tuple, filename: str
) -> dict:
    """
    根据 pyvista 相机参数计算并返回完整的相机信息字典。

    坐标约定（OpenCV）：
      • X 轴 = 相机右方
      • Y 轴 = 相机下方（OpenCV 惯例，Y 朝下）
      • Z 轴 = 相机朝向场景方向（正 Z 指向焦点）

    字段说明：
      camera_position_mm  : 相机在世界坐标系中的位置（mm，与场景同单位）
      focal_point_mm      : 相机注视点的世界坐标（mm）
      up_vector           : 相机"上"方向单位向量（世界坐标系）
      fov_vertical_deg    : 垂直视场角（°）
      focal_length_px     : 等效焦距（像素），由 FOV 和图像高度推导
      K                   : 3×3 相机内参矩阵（像素坐标，cx=W/2, cy=H/2）
      c2w_opencv          : 4×4 相机到世界变换矩阵（OpenCV 轴约定）
      w2c_opencv          : 4×4 世界到相机变换矩阵（c2w 的逆，用于投影）
    """
    W, H = img_size
    p = np.array(cam_pos,    dtype=np.float64)
    f = np.array(focal_point, dtype=np.float64)
    u = np.array(up,          dtype=np.float64)

    # ── 内参 ────────────────────────────────────────────────
    fov_rad = math.radians(fov_deg)
    fy = (H / 2.0) / math.tan(fov_rad / 2.0)   # 由垂直 FOV 推导
    fx = fy                                       # pyvista 正方形像素
    K = [
        [fx,  0.0, W / 2.0],
        [0.0, fy,  H / 2.0],
        [0.0, 0.0, 1.0    ],
    ]

    # ── 外参：构建相机轴 ─────────────────────────────────────
    fwd = f - p
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-9:
        fwd = np.array([0.0, 1.0, 0.0])
    else:
        fwd /= fwd_norm

    right = np.cross(fwd, u)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        # up 与 forward 平行（俯视/仰视时），取默认右方向
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_norm

    cam_up = np.cross(right, fwd)   # 重新正交化 up
    cam_up /= np.linalg.norm(cam_up)

    # c2w：列向量分别为 right、-cam_up（OpenCV Y 朝下）、fwd、位置
    c2w = np.eye(4)
    c2w[:3, 0] =  right
    c2w[:3, 1] = -cam_up    # OpenCV 约定：Y 轴向下
    c2w[:3, 2] =  fwd
    c2w[:3, 3] =  p

    # w2c = c2w 的逆（旋转转置 + 平移变换）
    R = c2w[:3, :3]
    t = p
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3,  3] = -(R.T @ t)

    def _m(m): return [[round(v, 6) for v in row] for row in m.tolist()]

    return {
        "filename":            os.path.basename(filename),
        "camera_position_mm":  [round(v, 3) for v in p.tolist()],
        "focal_point_mm":      [round(v, 3) for v in f.tolist()],
        "up_vector":           [round(v, 6) for v in u.tolist()],
        "fov_vertical_deg":    round(fov_deg, 4),
        "focal_length_px":     round(fy, 4),
        "K":                   _m(np.array(K)),
        "c2w_opencv":          _m(c2w),
        "w2c_opencv":          _m(w2c),
    }


def _render_stl(stl_path: str, out_dir: str, base: str,
                azimuths, eye_height, look_down, overview, img_size,
                corners=True, iso=True):
    """每个场景独立创建 Plotter（避免 clear() 残留状态导致空白帧）。
    corners: 渲染 4 个角落透视图（展示室内全貌）
    iso:     渲染 1 个 45° 等角斜视图（鸟瞰布局）
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        mesh = pv.read(stl_path)
    except Exception as e:
        print(f"  ❌ STL 加载失败: {e}")
        return False

    if mesh.n_points == 0:
        print(f"  ❌ 网格为空（0个顶点）")
        return False

    bounds = mesh.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    # 房间高度：取 Z 方向范围（不依赖 zmax，防止家具超高导致偏差）
    room_z_min = bounds[4]
    room_z_max = bounds[5]
    H = room_z_max - room_z_min

    half_L = (bounds[1] - bounds[0]) / 2
    half_W = (bounds[3] - bounds[2]) / 2

    # 相机在房间内，偏离中心 1/4 处，高度 = 地板 + 人眼高度
    cam_x = cx - half_L * EYE_OFFSET_RATIO
    cam_y = cy - half_W * EYE_OFFSET_RATIO
    cam_z = room_z_min + min(eye_height, H * 0.6)  # 防止超过房间高度

    look_dist    = max(half_L, half_W) * 0.8
    look_down_dz = look_dist * math.tan(math.radians(look_down))

    # 视锥近裁剪面：设为 1mm（场景单位 mm），远裁剪面：房间对角线的 3 倍
    diag = math.sqrt((2*half_L)**2 + (2*half_W)**2 + H**2)
    clip_near = 1.0
    clip_far  = diag * 3.0

    # 每个场景独立创建 Plotter，避免复用时残留 GPU 状态
    pv.global_theme.multi_samples = MSAA_SAMPLES
    pl = pv.Plotter(off_screen=True, window_size=list(img_size))
    try:
        pl.enable_anti_aliasing('msaa')
    except Exception:
        pass
    # 边缘增强：Eye Dome Lighting 让墙角/家具轮廓更清晰，不需要特殊帧缓冲
    try:
        pl.enable_eye_dome_lighting()
    except Exception:
        pass

    # 科技感配色：按连通体分类（地板/天花板/墙壁/家具），分别着色
    pl.set_background(TECH_BG_COLOR)
    _split_and_color_scene(pl, mesh, bounds)
    _add_tech_lights(pl, cx, cy, room_z_min, H)

    # 广角镜头（FOV 由配置决定），接近室内实景拍摄手感
    pl.camera.view_angle = HORIZONTAL_FOV
    camera_records: list[dict] = []   # 收集本场景所有视角的相机参数

    def _shoot(cam_pos, focal_point, up, filename, is_ov=False, fov=None):
        effective_fov = fov if fov is not None else (65.0 if is_ov else HORIZONTAL_FOV)
        pl.camera_position = [cam_pos, focal_point, up]
        pl.camera.view_angle = effective_fov
        pl.camera.clipping_range = (clip_near, clip_far if not is_ov else clip_far * 2)
        pl.render()
        pl.screenshot(filename)
        _post_process(filename, is_overview=is_ov)
        # 记录相机参数（仅当 numpy 可用时）
        if _PIL_OK:   # _PIL_OK 表示 numpy 也已导入
            camera_records.append(
                _compute_camera_params(cam_pos, focal_point, up,
                                       effective_fov, img_size, filename)
            )

    # ── 水平游走视角（原有功能）──────────────────────────────────
    for az in azimuths:
        az_rad = math.radians(az)
        look_x = cam_x + look_dist * math.cos(az_rad)
        look_y = cam_y + look_dist * math.sin(az_rad)
        look_z = cam_z - look_down_dz
        _shoot(
            (cam_x, cam_y, cam_z),
            (look_x, look_y, look_z),
            (0, 0, 1),
            os.path.join(out_dir, f"{base}_az{az:03d}.png"),
        )

    # ── 天花板四角俯视（相机在房间内侧角落天花板处，朝向室内中心）────
    if corners:
        # 相机在房间 XY 边界内侧 10% 处，贴近天花板，向下看室内
        inset = 0.10   # 距墙壁内侧比例，防止穿墙
        corner_eye_z = room_z_min + H * 0.92   # 接近天花板
        look_z_c = room_z_min + H * 0.25       # 俯视焦点（约地面以上 1/4）

        corners_def = [
            ("NE",  bounds[0] + (bounds[1]-bounds[0])*inset,  bounds[2] + (bounds[3]-bounds[2])*(1-inset)),
            ("NW",  bounds[0] + (bounds[1]-bounds[0])*(1-inset), bounds[2] + (bounds[3]-bounds[2])*(1-inset)),
            ("SW",  bounds[0] + (bounds[1]-bounds[0])*(1-inset), bounds[2] + (bounds[3]-bounds[2])*inset),
            ("SE",  bounds[0] + (bounds[1]-bounds[0])*inset,  bounds[2] + (bounds[3]-bounds[2])*inset),
        ]
        for label, cam_x_c, cam_y_c in corners_def:
            _shoot(
                (cam_x_c, cam_y_c, corner_eye_z),
                (cx, cy, look_z_c),
                (0, 0, 1),
                os.path.join(out_dir, f"{base}_corner_{label}.png"),
                fov=CORNER_FOV,
            )

    # ── 45° 等角斜视图（鸟瞰布局，兼顾深度感）───────────────────
    # 摄像机在 SW 角高处斜视房间全貌；不使用正射投影，保留透视深度。
    if iso:
        margin = max(half_L, half_W) * CORNER_MARGIN_RATIO
        iso_cam = (
            bounds[0] + margin,              # X: 靠近 -X 侧墙
            bounds[2] + margin,              # Y: 靠近 -Y 侧墙
            room_z_min + H * ISO_HEIGHT_RATIO,
        )
        iso_focal = (
            cx + half_L * 0.15,             # 焦点略偏向房间内侧
            cy + half_W * 0.15,
            room_z_min + H * 0.15,          # 焦点在家具中部偏下
        )
        pl.disable_parallel_projection()
        _shoot(
            iso_cam, iso_focal, (0, 0, 1),
            os.path.join(out_dir, f"{base}_iso45.png"),
            fov=ISO_FOV,
        )

    # ── 正射俯视图（平面布局参考）────────────────────────────────
    if overview:
        pl.disable_parallel_projection()
        pl.enable_parallel_projection()
        _shoot(
            (cx, cy, room_z_min + H * 2.5),
            (cx, cy, room_z_min),
            (0, 1, 0),
            os.path.join(out_dir, f"{base}_overview.png"),
            is_ov=True,
        )

    pl.close()

    # ── 保存相机参数 JSON ─────────────────────────────────────────
    if camera_records:
        cam_json = {
            "scene":        base,
            "image_size_wh": list(img_size),   # [width, height] px
            "unit":         "mm",               # 与场景 STEP 文件单位一致
            "convention":   "OpenCV",           # X=right, Y=down, Z=into_scene
            "views":        camera_records,
        }
        cam_path = os.path.join(out_dir, f"{base}_cameras.json")
        with open(cam_path, 'w', encoding='utf-8') as jf:
            json.dump(cam_json, jf, indent=2, ensure_ascii=False)

    return True


# ──────────────────────────────────────────
# 主流水线
# ──────────────────────────────────────────

def _scene_is_done(out_dir: str, base: str, azimuths, corners: bool,
                   iso: bool, overview: bool) -> bool:
    """判断某个场景是否已完整渲染（输出目录存在且 PNG 数量达到预期）。"""
    if not os.path.isdir(out_dir):
        return False
    expected = len(azimuths)
    if corners:
        expected += 4   # NE/NW/SW/SE
    if iso:
        expected += 1
    if overview:
        expected += 1
    png_count = sum(1 for f in os.listdir(out_dir)
                    if f.startswith(base) and f.endswith('.png'))
    return png_count >= expected


def render_all_scenes(scene_dir=SCENE_DIR, output_dir=OUTPUT_DIR,
                      azimuths=HORIZONTAL_AZIMUTHS,
                      eye_height=EYE_HEIGHT_MM,
                      look_down=LOOK_DOWN_ANGLE,
                      overview=RENDER_OVERVIEW,
                      corners=RENDER_CORNERS,
                      iso=RENDER_ISO,
                      img_size=IMAGE_SIZE,
                      resume=False,
                      start=0, end=None):
    """
    start: 起始场景索引（含，0-based，按文件名排序）
    end:   终止场景索引（不含），None 表示到末尾
    多进程方式：手动在多个终端分别指定不同的 --start --end 范围并行启动。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_step_files = sorted([
        os.path.join(scene_dir, f)
        for f in os.listdir(scene_dir)
        if f.lower().endswith(('.step', '.stp'))
    ])
    if not all_step_files:
        print(f"⚠️  在 {scene_dir} 中未找到 STEP 文件")
        return

    total_all = len(all_step_files)

    # 按 start/end 切片
    end_idx = end if end is not None else total_all
    end_idx = min(end_idx, total_all)
    step_files = all_step_files[start:end_idx]
    if not step_files:
        print(f"⚠️  start={start} end={end_idx} 范围内无文件（总数 {total_all}）")
        return

    print(f"\n{'='*55}")
    print(f"  全部场景: {total_all}  本进程范围: [{start}, {end_idx})  共 {len(step_files)} 个")
    print(f"  MSAA: {MSAA_SAMPLES}×")
    _print_gpu_info()
    print(f"{'='*55}\n")

    # 断点续跑：跳过已完整渲染的场景
    skipped = 0
    if resume:
        before = len(step_files)
        step_files = [
            sp for sp in step_files
            if not _scene_is_done(
                os.path.join(output_dir, os.path.splitext(os.path.basename(sp))[0]),
                os.path.splitext(os.path.basename(sp))[0],
                azimuths, corners, iso, overview,
            )
        ]
        skipped = before - len(step_files)
        print(f"  [续跑] 已完成 {skipped} 个，剩余 {len(step_files)} 个待渲染\n")

    if not step_files:
        print("  本范围内所有场景均已完成，无需重新渲染。")
        return

    n_imgs_per_scene = (len(azimuths)
                        + (4 if corners else 0)
                        + (1 if iso else 0)
                        + (1 if overview else 0))

    ok, fail = 0, 0
    t0 = time.time()
    total_pending = len(step_files)

    for i, step_path in enumerate(step_files):
        base = os.path.splitext(os.path.basename(step_path))[0]
        out_dir = os.path.join(output_dir, base)
        print(f"[{start + i + 1 + skipped}/{end_idx}] {base}  (本进程 {i+1}/{total_pending})")

        # STEP → STL（当前进程内串行执行，简单可靠）
        stl_fd, stl_path = tempfile.mkstemp(suffix='.stl', prefix=f'{base}_')
        os.close(stl_fd)
        t_conv = time.time()
        _, stl_out, err = _convert_step_worker((step_path, stl_path, MESH_DEFLECTION))
        print(f"  网格化 {time.time()-t_conv:.1f}s", end='  ')

        if err or stl_out is None:
            print(f"❌ 转换失败: {err}")
            fail += 1
            try:
                os.remove(stl_path)
            except OSError:
                pass
            continue

        # STL → PNG（GPU 渲染）
        t_render = time.time()
        success = _render_stl(stl_path, out_dir, base,
                              azimuths, eye_height, look_down, overview, img_size,
                              corners=corners, iso=iso)
        dt = time.time() - t_render

        if success:
            ok += 1
            print(f"渲染 {dt:.1f}s  ✅ {n_imgs_per_scene} 张  → {out_dir}")
        else:
            fail += 1
            print(f"渲染 {dt:.1f}s  ❌ 失败")

        try:
            os.remove(stl_path)
        except OSError:
            pass

    total_time = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  完成: 成功 {ok} / 失败 {fail} / 跳过 {skipped} / 本进程共 {len(step_files)+skipped}  总耗时 {total_time:.1f}s")
    print(f"  平均每场景: {total_time/max(ok,1):.1f}s  输出: {output_dir}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GPU 加速室内场景渲染',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
多进程并行示例（在多个终端分别运行）：
  # 终端1：处理第 0~999 个场景
  python render_scenes.py --scene_dir p1_scene_output --out_dir p1_scene_renders --start 0 --end 1000

  # 终端2：处理第 1000~1999 个场景
  python render_scenes.py --scene_dir p1_scene_output --out_dir p1_scene_renders --start 1000 --end 2000

  # 终端3：处理第 2000 个之后的全部场景
  python render_scenes.py --scene_dir p1_scene_output --out_dir p1_scene_renders --start 2000

续跑（跳过已完成场景）：
  python render_scenes.py --scene_dir p1_scene_output --out_dir p1_scene_renders --start 0 --end 1000 --resume
""",
    )
    parser.add_argument('--scene_dir', default=SCENE_DIR,
                        help=f'STEP 场景目录（默认: {SCENE_DIR}）')
    parser.add_argument('--out_dir',   default=OUTPUT_DIR,
                        help=f'渲染输出目录（默认: {OUTPUT_DIR}）')
    parser.add_argument('--start',     type=int, default=0,
                        help='起始场景索引（含，0-based，按文件名排序，默认 0）')
    parser.add_argument('--end',       type=int, default=None,
                        help='终止场景索引（不含），默认到末尾')
    parser.add_argument('--n_views',   type=int, default=len(HORIZONTAL_AZIMUTHS),
                        help='水平游走视角数量（均匀分布，0=跳过）')
    parser.add_argument('--img_size',  type=int, default=1024,
                        help='图像分辨率（正方形边长，默认1024）')
    parser.add_argument('--no-corners', dest='corners', action='store_false',
                        help='跳过角落透视图（默认开启）')
    parser.add_argument('--no-iso',     dest='iso',     action='store_false',
                        help='跳过 45° 等角斜视图（默认开启）')
    parser.add_argument('--no-overview',dest='overview',action='store_false',
                        help='跳过正射俯视图（默认开启）')
    parser.add_argument('--resume',     action='store_true',
                        help='续跑：跳过已完整渲染的场景')
    parser.set_defaults(corners=RENDER_CORNERS, iso=RENDER_ISO, overview=RENDER_OVERVIEW)
    args = parser.parse_args()

    azimuths = [int(360 * i / args.n_views) for i in range(args.n_views)] if args.n_views > 0 else []
    render_all_scenes(
        scene_dir=args.scene_dir,
        output_dir=args.out_dir,
        azimuths=azimuths,
        corners=args.corners,
        iso=args.iso,
        overview=args.overview,
        img_size=(args.img_size, args.img_size),
        resume=args.resume,
        start=args.start,
        end=args.end,
    )
