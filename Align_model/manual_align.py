"""
manual_align.py — 手动校准需要 manual_review 的模型

流程：
  1. 读取 manual_review_summary.json，收集所有待手动校准的模型
  2. 对每个模型：复用 Align_model.py 的渲染逻辑生成 8 视角网格图并自动打开
  3. 用户输入 top_view_id / front_view_id（可选 flip_x）
  4. 计算对齐旋转，保存 alignment_meta.json（status="verified"）

视角编号参照（与 Align_model.py 完全一致）：
  ①(+X面)  ②(-X面)  ③(-Y面)  ④(+Y面)  ⑤(+Z俯视)  ⑥(-Z仰视)
  对立对：①↔②  ③↔④  ⑤↔⑥

用法：
  python manual_align.py                     # 从头开始
  python manual_align.py --start 10          # 从第10个未校准模型开始
  python manual_align.py --folder "D:/..."   # 只处理指定文件夹
  python manual_align.py --list              # 列出所有待校准模型

输入格式（每个模型提示时）：
  5 3          → top=5, front=3, flip=False
  1 3 flip     → top=1, front=3, flip=True
  s / skip     → 跳过（保留 manual_review 状态）
  q / quit     → 退出程序
  r / render   → 重新渲染并显示图像
  show         → 再次打开图像
"""

import os, sys, json, math, time, tempfile, subprocess, platform, argparse
import numpy as np
from scipy.spatial.transform import Rotation as SciR

# ── 复用 Align_model.py 的渲染函数 ──────────────────────────────────────────
_ALIGN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ALIGN_DIR)

# Align_model.py 在项目根目录，不在 Scene_sy 子目录
_PROJ_ROOT = os.path.join(_ALIGN_DIR)
sys.path.insert(0, _PROJ_ROOT)

import importlib.util as _ilu

def _import_align_model():
    spec = _ilu.spec_from_file_location(
        "Align_model",
        os.path.join(_PROJ_ROOT, "Align_model.py"),
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_am = _import_align_model()

# 直接引用 Align_model.py 里经过验证的函数
render_phase1_grid = _am.render_phase1_grid   # (stl_path, tmp_dir) -> path|None
load_center_stl    = _am.load_center_stl      # (step_path, tmp_dir) -> path|None
calculate_rot      = _am.calculate_rot
compute_forward_angle = _am.compute_forward_angle
validate_ids       = _am.validate_ids

# ══════════════════════════════════════════════════════════════ 配置区 ══════
DATASET_ROOT = r"D:\Lzm_Temp_Data\Li_temp_project\Dataset_Categoried\1_1000"
SUMMARY_PATH = os.path.join(DATASET_ROOT, "manual_review_summary.json")
# ═══════════════════════════════════════════════════════════════════════════


def find_stl_file(folder: str) -> str | None:
    """优先返回现成的 _MESH.STL，避免重新从 STEP 转换。"""
    for f in os.listdir(folder):
        if f.lower().endswith('_mesh.stl'):
            return os.path.join(folder, f)
    for f in os.listdir(folder):
        if f.lower().endswith('.stl'):
            return os.path.join(folder, f)
    return None


def find_step_file(folder: str) -> str | None:
    for f in os.listdir(folder):
        if f.lower().endswith(('.step', '.stp')):
            return os.path.join(folder, f)
    return None


def open_image(path: str):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"   ⚠️ 无法自动打开图片: {e}")
        print(f"   请手动打开: {path}")


def save_meta(folder: str, category: str,
              top_id: int, front_id: int, flip_x: bool,
              align_rot: list, forward_angle: float):
    meta = {
        "status": "verified",
        "align_rotation": align_rot,
        "flip_correction": flip_x,
        "confidence": "manual",
        "vlm_decision": {
            "reasoning": "手动校准",
            "top_view_id":    top_id,
            "front_view_id":  front_id,
            "is_upside_down": flip_x,
            "confidence":     "manual",
        },
        "verify_result": {
            "is_upright":    True,
            "front_faces_x": True,
            "correct":       True,
            "issues":        None,
        },
        "category_hint":    category,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "manual_corrected": True,
    }
    path = os.path.join(folder, "alignment_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"   ✅ 已保存: {path}")


# ══════════════════════════════════════════════════════════════ 交互 ══════

VIEW_LEGEND = """\
  视角编号对照表（与 VLM 看到的图完全一致）：
    ① (+X面)   ② (-X面)   ③ (-Y面)
    ④ (+Y面)   ⑤ (+Z俯视) ⑥ (-Z仰视)
  对立对 (不可同时选): ①↔②  ③↔④  ⑤↔⑥

  输入格式:
    <top> <front>         例: 5 3
    <top> <front> flip    例: 1 3 flip   (模型上下颠倒时加 flip)
    s / skip              跳过（保留 manual_review 状态）
    r / render            重新渲染并打开图像
    show                  重新打开图像
    q / quit              退出"""


def parse_input(line: str):
    line = line.strip().lower()
    if line in ("q", "quit"):
        return "quit"
    if line in ("s", "skip"):
        return "skip"
    if line in ("r", "render"):
        return "render"
    if line == "show":
        return "show"
    parts = line.split()
    if len(parts) >= 2:
        try:
            top_id   = int(parts[0])
            front_id = int(parts[1])
            flip_x   = len(parts) >= 3 and parts[2] in ("flip", "true", "1")
            return (top_id, front_id, flip_x)
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════ 主流程 ══════

def collect_pending(only_folder: str | None = None) -> list[dict]:
    if not os.path.exists(SUMMARY_PATH):
        print(f"❌ 找不到: {SUMMARY_PATH}")
        return []
    entries = json.load(open(SUMMARY_PATH, "r", encoding="utf-8"))
    pending = []
    for e in entries:
        folder = e.get("folder", "").strip()
        if not folder:
            continue
        if only_folder and os.path.abspath(folder) != os.path.abspath(only_folder):
            continue
        meta_path = os.path.join(folder, "alignment_meta.json")
        if os.path.exists(meta_path):
            try:
                m = json.load(open(meta_path, "r", encoding="utf-8"))
                if m.get("status") == "verified" and m.get("manual_corrected"):
                    continue
            except Exception:
                pass
        pending.append(e)
    return pending


def process_one(entry: dict, tmp_dir: str) -> str:
    folder   = entry.get("folder", "").strip()
    category = entry.get("category", "?")
    label    = os.path.basename(folder)

    print(f"\n{'='*70}")
    print(f"  模型: {label}  (类别: {category})")
    print(f"  路径: {folder}")

    for a in entry.get("attempts", []):
        issues = a.get("phase2", {}).get("issues")
        if issues:
            print(f"  VLM失败: {issues}")

    # 找 STL（优先现成的 _MESH.STL，省去 STEP→STL 转换时间）
    stl_path = find_stl_file(folder)
    if stl_path is None:
        # 回退：从 STEP 实时转换
        step_path = find_step_file(folder)
        if step_path is None:
            print("  ❌ 找不到 STL 也找不到 STEP，跳过")
            return "skipped"
        print("  ⏳ 未找到 STL，从 STEP 转换中...")
        stl_path = load_center_stl(step_path, tmp_dir)
        if stl_path is None:
            print("  ❌ STEP 转换失败，跳过")
            return "skipped"

    # 渲染输出路径（固定文件名，重渲时覆盖）
    grid_path = os.path.join(tmp_dir, "grid.jpg")

    def do_render() -> bool:
        print("  🔄 渲染中，请稍等...")
        result = render_phase1_grid(stl_path, tmp_dir)
        # Align_model.render_phase1_grid 把结果写到 tmp_dir/phase1_grid.jpg
        src = os.path.join(tmp_dir, "phase1_grid.jpg")
        if result and os.path.exists(src):
            import shutil
            shutil.copy2(src, grid_path)
            print(f"  📸 图像: {grid_path}")
            open_image(grid_path)
            return True
        print("  ❌ 渲染失败")
        return False

    do_render()
    print(VIEW_LEGEND)

    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit"

        result = parse_input(line)
        if result is None:
            print("  ⚠️ 格式错误，例: 5 3  或  1 3 flip")
            continue
        if result == "quit":
            return "quit"
        if result == "skip":
            print("  ⏭️  已跳过")
            return "skipped"
        if result == "render":
            do_render()
            continue
        if result == "show":
            open_image(grid_path)
            continue

        top_id, front_id, flip_x = result
        if not validate_ids(top_id, front_id):
            print(f"  ❌ 非法组合 top={top_id} front={front_id}：范围需 [1-6]，不相同，不互为对立对")
            continue

        rot = calculate_rot(top_id, front_id)
        if rot is None:
            print("  ❌ 旋转解算失败")
            continue

        fwd = compute_forward_angle(rot, front_id, flip_x)
        print(f"  📐 align_rotation={rot}  forward_angle={fwd:.1f}°  flip_x={flip_x}")

        confirm = input("  确认保存？(y/n) > ").strip().lower()
        if confirm in ("y", "yes", ""):
            save_meta(folder, category, top_id, front_id, flip_x, rot, fwd)
            return "saved"
        else:
            print("  取消，请重新输入")


def main():
    parser = argparse.ArgumentParser(
        description="手动校准 manual_review 模型的朝向",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", type=int, default=1,
                        help="从第 N 个未校准模型开始（默认 1）")
    parser.add_argument("--folder", default=None,
                        help="只处理指定文件夹（绝对路径）")
    parser.add_argument("--list", action="store_true",
                        help="只列出所有待校准模型，不进入交互")
    args = parser.parse_args()

    pending = collect_pending(only_folder=args.folder)
    if not pending:
        print("✅ 没有待手动校准的模型！")
        return

    if args.list:
        print(f"\n待校准模型（共 {len(pending)} 个）:")
        for i, e in enumerate(pending, 1):
            print(f"  [{i:3d}] {e.get('category','?'):25s}  {os.path.basename(e.get('folder',''))}")
        return

    pending = pending[max(0, args.start - 1):]
    total   = len(pending)

    print(f"\n{'='*70}")
    print(f"  手动校准工具  —  共 {total} 个模型待处理（从第 {args.start} 个开始）")
    print(f"{'='*70}")

    saved = skipped = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, entry in enumerate(pending, 1):
            print(f"\n[{i}/{total}]", end="")
            result = process_one(entry, tmp_dir)
            if result == "saved":
                saved += 1
            elif result == "skipped":
                skipped += 1
            elif result == "quit":
                break

    print(f"\n{'='*70}")
    print(f"  完成：保存={saved}  跳过={skipped}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
