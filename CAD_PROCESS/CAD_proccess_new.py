import os
import re
import json
import time
import psutil
import win32com.client
import pythoncom

# --- SolidWorks API 常量 ---
swDocPART = 1
swDocASSEMBLY = 2
swSaveAsOptions_Silent = 1
swSaveAsCurrentVersion = 0
FILE_SIZE_LIMIT = 100 * 1024 * 1024  # 100MB 阈值

SURFACE_IDENTITY_MAP = {

    0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere",
    4: "Torus", 5: "BSurf (BSpline)", 6: "Bezier", 7: "Other",
}

class SolidWorksProcessor:
    def __init__(self, process_limit=15):
        self.swApp = None
        self.process_limit = process_limit
        self.processed_count = 0
        self.start_sw()

    def start_sw(self):
        print("\n正在启动/连接 SolidWorks 2025...")
        try:
            pythoncom.CoInitialize()
            self.swApp = win32com.client.DispatchEx("SldWorks.Application")
            self.swApp.Visible = True
            self.swApp.UserControl = True
            self.swApp.SetUserPreferenceToggle(73, False)
        except Exception as e:
            print(f"启动失败: {e}")

    def restart_sw(self):
        print(f"\n[内存清理] 重启 SolidWorks 以释放资源...")
        if self.swApp:
            try: self.swApp.ExitApp()
            except: pass
        self.swApp = None
        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] and "SLDWORKS" in proc.info["name"].upper():
                    proc.kill()
            except: pass
        time.sleep(5)
        pythoncom.CoUninitialize()
        self.start_sw()
        self.processed_count = 0

    def sanitize_name(self, name):
        return re.sub(r'[\\/*?:"<>|]', "", name).strip()

    def is_task_complete(self, out_folder, base_name):
        """
        断点续传检测：检查核心 B-Rep 文件 (.x_t) 是否已经生成。
        文件名格式需与 export_formats 函数中定义的保持一致。
        """
        marker_file = os.path.join(out_folder, f"{base_name}_ORIGINAL_BREP.x_t")
        return os.path.exists(marker_file)

    def safe_save(self, model, path):
        errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        res = model.SaveAs4(path, swSaveAsCurrentVersion, swSaveAsOptions_Silent, errors, warnings)
        return res

    def process_dataset(self, input_dir, output_dir):
        total_skipped = 0
        total_processed = 0

        for root, dirs, files in os.walk(input_dir):
            if not files: continue

            valid_files = [f for f in files if not f.startswith("~$")]
            asms = [f for f in valid_files if f.lower().endswith(".sldasm")]
            prts = [f for f in valid_files if f.lower().endswith(".sldprt")]
            steps = [f for f in valid_files if f.lower().endswith((".step", ".stp"))]

            targets = []
            category = ""
            if asms: targets, category = asms, "ASM"
            elif prts: targets, category = prts, "PRT"
            elif steps: targets, category = steps, "STEP"
            else: continue

            for f in targets:
                file_path = os.path.join(root, f)
                if os.path.getsize(file_path) > FILE_SIZE_LIMIT:
                    continue

                # 构造独立的输出路径
                file_pure_name = os.path.splitext(f)[0]
                ext_name = os.path.splitext(f)[1][1:].upper()
                safe_folder_name = self.sanitize_name(f"{file_pure_name}_{ext_name}")
                out_folder = os.path.join(output_dir, safe_folder_name)

                # --- 断点续传逻辑：检查 .x_t 文件 ---
                if self.is_task_complete(out_folder, file_pure_name):
                    total_skipped += 1
                    # 打印当前跳过进度
                    print(f"[跳过] 核心文件已存在: {f} (累计跳过: {total_skipped})", end='\r')
                    continue
                
                # 确保文件夹存在
                os.makedirs(out_folder, exist_ok=True)
                print(f"\n[执行] 处理: {f} (类型: {category})")

                success = False
                if category == "STEP":
                    success = self.handle_step(file_path, out_folder, file_pure_name)
                else:
                    success = self.handle_native(file_path, out_folder, file_pure_name, category)

                if success:
                    total_processed += 1
                    self.processed_count += 1
                    if self.processed_count >= self.process_limit:
                        self.restart_sw()

        print(f"\n\n所有扫描结束。")
        print(f"本次新处理: {total_processed} 个")
        print(f"自动跳过: {total_skipped} 个")

    def handle_step(self, file_path, out_folder, base_name):
        try:
            importData = self.swApp.GetImportFileData(file_path)
            errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
            model = self.swApp.LoadFile4(file_path, "", importData, errors)
            if not model: return False

            self.export_formats(model, out_folder, base_name)
            self.extract_brep_metadata(model, out_folder, base_name)

            self.swApp.CloseAllDocuments(True)
            return True
        except Exception as e:
            print(f"  [失败] STEP: {e}")
            return False

    def handle_native(self, file_path, out_folder, base_name, category):
        try:
            doc_type = swDocASSEMBLY if category == "ASM" else swDocPART
            errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
            warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

            model = self.swApp.OpenDoc6(file_path, doc_type, swSaveAsOptions_Silent | 16, "", errors, warnings)
            if not model: return False

            self.extract_logic_tree(model, out_folder, base_name)

            if category == "ASM":
                print("  -> 合并装配体外壳...")
                temp_p = os.path.join(out_folder, "temp_shell.sldprt")
                self.safe_save(model, temp_p)
                self.swApp.CloseDoc(model.GetTitle)
                model = self.swApp.OpenDoc6(temp_p, swDocPART, swSaveAsOptions_Silent, "", errors, warnings)

            self.export_formats(model, out_folder, base_name)
            self.extract_brep_metadata(model, out_folder, base_name)

            self.swApp.CloseAllDocuments(True)
            tp = os.path.join(out_folder, "temp_shell.sldprt")
            if os.path.exists(tp): os.remove(tp)
            return True
        except Exception as e:
            print(f"  [失败] 原生: {e}")
            return False

    def export_formats(self, model, out_folder, base_name):
        # 核心导出：Parasolid 格式 (.x_t)
        self.safe_save(model, os.path.join(out_folder, f"{base_name}_ORIGINAL_BREP.x_t"))
        # 其他衍生格式
        self.safe_save(model, os.path.join(out_folder, f"{base_name}_SHELL.step"))
        self.safe_save(model, os.path.join(out_folder, f"{base_name}_MESH.stl"))

    def extract_brep_metadata(self, model, out_folder, base_name):
        try:
            meta = {"bodies": []}
            bodies = model.GetBodies2(0, True)
            if bodies:
                for b_idx, body in enumerate(bodies):
                    b_info = {"id": b_idx, "faces": []}
                    faces = body.GetFaces()
                    if faces:
                        for f_idx, face in enumerate(faces):
                            surf = face.GetSurface()
                            type_id = surf.Identity
                            b_info["faces"].append({
                                "id": f_idx,
                                "type": SURFACE_IDENTITY_MAP.get(type_id, "Other"),
                                "area": face.GetArea(),
                            })
                    meta["bodies"].append(b_info)
            with open(os.path.join(out_folder, f"{base_name}_brep_meta.json"), "w") as f:
                json.dump(meta, f, indent=4)
        except Exception as e:
            print(f"  [警告] 元数据提取跳过: {e}")

    def extract_logic_tree(self, model, out_folder, base_name):
        try:
            tree = []
            feat = model.FirstFeature()
            while feat:
                try:
                    name, f_type = feat.Name, feat.GetTypeName
                except:
                    name = feat.Name() if callable(feat.Name) else "Unknown"
                    f_type = feat.GetTypeName() if callable(feat.GetTypeName) else "Unknown"
                if f_type not in ["HistoryFolder", "SelectionViewFolder", "OriginProfileFeature", "MaterialFolder"]:
                    tree.append({"feature": name, "type": f_type})
                feat = feat.GetNextFeature()
            if tree:
                with open(os.path.join(out_folder, f"{base_name}_logic_tree.json"), "w", encoding="utf-8") as f:
                    json.dump(tree, f, indent=4, ensure_ascii=False)
        except: pass


if __name__ == "__main__":
    IN = r"D:\Lzm_Temp_Data\Li_temp_project\grabcad_downloads\9001_10000"
    OUT = r"D:\Lzm_Temp_Data\Li_temp_project\Dataset_1\9001_10000"

    proc = SolidWorksProcessor(process_limit=100)
    proc.process_dataset(IN, OUT)
