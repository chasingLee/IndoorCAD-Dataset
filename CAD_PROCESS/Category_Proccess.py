import os
import shutil
import csv
import json

def str_to_bool(val):
    """辅助函数：将字符串转换为布尔值"""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ['true', '1', 'yes', 'y', '合格', 'pass']

def read_list_file(file_path):
    """最强兼容版：自动处理空格、BOM和编码问题"""
    data = []
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    elif ext == '.csv':
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames if name]
            
            headers = reader.fieldnames
            print(f"--- 调试信息：识别到的表头为：{headers} ---")

            # 根据你的 CSV 实际列名进行匹配
            target_col = 'folder_name'
            type_col = 'category'  # 这里已改为你 CSV 中的 'category'
            qual_col = 'is_qualified'

            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items() if k}
                try:
                    data.append({
                        'folder_name': row[target_col],
                        'file_type': row[type_col],
                        'is_qualified': str_to_bool(row[qual_col])
                    })
                except KeyError as e:
                    print(f"❌ 找不到列: {e}。请检查 CSV 表头。")
                    exit()
    return data

def count_files_in_dir(directory):
    """统计一个文件夹内所有文件的数量（包括子目录里的文件）"""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def process_folders(list_file, source_base_dir, target_qualified_dir, target_type_dir):
    """核心处理逻辑，带计数功能"""
    
    # 初始化计数器
    qualified_folders_count = 0
    total_files_count = 0
    
    print(f"正在读取配置文件: {list_file} ...")
    folder_data_list = read_list_file(list_file)
    print(f"共读取到 {len(folder_data_list)} 条记录。\n" + "-"*40)
    
    for item in folder_data_list:
        folder_name = item['folder_name']
        file_type = item['file_type']
        is_qualified = item['is_qualified']
        
        src_path = os.path.join(source_base_dir, folder_name)
        
        if not os.path.exists(src_path) or not os.path.isdir(src_path):
            print(f"⚠️ [跳过] 找不到原始文件夹: {folder_name}")
            continue

        if is_qualified:
            # --- 任务 1：汇总合格文件夹 ---
            dst_qual_path = os.path.join(target_qualified_dir, folder_name)
            shutil.copytree(src_path, dst_qual_path, dirs_exist_ok=True)
            
            # --- 任务 2：按类别分类 ---
            dst_type_path = os.path.join(target_type_dir, file_type, folder_name)
            shutil.copytree(src_path, dst_type_path, dirs_exist_ok=True)
            
            # 统计计数
            qualified_folders_count += 1
            # 计算当前这个文件夹里有多少个文件，累加到总数
            current_files = count_files_in_dir(src_path)
            total_files_count += current_files
            
            print(f"✅ [处理成功] {folder_name} (类别: {file_type}) - 包含 {current_files} 个文件")
        else:
            print(f"❌ [不合格跳过] {folder_name}")

    # ================= 最终统计报告 =================
    print("\n" + "="*40)
    print("        🎉 任务处理完成报告")
    print("-" * 40)
    print(f" 📂 成功处理合格文件夹总数: {qualified_folders_count} 个")
    print(f" 📄 累计复制文件总数:       {total_files_count} 个")
    print(f" 📍 合格总库地址: {target_qualified_dir}")
    print(f" 📍 分类库地址:   {target_type_dir}")
    print("=" * 40)


if __name__ == "__main__":
    # 路径配置
    LIST_FILE_PATH = r"D:\Lzm_Temp_Data\Li_temp_project\Dataset_1\8001_9000\summary.csv" 
    SOURCE_BASE_DIR = r"D:\Lzm_Temp_Data\Li_temp_project\Dataset_1\8001_9000"
    TARGET_QUALIFIED_DIR = r"D:\Lzm_Temp_Data\Li_temp_project\Dataset_processed\8001_9000"
    TARGET_TYPE_DIR = r"D:\Lzm_Temp_Data\Li_temp_project\Dataset_Categoried\8001_9000"

    process_folders(
        list_file=LIST_FILE_PATH,
        source_base_dir=SOURCE_BASE_DIR,
        target_qualified_dir=TARGET_QUALIFIED_DIR,
        target_type_dir=TARGET_TYPE_DIR
    )