import os
import json
import base64
import time
import csv
import requests
from PIL import Image

# ================= 配置区 =================
API_KEY = ""  # 替换为你的实际 Key
URL = ""
MODEL_NAME = "qwen3vl"  # 确认这是校内平台提供的模型 ID

DATASET_ROOT = "./Dataset_1/8001_9000"  # 数据集根目录
SUMMARY_JSON_PATH = "./Dataset_1/8001_9000/summary.json"
SUMMARY_CSV_PATH = "./Dataset_1/8001_9000/summary.csv"

# 限速：每分钟10万Token。Vision模型单次请求大，建议每分钟处理 6-10 个
SAFE_WAIT_TIME = 1
# ==========================================


def get_image_subfolder(obj_folder):
    """寻找存放图片的文件夹"""
    subs = [os.path.join(obj_folder, d) for d in os.listdir(obj_folder) if os.path.isdir(os.path.join(obj_folder, d))]
    return subs[0] if subs else obj_folder


def encode_image(image_path):
    """编码图片为Base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_4_grids(image_folder, temp_dir):
    """32张采样16张，拼成4张2x2的大图并缩放"""
    all_imgs = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not all_imgs:
        return []

    step = max(1, len(all_imgs) // 16)
    selected_imgs = all_imgs[::step][:16]

    grid_paths = []
    for i in range(0, len(selected_imgs), 4):
        batch = selected_imgs[i : i + 4]
        with Image.open(batch[0]) as first_img:
            w, h = first_img.size
            grid_img = Image.new("RGB", (w * 2, h * 2))
            for idx, img_path in enumerate(batch):
                with Image.open(img_path) as img:
                    grid_img.paste(img, ((idx % 2) * w, (idx // 2) * h))

            # 缩放至1024以内以节省校内算力配额和Token
            grid_img.thumbnail((1024, 1024))
            p = os.path.join(temp_dir, f"temp_grid_{i//4}.jpg")
            grid_img.save(p, quality=85)
            grid_paths.append(p)
    return grid_paths


def query_hpc_vlm(image_paths):
    """
    根据交大示例改写的 Vision 调用函数
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    # 构建多模态内容
    content = [
        {
            "type": "text",
            "text": (
                "你是一个专业的3D资产审核员。请观察提供的4张拼接图（每张含4个视角）。\n"
                "任务1：分类。仅限：[椅子, 桌子, 柜子(储存器具), 沙发, 桌上物品，其他家具，室内场景]。\n"
                "任务2：质量判断。若模型为工业零件类（如螺丝，合页等）模型破碎、渲染错误或不属上述类别，is_qualified设为false。\n"
                "任务3：描述。撰写约200字详细描述（不少于180字），涵盖物品类型、形状、设计风格及细节。\n"
                "请严格以JSON格式输出：{'category': '...', 'is_qualified': true/false, 'description': '...'}"
            ),
        }
    ]

    for p in image_paths:
        base64_data = encode_image(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}})

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.01,  # 降低随机性，保证分类准确
        "top_p": 0.9,
        "stream": False,
    }

    # 尝试调用，包含简单的重试逻辑
    for _ in range(3):
        try:
            response = requests.post(URL, headers=headers, json=data, timeout=120)
            if response.status_code == 200:
                res_json = response.json()
                raw_text = res_json["choices"][0]["message"]["content"].strip()
                # 清洗 JSON 格式
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_json)
            else:
                print(f" 请求失败({response.status_code}): {response.text}")
                time.sleep(10)
        except Exception as e:
            print(f" 出错: {e}")
            time.sleep(10)
    return None


def main():
    summary_list = []
    temp_dir = "./Dataset_1/1001_2000/temp_grids"
    os.makedirs(temp_dir, exist_ok=True)

    objects = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]

    for idx, object_id in enumerate(objects):
        obj_path = os.path.join(DATASET_ROOT, object_id)
        img_folder = get_image_subfolder(obj_path)

        print(f"[{idx+1}/{len(objects)}] 正在处理: {object_id}...")

        grids = create_4_grids(img_folder, temp_dir)
        if not grids:
            continue

        res = query_hpc_vlm(grids)

        if res:
            # 1. 保存 description.txt 到子文件夹
            with open(os.path.join(obj_path, "description.txt"), "w", encoding="utf-8") as f:
                f.write(res.get("description", ""))

            # 2. 存入汇总数据
            summary_list.append(
                {"folder_name": object_id, "category": res.get("category"), "is_qualified": res.get("is_qualified")}
            )
            print(f"   成功: {res.get('category')} | 合格: {res.get('is_qualified')}")

        # 清理临时图
        for p in grids:
            if os.path.exists(p):
                os.remove(p)

        # 3. 限速休眠（TPM 100k 限制）
        time.sleep(SAFE_WAIT_TIME)

    # --- 输出汇总结果 ---
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, ensure_ascii=False, indent=4)

    if summary_list:
        with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=summary_list[0].keys())
            writer.writeheader()
            writer.writerows(summary_list)

    print(f"\n任务处理完成！汇总表见: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
