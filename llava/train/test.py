import json

def sum_doc_length_and_num_images(jsonl_file_path):
    total_doc_length = 0
    total_num_images = 0
    datas = []
    with open(jsonl_file_path, "r") as f:
        jsonl_data = [json.loads(line) for line in f]
        datas = [item for item in jsonl_data if 'content_image' in item and len(item['content_image']) > 0]
        print(len(datas))
    for data in datas:

        # 提取 doc_length 和 num_imgs
        doc_length = data.get("quality_signals", {}).get("doc_length", 0)
        num_images = data.get("quality_signals", {}).get("num_imgs", 0)
        # 累加统计
        total_doc_length += doc_length
        total_num_images += num_images

    return total_doc_length, total_num_images

# 示例调用
jsonl_file_path = "/root/model/in_dataset/data/IN-PMC/part00/part00.jsonl"  # 替换为你的 JSONL 文件路径
total_doc_length, total_num_images = sum_doc_length_and_num_images(jsonl_file_path)
print(f"Total doc_length: {total_doc_length}")
print(f"Total num_images: {total_num_images}")