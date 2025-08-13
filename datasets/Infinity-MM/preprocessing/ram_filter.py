import json
import os

import sys

#阈值1.4
score_threshold = 1.4

input = sys.argv[1]
input_ls = input.split("/")
test_set = "--".join(input_ls[-2:]).split(".")[0]
#test_set = 'allava_vflan--ALLaVA-Caption-LAION-4V_dedup'
save_dir = f"./res-test_set/{test_set}"
os.makedirs(save_dir, exist_ok=True)
final_file = os.path.join(save_dir, f"ram-{test_set}.jsonl")

input_file_path = final_file
output_file_path = os.path.join(save_dir, f"filtered-ram-{test_set}.jsonl")


os.makedirs(os.path.dirname(output_file_path), exist_ok=True)


with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        image_path = data['image_path']
        scores = data['score']
        tags = data['tags']

        # 过滤score和tags
        filtered_scores_tags = [(score, tag) for score, tag in zip(scores, tags) if score >= score_threshold]
        if filtered_scores_tags:
            filtered_scores, filtered_tags = zip(*filtered_scores_tags)
        else:
            filtered_scores, filtered_tags = [], []

        # 生成新的数据字典
        new_data = {
            'image_path': image_path,
            'score': list(filtered_scores),
            'tags': list(filtered_tags)
        }

        # 写入到新的jsonl文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')