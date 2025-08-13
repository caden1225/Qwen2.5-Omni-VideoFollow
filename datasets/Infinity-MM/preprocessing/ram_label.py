'''
torchrun --nproc_per_node=8 ram_label.py
'''
'''
ram打标, 在当前文件目录下创建res-test_set子目录, 修改test_set和image_paths
'''

import sys
sys.path.insert(0, '/share/project/zhaohuxing/mask2former_classification/florence')

import argparse
import numpy as np
import random

import torch
import sys

from PIL import Image
import ram
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
import glob
from tqdm import tqdm
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

print(f"ram module is imported from: {ram.__file__}")


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8
    torch.cuda.set_device(rank)
    print(f"Process {rank}/{world_size} initialized on GPU {torch.cuda.current_device()}")

    return rank, world_size


def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":

    rank, world_size = setup()

    weight_path = "/share/project/zhaohuxing/mask2former_classification/florence/weight/ram_plus_swin_large_14m.pth"

    # test_set = "allava_laion"
    # test_set = "DenseFusion-4V-100K"
    # test_set = "new_imgs"
    # test_set = "Cambrian7M"
    input = sys.argv[1]
    input_ls = input.split("/")
    test_set = "--".join(input_ls[-2:]).split(".")[0]
    save_dir = f"/share/project/zzy_jjt_hzc/res-test_set/{test_set}"

    os.makedirs(save_dir, exist_ok=True)

    out_file = os.path.join(save_dir, f"ram-{test_set}-{rank}.jsonl")
    problem_file = os.path.join(save_dir, f"ram-{test_set}-problems.txt")
    print(test_set)

    image_size = 384

    #image_paths = [os.path.join(input, item) for item in os.listdir(input)]

    # image_paths = []
    # image_file = "/share/project/zzy/120M_ds/temp/new_imgs.txt"
    # with open(image_file, "r") as f:
    #     for line in f:
    #         image_paths.append(line.strip())
    
    # json
    with open(input, 'r', encoding='utf-8') as file:
        data = json.load(file)
    image_paths = [item['image'] for item in data if 'image' in item]

    # jsonl
    # image_paths = []
    # with open('/share/projset/mmdatasets-raw/Cambrian-Finetune-10M/gpt4v_77k.jsonl', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         line = json.loads(line)
    #         image_paths.append(os.path.join("/share/projset/mmdatasets-raw/Cambrian-Finetune-10M", line["image"]))

    image_num = len(image_paths)

    print(f"The number of images is {image_num}")

    transform = get_transform(image_size=image_size)

    ####### load model
    model = ram_plus(pretrained=weight_path,
                             image_size=384,
                             vit='swin_l').to(rank)

    model = DDP(model, device_ids=[rank])
    model.eval()

    results = []

    problems = []

    per_process = len(image_paths) // world_size
    start_index = rank * per_process
    end_index = start_index + per_process if rank != world_size - 1 else len(image_paths)
    local_paths = image_paths[start_index:end_index]

    for i, image_path in enumerate(tqdm(local_paths, total=len(local_paths))):
        try:
            image = transform(Image.open(image_path)).unsqueeze(0).to(f"cuda:{rank}")

            res = inference(image, model.module)
            results.append({"image_path": image_path, "score": res[0], "tags": res[1], "标签": res[2]})
        except Exception as e:
            print(e, image_path)
            problems.append(image_path)
            continue

        if (i + 1) % 1000 == 0:  # 每处理1000个图像后保存
            with open(out_file, "a+", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            results = []

    with open(out_file, "a+", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if rank == 0 and problems:
        with open(problem_file, 'a+') as f:
            for item in problems:
                f.write(item + '\n')
    cleanup()
    