import os 
import json
from PIL import Image
from tqdm import tqdm
import imagehash
import sys
from multiprocessing import Pool, cpu_count

# '''以下是单进程运行'''
# def generate_phash_for_json(json_file, output_file):
#     image_data = []
#     i = 0

#     with open(json_file, 'r') as file:
#         data = json.load(file)

#     for entry in tqdm(data, desc="processing data"):
#         try:
#             with Image.open(entry['image']) as image:
#                 hash_value = imagehash.phash(image)
#                 phash_value = int(str(hash_value), 16)
#                 entry['phash'] = phash_value

#             i += 1
#             image_data.append(entry)
        
#         except Exception as e:
#             print(f"Error processing: {e} in {entry['image']} in file {json_file}. ")

#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with open(output_file, 'w') as f:
#         json.dump(image_data, f, indent=4)

#     print(f"phash data of {i} images have been saved to {output_file}")



# #json_file = str(sys.argv[1]).strip()
# #name = "-".join(json_file.split("/")[-2:])

# json_file = '/share/project/zhangjialing/RAM+sample/sample_10M_result_from_EVE-Pretrain-33M_laion_dedump_openimages_sam_safe/merge_priority_sample_10M-EVE-Pretrain-33M_laion_dedump_openimages_sam_safe.json'
# name = json_file.split('/')[-1]
# output_file = f"/share/project/zsy/9_25/phash/{name}"
# print(output_file)

# generate_phash_for_json(json_file, output_file)




'''多进程运行'''
#处理json中的image
def process_image(entry):
    if 'image' not in entry or not entry['image']:
        return entry
    try:
        with Image.open(entry['image']) as image:
            hash_value = imagehash.phash(image)
            phash_value = int(str(hash_value), 16)
            entry['phash'] = phash_value
        return entry
    except Exception as e:
        print(f"Error processing: {e} in {entry['image']}.")
        return {'image': entry['image']}   #处理失败只返回image字段

#多进程处理所有图像
def generate_phash_for_json(json_file, output_file, error_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    #使用多进程
    with Pool(processes=cpu_count()) as pool:   #根据CPU核心数创建进程池
        results = list(tqdm(pool.imap(process_image, data), total=len(data), desc="Processing data"))

    #记录失败的image
    failed_images = []
    for entry in results:
        if 'phash' not in entry and 'image' in entry and entry['image']:
            failed_images.append(entry['image'])

    #处理成功的image
    processed_data = [entry for entry in results if 'phash' in entry]  #处理只有image的数据集
    # processed_data = [entry for entry in results]

    # 使用列表解析的方式进行过滤，并处理没有 'image' 字段的情况
    filtered_results = [entry for entry in results if 'image' not in entry or entry['image'] not in failed_images]    #处理部分text和image共存的数据集


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=4)

    if failed_images:
        os.makedirs(os.path.dirname(error_file), exist_ok=True)
        with open(error_file, 'w') as f:
            json.dump(failed_images, f, ensure_ascii=False, indent=4)

    print(f"phash data of {len(filtered_results)} images have been saved to {output_file}")
    print(f"Failed images of {len(failed_images)} have been saved to {error_file}")


# json_file = '/share/projset/2024082801/lmms-lab/json_files/all_data_filter2.json'
json_file = str(sys.argv[1]).strip()
name = json_file.split('/')[-1]
output_file = f"/share/project/zsy/10_24/phash/{name}"
error_file = f"/share/project/zsy/9_3/check_image/{name}"
print(f"The output file is {output_file}")

generate_phash_for_json(json_file, output_file, error_file)






# '''下边是给目录中的图片加上phash'''
# dir = str(sys.argv[1]).strip()

# name = "-".join(dir.split('priority_sample_benchmark_image/')[-1].split('/'))
# output_file = f"/share/project/zsy/9_19/phash/test_dataset_new/{name}.json"

# os.makedirs(os.path.dirname(output_file), exist_ok=True)

# image_phashes = []

# def process_image(image_dir, output_file):
#     for root, dirs, files in tqdm(os.walk(image_dir), desc="Processing each dir"):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
#                  file_path = os.path.join(root, file)

#                  try:
#                     with Image.open(file_path) as img:
#                         hash_value = imagehash.phash(img)
#                         phash_value = int(str(hash_value), 16)
#                         image_phashes.append({
#                             'image': file_path,
#                             'phash': phash_value
#                         })
#                  except Exception as e:
#                      print(f"Error processing {file}: {e}")

#     with open(output_file, 'w') as f:
#         json.dump(image_phashes, f, indent=4)


# process_image(dir, output_file)

# print(f"pHash values saved to {output_file}")                