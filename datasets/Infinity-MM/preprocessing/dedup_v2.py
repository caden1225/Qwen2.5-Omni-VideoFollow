#此代码对 同一数据集下边的所有文件
#1. 实现去重(phash和conversations相同)，存储在 output_file_path中
#2. 记录重复信息（image相同） ，存储在 duplication_info_file_path中
import json
from collections import defaultdict
import os
from tqdm import tqdm


# Directory and file paths for the specific JSON files to process
input_file_paths = [

'/share/project/zsy/9_25/allava_vflan/ALLaVA-Instruct-LAION-4V.json',
'/share/project/zsy/9_25/allava_vflan/ALLaVA-Caption-LAION-4V.json',



]

# Base directories for output，根据自己想保存的路径修改dir
#base_duplication_info_dir = "/share/project/zsy/9_6/check_llava-onevision/infographic_vqa"
#base_output_dir = "/share/project/zsy/9_6/dataset_after_deduplication/infographic_vqa"

base_duplication_info_dir = "/share/project/zsy/9_6/check/allava_vflan"
base_output_dir = "/share/project/zsy/9_6/dataset_after_deduplication/allava_vflan"

# Ensure the output directories exist
os.makedirs(base_duplication_info_dir, exist_ok=True)
os.makedirs(base_output_dir, exist_ok=True)

# Step 1: Load all JSON data from all input files
all_data = []
source_files = {}

for input_file_path in input_file_paths:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            entry['source_file'] = input_file_path     #Keep track of the source file for each entry
        all_data.extend(data)
        source_files[input_file_path] = data

# Step 2: Group IDs by the same image, including conversations
image_groups = defaultdict(list)
for entry in tqdm(all_data, desc="Grouping IDs by image"):
    image = entry['image']

    # Handle cases where `image` is a list
    if isinstance(image, list):
        image = tuple(image)  # Convert list to a tuple, which is hashable
    # If image is not a string, convert to a string for a consistent key
    elif not isinstance(image, str):
        image = str(image)

    # Store ID and conversations for manual review
    image_groups[image].append({
        'id': entry['id'],
        'source': entry['source'],
        'source_file': entry['source_file'],
        'conversations': entry['conversations'],
        'phash': entry['phash']
    })

# Filter out groups with only one ID
filtered_image_groups = {str(key): value for key, value in image_groups.items() if len(value) > 1}

# Save the filtered duplication info into a JSON file
duplication_info_file_path = os.path.join(base_duplication_info_dir, "duplication_info.json")  #保存同一数据集中的重复信息，记录了“image”，“id”，“source”，“source_file”,"conversations"信息
os.makedirs(os.path.dirname(duplication_info_file_path), exist_ok=True)
with open(duplication_info_file_path, 'w', encoding='utf-8') as duplication_file:
    json.dump(filtered_image_groups, duplication_file, ensure_ascii=False, indent=4)

# Step 3: Find and remove duplicates with the same image and conversation
unique_entries = []
seen_entries = set()  # To keep track of unique (image, conversations) pairs

for entry in tqdm(all_data, desc="Removing duplicates"):
    # image = entry['image']

    # # Handle cases where `image` is a list
    # if isinstance(image, list):
    #     image = tuple(image)  # Convert list to a tuple, which is hashable
    # # If image is not a string, convert to a string for a consistent key
    # elif not isinstance(image, str):
    #     image = str(image)

    # conversations = json.dumps(entry['conversations'], sort_keys=True)  # Convert conversations to a sorted JSON string for comparison
    # unique_key = (image, conversations)

    phash = entry['phash']

    if isinstance(phash, list):
        phash = tuple(image)
    elif not isinstance(phash, str):
        phash = str(phash)

    conversations = json.dumps(entry['conversations'], sort_keys=True)
    unique_key = (phash, conversations)
     

    if unique_key not in seen_entries:
        seen_entries.add(unique_key)
        unique_entries.append(entry)

# Save the filtered data to new JSON files, split by original input file
for input_file_path in input_file_paths:
    original_file_name = os.path.basename(input_file_path).replace('.json', '')
    output_file_path = os.path.join(base_output_dir, f"{original_file_name}_dedup.json")      #保存去重后的json

    # Filter unique entries for the specific input file
    filtered_entries = [entry for entry in unique_entries if entry['source_file'] == input_file_path]

    # Save the filtered data to a new JSON file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(filtered_entries, output_file, ensure_ascii=False, indent=4)

    print(f"Filtered data saved to {output_file_path}")

print(f"Duplication information saved to {duplication_info_file_path}")
