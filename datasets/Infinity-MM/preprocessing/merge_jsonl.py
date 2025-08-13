
import json
import os

from tqdm import tqdm

# Paths to the main JSON file and the directory containing the sub-files
main_json_path = '/share/project/zsy/10_24/phash/zh_filtered_translated.json'
sub_files_dir = '/share/project/zzy_jjt_hzc/res-test_set/phash--zh_filtered_translated'
file_name = os.path.basename(sub_files_dir)

# Load the main JSON file
with open(main_json_path, 'r', encoding='utf-8') as main_file:
    main_data = json.load(main_file)

print(len(main_data))
# Create a dictionary to map image_path to the main JSON entries
image_path_to_main_entry = {}
for entry in main_data:
    img_paths = entry['image']

    if isinstance(img_paths, list):
        for img_path in img_paths:
            img_path = img_path.strip()
            if img_path not in image_path_to_main_entry:
                image_path_to_main_entry[img_path] = []
            image_path_to_main_entry[img_path].append(entry)
    else:
        img_path = img_paths.strip()
        if img_path not in image_path_to_main_entry:
            image_path_to_main_entry[img_path] = []
        image_path_to_main_entry[img_path].append(entry)


    # img_path = entry['image'].strip()
    # if img_path not in image_path_to_main_entry:
    #     image_path_to_main_entry[img_path] = []
    # image_path_to_main_entry[img_path].append(entry)


print(len(image_path_to_main_entry))     #输出唯一图片路径的数量

# Iterate through each sub-file in the directory
for sub_file_name in os.listdir(sub_files_dir):
    sub_file_path = os.path.join(sub_files_dir, sub_file_name)
    if sub_file_name.startswith("filtered"):
        #print(f"{sub_file_name}")
        with open(sub_file_path, 'r', encoding='utf-8') as sub_file:
            first_line = True
            for line in sub_file:
                sub_entry = json.loads(line)
                image_path = sub_entry['image_path']
                score = sub_entry['score']
                tags = sub_entry['tags']


            #if image_path == "/share/projset/mmdatasets-raw/Cambrian-Finetune-10M/coco/train2017/000000033471.jpg":
            #    pass
            
            # Debug output for every line
            #if image_path == '/share/projset/mmdatasets-raw/Cambrian-Finetune-10M/coco/train2017/000000102858.jpg':
            #    print(f"Processing first record in {sub_file_name}:")
            #    print(f"image_path: {image_path}")
            #    print(f"score: {score}")
            #    print(f"tags: {tags}")
            #    first_line = False

            # Find the corresponding main entry and update it
                if image_path.strip() in image_path_to_main_entry:
                    temp = []
                    for main_entry in image_path_to_main_entry[image_path.strip()]:
                        #print(f"Before update: {main_entry}")
                        main_entry['score'] = score
                        main_entry['tags'] = tags

                        temp.append(main_entry)

                        #print(f"After update: {main_entry}")
                    image_path_to_main_entry[image_path.strip()] = temp
                else:
                    print(f"Warning: image_path {image_path} not found in main JSON.")

# Save the updated main JSON to a new file
res = []


for v in tqdm(image_path_to_main_entry.values()):
    res += v

output_json_path = f'/share/project/zsy/10_24/out/{file_name}.json'
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w', encoding='utf-8') as output_file:
    json.dump(res, output_file, ensure_ascii=False, indent=4)

print(f"Updated JSON saved to {output_json_path} and the number of the file is {len(res)}")
