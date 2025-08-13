# **Data Process**
We build the following pipeline to process our data, and we'll elucidate our pipeline steps by steps.

### **1. Uniform Format:**
In this part, we want to unify all json format like:
```json
{"id": "10", 
 "image": "/path/of/the/image", 
 "source": "llava", 
 "source_file": "/the/original/json",
 "conversations": [
 {
 "from": "human", 
 "value": "<image> Can you create a descriptive poem about the scene?"
 }, 
 {
 "from": "gpt", 
 "value": "Upon a plate of verdant hue,\nA cluster of roasted nuts we view.\nCashews golden, shining bright,\nAlmonds dark, a roasted delight.\nSprinkled seeds of sesame light,\nEnhancing flavors, a gentle bite.\nOn white wood they rest in grace,\nA rustic charm, a peaceful place.\nSimple joys in a vibrant array,\nA perfect treat to start the day."
 }], 
 #additional keys you want to save.
 }
```
And we use the following code：
```python
python process.py
```
However, due to the different format of each original json, you may need to write the code on your own or modify the related code as below:
```python
data_1_path = "/original/json/path"
data_1_path_raw = "/updated/json/with/absolute/image/path"
data_1_path_save = "/final/json/path/you/want/to/save"

#The following function should be modified according to the original json format
def process_data_2(p_raw, p, p_save):
    
    raw_data = read_json(p_raw)
    data = read_json(p)

    base_dir = "/share/projset/mmdatasets-raw/Cambrian-Finetune-10M"

    final_data = []
    index = 0
    for raw_d in tqdm(raw_data, total=len(raw_data)):


        if index == 0:
            print(raw_d)

        samples = raw_d["samples"]

        for sam in samples:
            index += 1

            ids = index

            image_path = sam["image"].replace("sharegpt4v", base_dir)

            # print(f"id is {ids} image_path is {image_path}")

            conversations = [
                {"from": "human", "value": "<image>"}, 
                {"from": "gpt", "value": f"{sam['text']}"}
            ]
            
            source = p_raw.split("/")[-1].split(".json")[0]
            source_path = p_raw 

            conversations_1 = conversations[0]
            
            human_text = conversations_1["value"]
            if "<image>" not in human_text:
                print("lack <image> token")

                exit(0)
    
            final_data.append({
                "id": ids,
                "image": image_path,
                "source": source,
                "source_file": source_path,
                "conversations": conversations,

                ## data_1
                # "index": raw_d["index"],
                # "logit": raw_d["logit"],
            })


            if index == 1:
                print(f"final data is {final_data}")
            

    with open(p_save, "w") as f:
        f.write(json.dumps(final_data))

process_data(data_1_path_raw, data_1_path, data_1_path_save)
#process_data(data_2_path_raw, data_2_path, data_2_path_save)
```

### **2. pHash label:**
- First, save the code phash.py, you need to change the 'output_file', 'error_file' on your own.
```python
output_file = f"/the/path/you/want/to/save/{name}"
error_file = f"/the/path/of/images/with/error/{name}"
```
- And then use the script:
```shell
bash run_batch_phash.sh
```
In run_batch_phash.sh, you need to modify the related code as following:
```shell
source /your/environment

PYTHON_SCRIPT="/actual/path/of/phash.py"
image_paths=(
the json files you want to add phash label
)
for image_path in "${image_paths[@]}"; do
    python $PYTHON_SCRIPT "$image_path" &
done

wait 

echo "All processes completed."
```
After that, you will get a new json {output_file} with a new key value **phash** for all successful record. Meanwhile, the images that can not be added phash will be saved in {error_file}.

### **3. Data Deduplication:**
We set the **rule**:
If the records in the same dataset have the same 'phash' and 'conversations', we keep only one record.
```python
python dedup_v2.py
```
The following part may be modified:
```python
# input_file_paths contain the json you want to process 
input_file_paths = [
'/share/project/zsy/9_25/allava_vflan/ALLaVA-Instruct-LAION-4V.json',
'/share/project/zsy/9_25/allava_vflan/ALLaVA-Caption-LAION-4V.json',
]

#You need to modify the code on your own
base_duplication_info_dir = "/the/path/of/duplication/directory"
base_output_dir = "/the/output/directory/of/unique/record"
```

### **4. Check the data format:**
To ensure the data format meets out expectations, we take the following step:
```python
python check_list.py
```
The 'files' and 'save_path' need to be modified:
```python
files = [
    "/the/jsons/you/want/to/check"
]
save_path = "/the/directory/of/checked/json"
```
After that, you will find the files with suffix 'checked', which guarantee that the files meet out expectations.

### **5. Ram++ label:**
We use Recognize Anything Plus Model to label each images.
- First, save the following code as 'ram_label.py':
In ram_label.py, you need to modify the following code below:
```python
import sys
sys.path.insert(0, '/share/project/zhaohuxing/mask2former_classification/florence')
#Adjust the code according to your reality

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
    #You can change the save_dir on your own
    os.makedirs(save_dir, exist_ok=True)

    out_file = os.path.join(save_dir, f"ram-{test_set}-{rank}.jsonl")
    problem_file = os.path.join(save_dir, f"ram-{test_set}-problems.txt")
    print(test_set)

    '''You may need to modify the following code to read all images of json'''
    
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
```
- Second, save the following code as 'merge.py':
In the merge.py, the following related code need to be modified:
```python
input = sys.argv[1]
input_ls = input.split("/")
test_set = "--".join(input_ls[-2:]).split(".")[0]
save_dir = f"./res-test_set/{test_set}"
final_file = os.path.join(save_dir, f"ram-{test_set}.jsonl")
```
- Third, save the following code as 'ram_filter.py':
In the ram_filter.py, the following part may need to be modified:
```python
#阈值1.4
score_threshold = 1.4

input = sys.argv[1]
input_ls = input.split("/")
test_set = "--".join(input_ls[-2:]).split(".")[0]
save_dir = f"./res-test_set/{test_set}"
os.makedirs(save_dir, exist_ok=True)
final_file = os.path.join(save_dir, f"ram-{test_set}.jsonl")
input_file_path = final_file
output_file_path = os.path.join(save_dir, f"filtered-ram-{test_set}.jsonl")
```
- run the following script:
```shell
bash run6.sh
```
And you need to modify the following script:
```shell
image_dirs=(
/the/jsons/that/you/want/to/label/ram++
)

for image_dir in "${image_dirs[@]}"; do
    torchrun --nproc_per_node=8 /actual/path/of/ram_label.py "$image_dir" 
    python /actual/path/of/merge.py "$image_dir"
    python /actual/path/of/ram_filter.py "$image_dir" 
done

wait 

echo "All processes completed."
```
- After that, we need to **combine** the new jsonl and the raw json, run the following script:
```python
python merge_jsonl.py
```
In merge_jsonl.py, you need to modify the following part:
```python
# Paths to the main JSON file and the directory containing the sub-files
main_json_path = '/the/origin/json/processed/in/step3'
sub_files_dir = '/the/directory/of/jsonl/in/ram_filter.py'
output_json_path = f'/the/path/you/want/to/save/the/json/{file_name}.json'
```
**After Step 5, you can gain another 2 key values, 'score' and 'tags'.**

### **6. Qwen2-vl-loss:**
- First, we need to transform our conversations format to conversations_new that Qwen2-vl model can process the conversation:
```python
python format_transform.py
```
And you to need modify the following code:
```python
file_path = [
    '/the/jsons/that/you/want/to/process'
]
    
for i in tqdm(range(len(data_list))):
    with open('/the/path/you/want/to/save/json/after/transforming/'+file_path[i].split('/')[-1], 'w') as f:
```
- Second, run the following code. 
```python
python t.py
```
You need to change the number of 'chunk_size' according to your reality, in our data, wu use 16 A100.


- Third, save the following code as 'compute_loss.py':
You need to modify the related code below:
```python
import sys
sys.path.append('/share/project/zhaohuxing/qwen2_vl_test')

from transformers_yt import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch 
import concurrent.futures
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# device = "cuda:0"

from argparse import ArgumentParser

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--split_id", type=int, default=0)
    arg_parser.add_argument("--device", type=int, default=0)
    args = arg_parser.parse_args()

    print(args)
    import json
    from tqdm import tqdm

    with open(f"/the/directory/of/the/chunk/of/json/of/the/former/part/chunk_{args.split_id}.json", "r") as f:
        data = json.load(f)


    processor = AutoProcessor.from_pretrained("./qwen2_vl_7b")
    tokenizer = AutoTokenizer.from_pretrained("./qwen2_vl_7b")
    split_id = args.split_id

                np.save(f"/the/directory/of/numpy/of/the/chunk/answer_losses_{split_id}.npy", answer_losses)
```
- Then save the following script as 'start_compute.sh':
You need to modify the following code:
 ```shell
 cd /your/work/directory
 
NUM_GPUS=8
START_ID=$1

# Loop through each GPU and start a process
for ((i=0; i<$NUM_GPUS; i++)); do
    # Run the script for each split on a different GPU
    # --split_id will match the GPU number
    CUDA_VISIBLE_DEVICES=$i /your/actual/python/env/python /actual/path/of/compute_loss.py --split_id $((i + START_ID)) --device 0 &
done

wait

echo "All processes finished."
 ```
- And Run the following script:
``` shell
bash start_12.sh
```
You need to modify the following code:
```shell
HOSTFILE=/your/actuacl/hostfile
START_ID=0
LINE_COUNT=0

while IFS= read -r line; do

    url=$(echo $line | awk '{print $1}')   # 使用空格作为分隔符提取IP地址

    ssh $url "bash /the/actual/path/of/start_compute.sh $START_ID" &
    
    START_ID=$((START_ID+8))
    LINE_COUNT=$((LINE_COUNT+1))

done < "$HOSTFILE"

wait
```
After that, you will get loss of each chunk of json.

- Finally, run the following script to **combine** loss of each json:
```python
python load_loss.py
```
The following part need to be modified:
```python
file_path = [
'/the/jsons/that/you/want/to/process/in/the/first/part'
]

for i in tqdm(range(16), desc="Combining all loss"):
    loss = np.load(f'/the/directory/of/numpy/of/the/chunk/answer_losses_{i}.npy')
    losses.extend(loss)

for i in tqdm(range(len(file_path)), desc="Save each file"):
    with open('/the/path/you/want/to/save/the/json/with/Qwen2-vl_loss/'+file_path[i].split('/')[-1], 'w') as f:
        json.dump(datas[i], f, ensure_ascii=False, indent=4)
        print(len(datas[i]))
```

### **After the above steps:**
You can get new json format:
```json
 {
"id": "1",
"source": "chartqa",
"source_file": "/source/of/original/json",
"image": "/the/path/of/the/image",
"conversations": [
{
"from": "human",
"value": "How much did Tunisia's population increase in 2019?\n<image>"
},
{
"from": "gpt",
"value": "1.11"}
],
"phash": 18068869667165728448,
"score": [
6.522897243499756,
6.415268421173096,
1.8905785083770752
],
"tags": [
"graph",
"number",
"screenshot"
],
"conversations_new": [
{
"role": "user",
"content": [
{
"type": "image",
"image": "/the/path/of/the/image",
"resized_height": 500,
"resized_width": 500},
{
"type": "text",
"text": "How much did Tunisia's population increase in 2019?\n<image>"}]
},
{
"role": "assistant",
"content": [
{
"type": "text",
"text": "1.11"}]
}
],
"qw2vl_loss": 3.8328657150268555
},
```