
print("hello")
import json 
import os 


data_1_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preference_top_20k_v6.json"
data_1_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/share4v_preference_top_20k_v6.json"
data_1_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preference_top_20k_v6_save.json"

data_2_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_laion4v_20k_v1.json"
data_2_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/share4v_laion4v_20k_v1.json"
data_2_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_laion4v_20k_v1_save.json"

data_3_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/sharegpt4v_v3_50k.json"
data_3_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/sharegpt4v_v3_50k.json"
data_3_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/sharegpt4v_v3_50k_save.json"

data_4_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/sharegpt4v_v2_102k.json"
data_4_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/sharegpt4v_v2_102k.json"
data_4_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/sharegpt4v_v2_102k_save.json"

data_5_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/sharegpt4v_v1_12k.json"
data_5_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/sharegpt4v_v1_12k.json"
data_5_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/sharegpt4v_v1_12k_save.json"

data_6_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preferences.json"
data_6_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/share4v_preferences.json"
data_6_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preferences_save.json"

data_7_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preference_top_50k_v7.json"
data_7_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/share4v_preference_top_50k_v7.json"
data_7_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preference_top_50k_v7_save.json"

data_8_path = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preference_sampled_20k_v6.json"
data_8_path_raw = "/share/projset/mmdatasets-raw/sharegpt4v/share4v_preference_sampled_20k_v6.json"
data_8_path_save = "/share/project/wyx/mmdatasets/json_output/raw/sharegpt4v/share4v_preference_sampled_20k_v6_save.json"

data_9_path = "/share/project/wyx/mmdatasets/json_output/hf/Lin-Chen/share-captioner_coco_lcs_sam_1246k_1107.json"
data_9_path_raw = "/share/projset/mmdatasets-hf/20240731/Lin-Chen/ShareGPT4V_20240725/share-captioner_coco_lcs_sam_1246k_1107.json"
data_9_path_save = "/share/projset/mmdatasets-hf/20240731/Lin-Chen/ShareGPT4V_20240725/share-captioner_coco_lcs_sam_1246k_1107_save.json"

data_10_path = "/share/project/wyx/mmdatasets/json_output/hf/Lin-Chen/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
data_10_path_raw = "/share/projset/mmdatasets-hf/20240731/Lin-Chen/ShareGPT4V_20240725/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
data_10_path_save = "/share/projset/mmdatasets-hf/20240731/Lin-Chen/ShareGPT4V_20240725/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_save.json"

# data_11_path_raw = "/share/project/wyx/mmdatasets/json_output/hf/Lin-Chen/share-captioner_coco_lcs_sam_1246k_1107.json"
# data_11_path_save = "/share/project/wyx/mmdatasets/json_output/hf/Lin-Chen/share-captioner_coco_lcs_sam_1246k_1107_save.json"

def read_json(p):
    with open(p) as f:
        content = f.read()

    content = json.loads(content)

    print("data length is " + str(len(content)))
    return content 

from tqdm import tqdm

def process_data(p_raw, p, p_save):
    
    raw_data = read_json(p_raw)
    data = read_json(p)

    data_ = {}
    for d in data:
        data_[d["id"]] = d["image"]
    
    final_data = []
    index = 0
    for raw_d in tqdm(raw_data, total=len(raw_data)):

        index += 1

        if index == 1:
            print(raw_d)

        ids = raw_d["id"]
        if ids not in data_.keys():
            continue

        image_path = data_[ids]
        # print(f"id is {ids} image_path is {image_path}")
        conversations = raw_d["conversations"]
        source = p_raw.split("/")[-1].split(".json")[0]

        conversations_1 = conversations[0]
        human_text = conversations_1["value"]
        if "<image>" not in human_text:
            print("lack <image> token")

            exit(0)


        ## additional information            

        source_path = p_raw 

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
        # print(final_data)
        # break 
        # print()

    with open(p_save, "w") as f:
        f.write(json.dumps(final_data))


def process_data_6(p_raw, p, p_save):
    
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


def process_data_9(p_raw, p, p_save):

    base_dir = "/share/projset/mmdatasets-raw/Cambrian-Finetune-10M/"
    raw_data = read_json(p_raw)

    final_data = []
    index = 0
    for raw_d in tqdm(raw_data, total=len(raw_data)):

        index += 1

        if index == 1:
            print(raw_d)

        ids = raw_d["id"]
        image_path = os.path.join(base_dir, raw_d["image"])

        # print(f"id is {ids} image_path is {image_path}")
        conversations = raw_d["conversations"]
        source = p_raw.split("/")[-1].split(".json")[0]

        conversations_1 = conversations[0]
        human_text = conversations_1["value"]
        if "<image>" not in human_text:
            print("lack <image> token")

            exit(0)


        ## additional information            

        source_path = p_raw 

        final_data.append({
            "id": ids,
            "image": image_path,
            "source": source,
            "source_path": source_path,
            "conversations": conversations,

            ## data_1
            # "index": raw_d["index"],
            # "logit": raw_d["logit"],
        })


        if index == 1:
            print(f"final data is {final_data}")
        # print(final_data)
        # break 
        # print()

    with open(p_save, "w") as f:
        f.write(json.dumps(final_data))

# process_data(data_1_path_raw, data_1_path, data_1_path_save)
# process_data(data_2_path_raw, data_2_path, data_2_path_save)
process_data(data_3_path_raw, data_3_path, data_3_path_save)
process_data(data_4_path_raw, data_4_path, data_4_path_save)
process_data(data_5_path_raw, data_5_path, data_5_path_save)

process_data_6(data_6_path_raw, data_6_path, data_6_path_save)

process_data(data_7_path_raw, data_7_path, data_7_path_save)
process_data(data_8_path_raw, data_8_path, data_8_path_save)
# process_data_9(data_9_path_raw, data_9_path, data_9_path_save)
# process_data_9(data_11_path_raw, data_9_path, data_11_path_save)
# 
process_data(data_9_path_raw, data_9_path, data_9_path_save)
process_data(data_10_path_raw, data_10_path, data_10_path_save)




    
    


