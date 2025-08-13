import json
from tqdm import tqdm
import copy

from PIL import Image
import re
import os
import yaml
import pdb

def check_image(data):
    count = 0
    res = []
    n = 0
    for item in tqdm(data):
        c = 0
        n += 1
        for tmp in item['conversations']:
            c += tmp['value'].count('<image>')
        if c > 1:
            count += 1
            # print(f"current record:{n} has more than 1 <image>")
            # print(item)
            # pdb.set_trace()
        elif c == 0 and 'image' in item.keys():
            count += 1
            # print(f"current record:{n} has no <image>")
            # print(item)
            # pdb.set_trace()
        else:
            res.append(item)
    return count, res

def check_null(data):
    count = 0
    res = []
    for item in tqdm(data):
        c = 0
        item_t = copy.deepcopy(item)
        item_t['conversations'] = []
        flag = False
        for idx, tmp in enumerate(item['conversations']):
            if flag:
                flag = False
                continue
            if len(tmp['value'].strip()) != 0:
                item_t['conversations'].append(tmp)
            else:
                count += 1
                if idx % 2 == 0:
                    flag = True 
                else:
                    item_t['conversations'].pop()
                    
                # c += 1
                # import pdb
                # # pdb.set_trace()
                # break
        if len(item_t['conversations']) > 0:
            res.append(item_t)
        
    return count, res


def check_con(data):
    count = 0
    res = []
    for item in tqdm(data):
        for idx, tmp in enumerate(item['conversations']):
            if 'from' not in tmp.keys() or 'value' not in tmp.keys():
                count += 1
                break
            if idx % 2 == 0:
                if tmp['from'] != 'human':
                    count += 1
                    break
            else:
                if tmp['from'] != 'gpt':
                    count += 1
                    break
        if idx % 2 == 1:
            res.append(item)
    return count, res
            
                
def check_image_file(data):
    count = 0
    res = []
    for item in tqdm(data):
        if 'image' in item.keys():
            try:
                img = Image.open(item['image'])
                img.verify()
                res.append(item)
            except:
                print("broken:", item['image'])
                count += 1
        else:
            res.append(item)
    return count, res




def check_null(data):
    count = 0
    res = []
    for item in tqdm(data):
        c = 0
        item_t = copy.deepcopy(item)
        item_t['conversations'] = []
        flag = False
        for idx, tmp in enumerate(item['conversations']):
            if flag:
                flag = False
                continue
            if len(tmp['value'].strip()) != 0:
                item_t['conversations'].append(tmp)
            else:
                count += 1
                if idx % 2 == 0:
                    flag = True 
                else:
                    item_t['conversations'].pop()
                    
                # c += 1
                # import pdb
                # # pdb.set_trace()
                # break
        if len(item_t['conversations']) > 0:
            res.append(item_t)
        
    return count, res


def check_answer_in_question(data):
    count = 0
    res = []
    for item in tqdm(data):
        item_t = copy.deepcopy(item)
        item_t['conversations'] = []
        item_t['instruction label'] = []
        item_t['score'] = []
        for idx in range(int(len(item['conversations']) / 2)):
            if 'Answer:' in item['conversations'][2*idx]['value']:
                count += 1
                continue
            else:
                item_t['conversations'].append(item['conversations'][2*idx])
                item_t['conversations'].append(item['conversations'][2*idx + 1])
                # item_t['instruction label'].append(item['instruction label'][idx])
                # item_t['score'].append(item['score'][idx])
        if len(item_t['conversations']) > 0:
            res.append(item_t)
    return count, res   

def check_short_question(data):
    count = 0
    res = []
    for item in tqdm(data):
        item_t = copy.deepcopy(item)
        item_t['conversations'] = []
        item_t['instruction label'] = []
        item_t['score'] = []
        for idx in range(int(len(item['conversations']) / 2)):
            if len(item['conversations'][2*idx]['value'].replace("<image>", "").strip()) <= 5:
                count += 1
                continue
            else:
                item['conversations'][2*idx]['value'] = item['conversations'][2*idx]['value'].strip()
                item['conversations'][2*idx + 1]['value'] = item['conversations'][2*idx + 1]['value'].strip()
                item_t['conversations'].append(item['conversations'][2*idx])
                item_t['conversations'].append(item['conversations'][2*idx + 1])
                # item_t['instruction label'].append(item['instruction label'][idx])
                # item_t['score'].append(item['score'][idx])
        if len(item_t['conversations']) > 0:
            res.append(item_t)
    return count, res

def check_yes(data):
    import random
    yes = 0
    no = 0
    for item in tqdm(data):
        for idx in range(int(len(item['conversations']) / 2)):
            count += 1
            if item['conversations'][2*idx + 1]['value'].startswith('Yes'):
                yes += 1
            if item['conversations'][2*idx + 1]['value'].startswith('No'):
                no += 1
    ratio = no / yes
    
    count = 0
    res = []
    
    for item in tqdm(data):
        item_t = copy.deepcopy(item)
        item_t['conversations'] = []
        item_t['instruction label'] = []
        item_t['score'] = []
        for idx in range(int(len(item['conversations']) / 2)):
            # if 'Yes,' in item['conversations'][2*idx + 1]['value'] or item['conversations'][2*idx + 1]['value'] == 'Yes':
            if item['conversations'][2*idx + 1]['value'].startswith('Yes') and random.random() > ratio:
                count += 1
                continue
            else:
                item_t['conversations'].append(item['conversations'][2*idx])
                item_t['conversations'].append(item['conversations'][2*idx + 1])
                # item_t['instruction label'].append(item['instruction label'][idx])
                # item_t['score'].append(item['score'][idx])
        if len(item_t['conversations']) > 0:
            res.append(item_t)
    return count, res    

def add_image(data):
    res = []
    for item in tqdm(data):
        if '<image>' not in item['conversations'][0]['value']:
            item['conversations'][0]['value'] = '<image>\n' + item['conversations'][0]['value']
        res.append(item)
    return res
                
files = ["/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_chart_release_update.json", "/share/project/zzy/120M_ds/zzy/filter_out/LVIS-Instruct-lvis_instruct4v_cap_111k_dedup.json", "/share/project/zzy/120M_ds/zzy/filter_out/LVIS-Instruct-lvis_instruct4v_vqa_111k_dedup.json", "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-CC3M-Pretrain-595K-chat_dedup.json", "/share/project/zzy/120M_ds/zzy/filter_out/Visdial-Visdial-20k.json", "/share/project/zzy/120M_ds/zzy/filter_out/Visdial-Visdial-123k.json", "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-sharegpt4v_v4_50k.json", "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-share4v_preference_sampled_50k_v5.json", "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-sharegpt4v_instruct_gpt4-vision_cap100k.json", "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-sharegpt4v_v2_102k.json", "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-share-captioner_coco_lcs_sam_1246k_1107.json", "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-Sharegpt4v_mix665k.json", "/share/project/zzy/120M_ds/zzy/filter_out/STVQA_stvqa-19k.json", "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_chart_release_update.json", "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_chart_data_part3.json", "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_mmc_instruction_arxiv_text.json", "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_mmc_instruction_non-arxiv_text.json", "/share/project/zzy/120M_ds/zzy/filter_out/MathV360K-train_samples_all_tuning.json", "/share/project/zzy/120M_ds/zzy/filter_out/MMC-Alignment-mmc_chart_text_alignment_arxiv_text.json", "/share/project/zzy/120M_ds/zzy/filter_out/DocReason-docreason-26k.json", "/share/project/zzy/120M_ds/zzy/filter_out/allava_laion-ALLaVA-Instruct-LAION-4V.json", "/share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Caption-LAION-4V_dedup.json", "/share/project/zzy/120M_ds/zzy/filter_out/cocotext-cocotext-16k.json", "/share/project/zzy/120M_ds/zzy/filter_out/Docvqa_docvqa_train_10k.json", "/share/project/zzy/120M_ds/zzy/filter_out/Docvqa_docvqa_5k_v6.json", "/share/project/zzy/120M_ds/zzy/filter_out/Geoqa+_geoqa+.json", "/share/project/zzy/120M_ds/zzy/filter_out/DocDownstream-docdownstream-574k.json", "/share/project/zzy/120M_ds/zzy/filter_out/DocDownstream-docdownstream-val.json", "/share/project/zzy/120M_ds/zzy/filter_out/DocDownstream-docdownstream-train.json", "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-detail_23k.json", "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-conversation_58k.json", "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-llava_v1_5_mix665k.json", "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-llava_inst_80k.json", "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-complex_reasoning_77k.json", "/share/project/zsy/9_10/Cambrian-Finetune-10M/Cambrian10M.json", "/share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4o_60k_dedup.json", "/share/project/zzy/120M_ds/zzy/filter_out/Cambrian-Finetune-10M-data_engine_161k.json", "/share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4v_77k_dedup.json", "/share/project/zzy/120M_ds/zzy/filter_out/DocStruct4M--multi_grained_text_localization.json", "/share/project/zzy/120M_ds/zzy/filter_out/DocStruct4M-struct_aware_parse.json", "/share/project/zsy/8_29/dataset_after_deduplication_v2/MMDU/mmdu-45k_transformed_dedup.json", "/share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Caption-LAION-4V_dedup.json", "/share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Instruct-LAION-4V_dedup.json", "/share/projset/2024082801/lmms-lab/json_files/all_data_filter2.json"]
# files = ["/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_chart_release_update.json"]
# files = files[:12]
# files = files[12:23]
# files = files[23:34]
# files = files[34:]
# files = ["/share/project/zsy/9_12/out/Sharegpt4v-share4v_preferences.json", "/share/project/zsy/9_12/out/MMC-INST_chart_data_nonarxiv_summarization.json", "/share/project/zsy/9_12/out/MMC-Alignment-mmc_chart_text_alignment_arxiv_text.json"]

files = ["/share/project/zsy/9_6/synthdog_en_100k/synthdog_en_processed_new.json", "/share/project/zsy/9_6/synthdog_zh_100k/synthdog_zh_processed_new.json", "/share/project/zsy/9_6/ureader_tr_sft/ureader_tr_processed_new.json"]
files = ["/share/project/zsy/9_6/Evol-Instruct-GPT4-Turbo-143K/evol_instruct_processed.json", "/share/project/lijijie/tools/instruction_follow/0625/7M/7M_0712_math_plus_system_release_0802.json"]
files = ["/share/projset/Docmatix/HuggingFaceM4/Docmatix_20240911/docmatix.json"]
files = ["/share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/priority_smaple_50%_916_split_split.json"]
files = ["/share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/priority_smaple_50%_loss_1.5_917_split.json"]
files = ["/share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/priority_smaple_50%_loss_2_917_split_split.json"]
files = ["/share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/priority_smaple_50%_loss_2.5_917_split_split.json"]
files = ["/share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/mmvet_strengthen_0_4_919_split.json"]
files = ["/share/project/zhangjialing/R-error_analysis/error_analysis-train_0.5Prompt+Loss/error_analysis_MMVet/sample_1M_QA_ram-multi_datasets+allava+llava_660k_combine_split_split.json"]
files = ["/share/project/zhangjialing/R-error_analysis/error_analysis-train_0.5Prompt+Loss/instruction_lookup/MMVet-Tertiary+secondary_labels-merge+de-duplicate_format+prompt+loss_multi_datasets+allava+llava_660k_combine_split2.json"]
files = ["/share/project/zhangjialing/R-error_analysis/error_analysis-train_0.5Prompt+Loss/instruction_lookup/MMVet-sample_RAM_300k-counterpart_train_instruction_label-Tertiary+secondary_labels-merge+de-duplicate_format+prompt+loss_multi_datasets+allava+llava_660k_combine_split2.json"]
files = ["/share/project/zhangjialing/R-error_analysis/error_analysis-train_0.5Prompt+Loss/error_analysis_MMVet/sample_500k_QA_ram-multi_datasets+allava+llava_660k_combine_split4.json"]
files = ["/share/project/zhangjialing/R-error_analysis/error_analysis-train_0.5Prompt+Loss/instruction_lookup/MMVet-Tertiary_labels_500k-merge+de-duplicate_format+prompt+loss_multi_datasets+allava+llava_660k_combine_split.json"]
files = [
    # "/share/project/zzy/120M_ds/zzy/filter_out/LVIS-Instruct-lvis_instruct4v_cap_111k_dedup.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/LVIS-Instruct-lvis_instruct4v_vqa_111k_dedup.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-CC3M-Pretrain-595K-chat_dedup.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Visdial-Visdial-20k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Visdial-Visdial-123k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-sharegpt4v_v4_50k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-share4v_preference_sampled_50k_v5.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-sharegpt4v_instruct_gpt4-vision_cap100k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-sharegpt4v_v2_102k.json",
    # "/share/project/zsy/9_12/out/Sharegpt4v-share4v_preferences.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-share-captioner_coco_lcs_sam_1246k_1107.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Sharegpt4v-Sharegpt4v_mix665k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/STVQA_train_task_1.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/STVQA_stvqa-19k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_chart_release_update.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_chart_data_part3.json",
    # "/share/project/zsy/9_12/out/MMC-INST_chart_data_nonarxiv_summarization.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_mmc_instruction_arxiv_text.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/MMC-INST_mmc_instruction_non-arxiv_text.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/MathV360K-train_samples_all_tuning.json",
    # "/share/project/zsy/9_12/out/MMC-Alignment-mmc_chart_text_alignment_arxiv_text.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/DocReason-docreason-26k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/allava_laion-ALLaVA-Instruct-LAION-4V.json",
    # "/share/project/zsy/9_24/filter_out/allava_laion--ALLaVA-Caption-LAION-4V_dedup.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/cocotext-cocotext-16k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Docvqa_docvqa_train_10k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Docvqa_docvqa_5k_v6.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Geoqa+_geoqa+.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/DocDownstream-docdownstream-574k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/DocDownstream-docdownstream-val.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/DocDownstream-docdownstream-train.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-detail_23k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-conversation_58k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-llava_v1_5_mix665k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-llava_inst_80k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/LLaVA-Instruct-150K-complex_reasoning_77k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/Cambrian-Finetune-10M-data_engine_161k.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/DocStruct4M--multi_grained_text_localization.json",
    # "/share/project/zsy/9_24/filter_out/synthdog_en_100k--synthdog_en_processed_new.json",
    # "/share/project/zsy/9_24/filter_out/synthdog_zh_100k--synthdog_zh_processed_new.json",
    # "/share/project/zsy/9_24/filter_out/ureader_tr_sft--ureader_tr_processed_new.json",
    # "/share/project/zsy/9_24/filter_out/Cambrian-Finetune-10M--Cambrian10M.json",
    # "/share/project/zsy/9_24/filter_out/Cambrian-Finetune-10M--gpt4o_60k_dedup.json",
    # "/share/project/zsy/9_24/filter_out/Cambrian-Finetune-10M--gpt4v_77k_dedup.json",
    # "/share/project/zzy/120M_ds/zzy/filter_out/DocStruct4M-struct_aware_parse.json",
    "/share/project/zsy/9_26/filter_out/phash--allava_vflan-ALLaVA-Caption-LAION-4V.json",
    "/share/project/zsy/9_26/filter_out/phash--allava_vflan-ALLaVA-Instruct-LAION-4V.json"
]

# save_path = "/share/project/gushuhao/2.data/infinity_mm/stage1.5/911_v1"
save_path = "/share/projset/Docmatix/HuggingFaceM4/Docmatix_20240911/images"
save_path = "/share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/"
save_path = "/share/project/zsy/10_5/check/"
os.makedirs(save_path, exist_ok=True)
log_file = os.path.join(save_path, 'log.txt')
log = open(log_file, 'a')


# data_path = "/share/project/gushuhao/1.research/LLaVA-NeXT-main/llavaonevision-qwen1.5b-sft-stage1.5-sample1.5M/stage1.5.yaml"
# with open(data_path, "r") as file:
#     yaml_data = yaml.safe_load(file)
#     datasets = yaml_data.get("datasets")
#     dataset_paths = [dataset.get("json_path") for dataset in datasets]

# files = [dataset.get("json_path") for dataset in datasets]

for file in files:
    f = open(file)
    data = json.load(f)
    image, data = check_image(data)
    nul, data = check_null(data)
    con, data = check_con(data) 
    # ais, data = check_answer_in_question(data)
    # yes, data = check_yes(data)
    # fil, data = check_image_file(data)
    # short, data = check_short_question(data)
    data = add_image(data)
    
    name = file.split('/')[-1]
    print(name, image)
    # print(f"{name}, {image}, {nul}, {con}, {ais}, {fil}")
    # log.write(f"{name}, {image}, {nul}, {con}, {ais}, {fil}\n")
    # print(f"{name}, {short}")
    print(len(data))
    
    path = os.path.join(save_path, name + '.checked')
    g = open(path, 'w')
    json.dump(data, g, indent=4)
    
    
    