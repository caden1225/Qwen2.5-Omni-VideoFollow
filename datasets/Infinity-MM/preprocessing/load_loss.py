# 读取loss，qaloss/answer_losses_0.npy，有24个文件

import numpy as np
import json
from tqdm import tqdm


file_path = [
# '/share/project/yt/mm_pretrain_data/allava_laion-ALLaVA-Instruct-LAION-4V.json',
# '/share/project/yt/mm_pretrain_data/DocReason-docreason-26k.json',
# '/share/project/yt/mm_pretrain_data/LLaVA-CC3M-Pretrain-595K-chat_dedup.json',
# '/share/project/yt/mm_pretrain_data/LVIS-Instruct-lvis_instruct4v_cap_111k_dedup.json',
# '/share/project/yt/mm_pretrain_data/LVIS-Instruct-lvis_instruct4v_vqa_111k_dedup.json',
# '/share/project/yt/mm_pretrain_data/MathV360K-train_samples_all_tuning.json',
# '/share/project/yt/mm_pretrain_data/MMC-Alignment-mmc_chart_text_alignment_arxiv_text.json',
# '/share/project/yt/mm_pretrain_data/MMC-INST_chart_data_nonarxiv_summarization.json',
# '/share/project/yt/mm_pretrain_data/MMC-INST_chart_data_part3.json',
# '/share/project/yt/mm_pretrain_data/MMC-INST_chart_release_update.json',
# '/share/project/yt/mm_pretrain_data/MMC-INST_mmc_instruction_arxiv_text.json',
# '/share/project/yt/mm_pretrain_data/MMC-INST_mmc_instruction_non-arxiv_text.json',
# '/share/project/yt/mm_pretrain_data/Sharegpt4v-share-captioner_coco_lcs_sam_1246k_1107.json',
# '/share/project/yt/mm_pretrain_data/Sharegpt4v-share4v_preference_sampled_50k_v5.json',
# '/share/project/yt/mm_pretrain_data/Sharegpt4v-share4v_preferences.json',
# '/share/project/yt/mm_pretrain_data/Sharegpt4v-sharegpt4v_instruct_gpt4-vision_cap100k.json',
# '/share/project/yt/mm_pretrain_data/Sharegpt4v-Sharegpt4v_mix665k.json',
# '/share/project/yt/mm_pretrain_data/Sharegpt4v-sharegpt4v_v2_102k.json',
# '/share/project/yt/mm_pretrain_data/Sharegpt4v-sharegpt4v_v4_50k.json',
# '/share/project/yt/mm_pretrain_data/STVQA_stvqa-19k.json',
# '/share/project/yt/mm_pretrain_data/STVQA_train_task_1.json',
# '/share/project/yt/mm_pretrain_data/Visdial-Visdial-20k.json',
# '/share/project/yt/mm_pretrain_data/Visdial-Visdial-123k.json',
# '/share/project/yt/mm_pretrain_data1/Cambrian-Finetune-10M--Cambrian10M.json',
# '/share/project/yt/mm_pretrain_data1/Cambrian-Finetune-10M--gpt4o_60k_dedup.json',
# '/share/project/yt/mm_pretrain_data1/Cambrian-Finetune-10M--gpt4v_77k_dedup.json',
# '/share/project/yt/mm_pretrain_data1/Cambrian-Finetune-10M-data_engine_161k.json',
# '/share/project/yt/mm_pretrain_data1/DocDownstream-docdownstream-574k.json',
# '/share/project/yt/mm_pretrain_data1/DocDownstream-docdownstream-train.json',
# '/share/project/yt/mm_pretrain_data1/DocDownstream-docdownstream-val.json',
# '/share/project/yt/mm_pretrain_data1/DocStruct4M--multi_grained_text_localization.json',
# '/share/project/yt/mm_pretrain_data1/DocStruct4M-struct_aware_parse.json',
# '/share/project/yt/mm_pretrain_data1/Docvqa_docvqa_5k_v6.json',
# '/share/project/yt/mm_pretrain_data1/Docvqa_docvqa_train_10k.json',
# '/share/project/yt/mm_pretrain_data1/Geoqa+_geoqa+.json',
# '/share/project/yt/mm_pretrain_data1/LLaVA-Instruct-150K-complex_reasoning_77k.json',
# '/share/project/yt/mm_pretrain_data1/LLaVA-Instruct-150K-conversation_58k.json',
# '/share/project/yt/mm_pretrain_data1/LLaVA-Instruct-150K-detail_23k.json',
# '/share/project/yt/mm_pretrain_data1/LLaVA-Instruct-150K-llava_inst_80k.json',
# '/share/project/yt/mm_pretrain_data1/LLaVA-Instruct-150K-llava_v1_5_mix665k.json',
# '/share/project/yt/mm_pretrain_data1/allava_laion--ALLaVA-Caption-LAION-4V_dedup.json',
# '/share/project/yt/mm_pretrain_data1/cocotext-cocotext-16k.json',
# '/share/project/yt/mm_pretrain_data1/synthdog_en_100k--synthdog_en_processed_new.json',
# '/share/project/yt/mm_pretrain_data1/synthdog_zh_100k--synthdog_zh_processed_new.json',
# '/share/project/yt/mm_pretrain_data1/ureader_tr_sft--ureader_tr_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/all/allava_laion--ALLaVA-Caption-LAION-4V_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/allava_laion-ALLaVA-Instruct-LAION-4V.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M--Cambrian10M.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M--gpt4o_60k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M--gpt4v_77k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M-data_engine_161k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/cocotext-cocotext-16k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/DocDownstream-docdownstream-574k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/DocDownstream-docdownstream-train.json',
# '/share/project/yt/stage1.5_pretrain_data/all/DocDownstream-docdownstream-val.json',
# '/share/project/yt/stage1.5_pretrain_data/all/DocReason-docreason-26k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/DocStruct4M--multi_grained_text_localization.json',
# '/share/project/yt/stage1.5_pretrain_data/all/DocStruct4M-struct_aware_parse.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Docvqa_docvqa_5k_v6.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Docvqa_docvqa_train_10k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Geoqa+_geoqa+.json',
# '/share/project/yt/stage1.5_pretrain_data/all/LLaVA-CC3M-Pretrain-595K-chat_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/LVIS-Instruct-lvis_instruct4v_cap_111k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/LVIS-Instruct-lvis_instruct4v_vqa_111k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/MathV360K-train_samples_all_tuning.json',
# '/share/project/yt/stage1.5_pretrain_data/all/MMC-Alignment-mmc_chart_text_alignment_arxiv_text.json',
# '/share/project/yt/stage1.5_pretrain_data/all/MMC-INST_chart_data_nonarxiv_summarization.json',
# '/share/project/yt/stage1.5_pretrain_data/all/MMC-INST_chart_data_part3.json',
# '/share/project/yt/stage1.5_pretrain_data/all/MMC-INST_chart_release_update.json',
# '/share/project/yt/stage1.5_pretrain_data/all/MMC-INST_mmc_instruction_arxiv_text.json',
# '/share/project/yt/stage1.5_pretrain_data/all/MMC-INST_mmc_instruction_non-arxiv_text.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Sharegpt4v-share-captioner_coco_lcs_sam_1246k_1107.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Sharegpt4v-share4v_preference_sampled_50k_v5.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Sharegpt4v-share4v_preferences.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Sharegpt4v-sharegpt4v_instruct_gpt4-vision_cap100k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Sharegpt4v-Sharegpt4v_mix665k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Sharegpt4v-sharegpt4v_v2_102k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Sharegpt4v-sharegpt4v_v4_50k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/STVQA_stvqa-19k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/STVQA_train_task_1.json',
# '/share/project/yt/stage1.5_pretrain_data/all/synthdog_en_100k--synthdog_en_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/all/synthdog_zh_100k--synthdog_zh_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/all/ureader_tr_sft--ureader_tr_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Visdial-Visdial-20k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Visdial-Visdial-123k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M--Cambrian10M.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M--gpt4o_60k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M--gpt4v_77k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M-data_engine_161k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_DocDownstream-docdownstream-574k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_DocDownstream-docdownstream-train.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_DocDownstream-docdownstream-val.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_DocReason-docreason-26k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_DocStruct4M--multi_grained_text_localization.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_DocStruct4M-struct_aware_parse.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Docvqa_docvqa_5k_v6.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Docvqa_docvqa_train_10k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Geoqa+_geoqa+.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_LLaVA-CC3M-Pretrain-595K-chat_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_LVIS-Instruct-lvis_instruct4v_cap_111k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_LVIS-Instruct-lvis_instruct4v_vqa_111k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_MMC-Alignment-mmc_chart_text_alignment_arxiv_text.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_MMC-INST_chart_data_nonarxiv_summarization.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_MMC-INST_chart_data_part3.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_MMC-INST_chart_release_update.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_MMC-INST_mmc_instruction_arxiv_text.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_MMC-INST_mmc_instruction_non-arxiv_text.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_MathV360K-train_samples_all_tuning.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_STVQA_stvqa-19k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_STVQA_train_task_1.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Sharegpt4v-Sharegpt4v_mix665k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Sharegpt4v-share-captioner_coco_lcs_sam_1246k_1107.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Sharegpt4v-share4v_preference_sampled_50k_v5.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Sharegpt4v-share4v_preferences.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Sharegpt4v-sharegpt4v_instruct_gpt4-vision_cap100k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Sharegpt4v-sharegpt4v_v2_102k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Sharegpt4v-sharegpt4v_v4_50k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Visdial-Visdial-123k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Visdial-Visdial-20k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_allava_laion--ALLaVA-Caption-LAION-4V_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_allava_laion-ALLaVA-Instruct-LAION-4V.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_cocotext-cocotext-16k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_synthdog_en_100k--synthdog_en_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_synthdog_zh_100k--synthdog_zh_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_ureader_tr_sft--ureader_tr_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/all/phash--allava_vflan-ALLaVA-Caption-LAION-4V.json',
# '/share/project/yt/stage1.5_pretrain_data/all/phash--allava_vflan-ALLaVA-Instruct-LAION-4V.json',
# '/share/project/yt/stage1.5b_data/Docmatix_20240911--docmatix.json',
# '/share/project/yt/stage1.5b_data/all_data_filter2_only_image.json',
# '/share/project/yt/stage2_data/LVIS-Instruct-lvis_instruct4v_cap_111k_dedup.json',
# '/share/project/yt/stage2_data/LVIS-Instruct-lvis_instruct4v_vqa_111k_dedup.json',
# '/share/project/yt/stage2_data/filtered--DenseFusion-4V-100K.json',
# '/share/project/yt/stage2_data/filtered--gpt4o_60k.json',
# '/share/project/yt/stage2_data/filtered--gpt4v_77k.json',
# '/share/project/yt/stage2_data/filtered--priority_smaple_50%_loss_2_917_split_split_noyes.json',
# '/share/project/yt/stage2_data/json--ocr_vqa_unique_prompt.json',
# '/share/project/yt/stage2_data/only_image--all_data_filter2_sample_10.json',
# '/share/project/yt/stage2_data/only_image--original_train.json'
'/share/project/yt/stage2_data_1/phash--ocr_split_39.json',
'/share/project/yt/stage2_data_1/phash--zh_filtered_translated.json'

]


# 不拼在一起的
data_list = []
for file in tqdm(file_path, desc="Processing the original file"):
    with open(file, 'r') as f:
        data = json.load(f)
        print(len(data))
        data_list.append(data)

# 全拼一起
all_data = []
for data in data_list:
    all_data.extend(data)


# 全部的loss拼一起
losses = []
for i in tqdm(range(16), desc="Combining all loss"):
    loss = np.load(f'/share/project/zhaohuxing/qwen2_vl_test/stage2_data_1_loss/answer_losses_{i}.npy')
    losses.extend(loss)

print('all_data', len(all_data), 'losses', len(losses))


for i in range(len(losses)):
    all_data[i]['qw2vl_loss'] = losses[i]


# 拆分回去
datas = []
cur_len = 0
for data in tqdm(data_list, desc="Split the data"):
    datas.append(all_data[cur_len:cur_len+len(data)])
    cur_len += len(data)
for i in datas:
    print(len(i))

# 保存：
for i in tqdm(range(len(file_path)), desc="Save each file"):
    with open('/share/project/zhaohuxing/qwen2_vl_test/stage2_data_1_with_loss/'+file_path[i].split('/')[-1], 'w') as f:
        json.dump(datas[i], f, ensure_ascii=False, indent=4)
        print(len(datas[i]))
    
