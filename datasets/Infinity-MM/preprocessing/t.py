import re 
import json


file_names = [
# '/share/project/yt/stage1.5b_data/Docmatix_20240911--docmatix.json',
# '/share/project/yt/stage1.5b_data/all_data_filter2_only_image.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M--Cambrian10M.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M--gpt4o_60k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M--gpt4v_77k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_Cambrian-Finetune-10M-data_engine_241k.json',
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
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_cocotext-cocotext-24k.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_synthdog_en_100k--synthdog_en_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_synthdog_zh_100k--synthdog_zh_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/loss_sampled_data/sampled_ureader_tr_sft--ureader_tr_processed_new.json',
# '/share/project/yt/stage1.5_pretrain_data/all/allava_laion--ALLaVA-Caption-LAION-4V_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/allava_laion-ALLaVA-Instruct-LAION-4V.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M--Cambrian10M.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M--gpt4o_60k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M--gpt4v_77k_dedup.json',
# '/share/project/yt/stage1.5_pretrain_data/all/Cambrian-Finetune-10M-data_engine_241k.json',
# '/share/project/yt/stage1.5_pretrain_data/all/cocotext-cocotext-24k.json',
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

data = []
for i in file_names:
    with open(i, "r") as f:
        print('loading ' + i)
        data.extend(json.load(f))

chunk_size = len(data) // 16
remainder = len(data) % 16

chunks = []
start = 0
for i in range(16):
    end = start + chunk_size + (1 if i < remainder else 0)  # Distribute the remainder
    chunks.append(data[start:end])
    start = end

from tqdm import tqdm
for i, chunk in tqdm(enumerate(chunks)):
    with open(f"/share/project/zhaohuxing/qwen2_vl_test/stage2_data_1/chunk_{i}.json", "w") as f:
        print(len(chunk))
        json.dump(chunk, f)

