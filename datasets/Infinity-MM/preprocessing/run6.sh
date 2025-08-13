image_dirs_all=(
    /share/project/zhangjialing/datasets_json/visdial/dataset_after_deduplication/visdial-20k_dedup.json
    /share/project/zhangjialing/datasets_json/visdial/dataset_after_deduplication/visdial_123k_out_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_v4_50k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_sampled_50k_v5_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_top_20k_v6_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_laion4v_20k_v1_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_v3_50k_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_v1_12k_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preferences_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_top_50k_v7_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_sampled_20k_v6_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107_save_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_250000_out_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian737k_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/docreason-26k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Docvqa/Docvqa_docvqa_train_10k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Docvqa/Docvqa_docvqa_5k_v6_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+_geoqa+_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+_geoqa-plus-augmented-72k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+3_geoqa-plus-original-12k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_save_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_train_task_1_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_train_task_2_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_train_task_3_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_stvqa-19k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocDownstream/docdownstream-574k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocDownstream/docdownstream-val_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocDownstream/docdownstream-train_dedup.json
    /share/project/hzc/8_30/MMC-INST/MMC-INST_arxiv_data_dedup.json
    /share/project/hzc/8_30/MMC-INST/MMC-INST_chart_release_update_dedup.json
    /share/project/hzc/8_30/MMC-INST/MMC-INST_chart_data_part3_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_out_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_300000_out_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_110020_out_new_dedup.json
    /share/project/hzc/8_30/MMC-INST/mm_inst_99_out_new_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/detail_23k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/conversation_58k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_665k_out_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_80k_out_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_150k_out_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_77k_out_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4o_60k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/data_engine_161k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4v_77k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian10M_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian7M_sample_100k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian7M_withsystemprompt_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian7M_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MathV360K/train_samples_all_tuning_with_source_and_conversations_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMDU/benchmark_transformed_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMDU/mmdu-45k_transformed_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocStruct4M/multi_grained_text_localization_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocStruct4M/val_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocStruct4M/struct_aware_parse_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMC-Alignment/mmc_chart_text_alignment_arxiv_text_with_source_and_conversations_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMC-benchmark/mmc_benchmark_text_with_source_and_conversations_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/detailed_explanation_dedup.json
    /share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Caption-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Instruct-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Instruct-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Caption-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Caption-LAION-4V-422k_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Instruct-LAION-4V-422k_dedup.json
    /share/project/yt/mmdataset-filter/refcocog/instances_dedup.json
    /share/project/yt/mmdataset-filter/refcoco+/instances_dedup.json
    /share/project/yt/mmdataset-filter/refcoco/instances_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/COCO_Text_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/cocotext-16k_dedup.json
)


image_dirs1=(
    /share/project/zhangjialing/datasets_json/visdial/dataset_after_deduplication/visdial-20k_dedup.json
    /share/project/zhangjialing/datasets_json/visdial/dataset_after_deduplication/visdial_123k_out_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_v4_50k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_sampled_50k_v5_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_top_20k_v6_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_laion4v_20k_v1_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_v3_50k_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_v1_12k_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preferences_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_top_50k_v7_save_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share4v_preference_sampled_20k_v6_save_dedup.json
)


image_dirs6=(
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107_save_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_250000_out_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian737k_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/docreason-26k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Docvqa/Docvqa_docvqa_train_10k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Docvqa/Docvqa_docvqa_5k_v6_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+_geoqa+_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+_geoqa-plus-augmented-72k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+3_geoqa-plus-original-12k_dedup.json
)


image_dirs2=(
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_save_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_train_task_1_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_train_task_2_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_train_task_3_dedup.json
    /share/project/hzc/8_30/STVQA/STVQA_stvqa-19k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocDownstream/docdownstream-574k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocDownstream/docdownstream-val_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocDownstream/docdownstream-train_dedup.json
    /share/project/hzc/8_30/MMC-INST/MMC-INST_arxiv_data_dedup.json
    /share/project/hzc/8_30/MMC-INST/MMC-INST_chart_release_update_dedup.json
    /share/project/hzc/8_30/MMC-INST/MMC-INST_chart_data_part3_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_out_dedup.json
)


image_dirs3=(
    /share/project/hzc/8_30/MMC-INST/mmc_inst_300000_out_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_110020_out_new_dedup.json
    /share/project/hzc/8_30/MMC-INST/mm_inst_99_out_new_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/detail_23k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/conversation_58k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_665k_out_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_80k_out_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_150k_out_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/LLaVA-Instruct-150K/llava_inst_77k_out_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4o_60k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/data_engine_161k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4v_77k_dedup.json    
)



image_dirs4=(
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian10M_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian7M_sample_100k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian7M_withsystemprompt_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian7M_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MathV360K/train_samples_all_tuning_with_source_and_conversations_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMDU/benchmark_transformed_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMDU/mmdu-45k_transformed_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocStruct4M/multi_grained_text_localization_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocStruct4M/val_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/DocStruct4M/struct_aware_parse_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMC-Alignment/mmc_chart_text_alignment_arxiv_text_with_source_and_conversations_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMC-benchmark/mmc_benchmark_text_with_source_and_conversations_dedup.json
)

image_dirs5=(
    /share/project/yt/mmdataset-filter/cocotext/detailed_explanation_dedup.json
    /share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Caption-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Instruct-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Instruct-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Caption-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Caption-LAION-4V-422k_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Instruct-LAION-4V-422k_dedup.json
    /share/project/yt/mmdataset-filter/refcocog/instances_dedup.json
    /share/project/yt/mmdataset-filter/refcoco+/instances_dedup.json
    /share/project/yt/mmdataset-filter/refcoco/instances_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/COCO_Text_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/cocotext-16k_dedup.json
)

image_dirs_extra=(
    /share/project/yt/mmdataset-filter/cocotext/cocotext-16k_dedup.json
    /share/project/yt/mmdataset-filter/cocotext/COCO_Text_dedup.json
)

image_dirs7=(
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107_save_dedup.json
    /share/project/hzc/8_30/MMC-INST/mmc_inst_250000_out_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/Cambrian737k_dedup.json
    
)

image_dirs8=(
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+_geoqa+_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+_geoqa-plus-augmented-72k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Geoqa+_geoqa+/Geoqa+3_geoqa-plus-original-12k_dedup.json
)

image_dirs9=(
    # /share/project/yt/mmdataset-filter/cocotext/docreason-26k_dedup.json
    # /share/project/zsy/9_2/dataset_after_deduplication_v2/Docvqa/Docvqa_docvqa_train_10k_dedup.json
    # /share/project/zsy/9_2/dataset_after_deduplication_v2/Docvqa/Docvqa_docvqa_5k_v6_dedup.json
    /share/project/zzy_jjt_hzc/images
)

image_dirs10=(
    /share/project/zsy/9_10/Cambrian-Finetune-10M/Cambrian10M.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4o_60k_dedup.json
    /share/project/zsy/9_2/dataset_after_deduplication_v2/Cambrian-Finetune-10M/gpt4v_77k_dedup.json
    /share/project/zsy/8_29/dataset_after_deduplication_v2/MMDU/mmdu-45k_transformed_dedup.json
    /share/project/zsy/9_6/synthdog_en_100k/synthdog_en_processed_new.json
    /share/project/zsy/9_6/synthdog_zh_100k/synthdog_zh_processed_new.json
    /share/project/zsy/9_6/ureader_tr_sft/ureader_tr_processed_new.json
    #/share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Caption-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_vflan/ALLaVA-Instruct-LAION-4V_dedup.json
    /share/project/yt/mmdataset-filter/allava_laion/ALLaVA-Caption-LAION-4V_dedup.json
)

image_dirs11=(
    /share/project/zsy/9_25/phash/merge_priority_sample_10M-EVE-Pretrain-33M_laion_dedump_openimages_sam_safe.json
)

image_dirs12=(
    /share/project/zsy/9_25/phash/allava_vflan-ALLaVA-Caption-LAION-4V.json
    /share/project/zsy/9_25/phash/allava_vflan-ALLaVA-Instruct-LAION-4V.json
)

image_dirs13=(
    /share/project/zsy/10_15/stage1.5b_only_image/all_data_filter2.json
)

image_dirs14=(
/share/project/zsy/10_9/filtered/gpt4v_77k.json
/share/project/zsy/10_21/stage2/json/only_image/original_train.json
/share/project/zsy/10_9/filtered/priority_smaple_50%_loss_2_917_split_split_noyes.checked.json
/share/project/zsy/10_21/stage2/json/merge+de-duplicate_format+prompt+loss5_multi_datasets+allava+llava_660k_OCR_deduplicate_loss_less_2.5_len_less_99.json
/share/project/zsy/10_9/filtered/DenseFusion-4V-100K.abs.json
/share/project/zsy/10_9/filtered/gpt4o_60k.json
/share/project/zsy/10_21/stage2/json/only_image/all_data_filter2_sample_10.json
/share/project/zsy/10_21/stage2/json/ocr_vqa_unique_prompt.json
)

image_dirs15=(
/share/project/zsy/10_24/phash/ocr_split_39.json
)

# 迭代每个图片路径名
for image_dir in "${image_dirs15[@]}"; do
    torchrun --nproc_per_node=8 /share/project/zzy_jjt_hzc/ram_pipeline/ram_label.py "$image_dir" 
    python /share/project/zzy_jjt_hzc/ram_pipeline/merge.py "$image_dir"
    python /share/project/zzy_jjt_hzc/ram_pipeline/ram_filter.py "$image_dir" 
done

wait 

echo "All processes completed."