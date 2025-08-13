#!/bin/bash

#cd /share/project/zsy/9_19/phash

source /share/project/zzy/venv/bin/activate

PYTHON_SCRIPT="/share/project/zsy/code/phash_only.py"

image_paths1=(
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MMBench_DEV_EN/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MMBench_DEV_EN_V11/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MMStar/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MMMU_DEV_VAL/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/AI2D_TEST/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/mm-vet/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MathVista_MINI/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/OCRBench/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/ChartQA_TEST/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/TextVQA_VAL/images
/share/project/zhangjialing/RAM+sample/benchmark/OCRVQA_TESTCORE/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/HallusionBench/images
)

image_paths2=(
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/br_score_all_ed.json
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/com_score_all_ed.json
/share/project/zsy/9_11/question_format/output_v1_filtered.json
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/sho_score_all_ed.json      #前四个是multidatasets中的
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/comanswer.json
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/branswer.json
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/shoanswer.json
/share/project/zsy/9_11/question_format/allava/choanswer_filtered.json                                      #中间四个是allava
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/branswer_second_batch.json
#/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/comanswer_second_batch.json
/share/project/zhangjialing/Q+R+A_json/Final_Harmonized_Format_Conversion_Result/score_filtered_first_batch.json
/share/project/zsy/9_11/question_format/choanswer_second_batch_filtered.json                                #最后四个是llava_660k
)

image_paths3=(
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/DocVQA/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/InfoVQA_TEST/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/OCR
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/artwork
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/celebrity
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/code_reasoning
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/color
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/commonsense_reasoning
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/count
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/existence
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/landmark
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/numerical_calculation
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/position
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/posters
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/scene
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MME/images/text_translation
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MMT-Bench_ALL/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MMVet/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MTVQA_TEST/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/MathVision/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/OCRVQA_TESTCORE/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/RealWorldQA/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/VCR_EN_EASY_ALL/images
/share/project/zhangjialing/benchmark/priority_sample_benchmark_image/VCR_ZH_EASY_ALL/images
)

image_paths4=(
# /share/project/zhangjialing/Q+R+A_json/merge+de-duplicate_format+prompt+loss/merge+de-duplicate_format+prompt+loss5_multi_datasets+allava+llava_660k_OCR_deduplicate_loss_less_2.5_len_less_99.json
# /share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/priority_smaple_50%_loss_2_917_split_split_noyes.checked.json
# /share/projset/mmdatasets-gpt4/DenseFusion-4V-100K.abs.json
# /share/project/zsy/9_2/image_update/Cambrian-Finetune-10M/gpt4v_77k.json
# /share/project/zsy/9_2/image_update/Cambrian-Finetune-10M/gpt4o_60k.json
# /share/project/zzy/120M_ds/zzy/filter_out/LVIS-Instruct-lvis_instruct4v_vqa_111k_dedup.json
# /share/project/zzy/120M_ds/zzy/filter_out/LVIS-Instruct-lvis_instruct4v_cap_111k_dedup.json
/share/projset/2024082801/lmms-lab/json_files/all_data_filter2_sample_10.json
# /share/project/zpf/code/LLaVA-copy/playground/data/gpt4_600k_data/ocr_vqa/ocr_vqa_unique_prompt.json
)

image_paths5=(
/share/project/zsy/10_9/filtered/gpt4v_77k.json
/share/project/zsy/10_21/stage2/json/only_image/original_train.json
/share/project/zsy/10_9/filtered/priority_smaple_50%_loss_2_917_split_split_noyes.checked.json
/share/project/zsy/10_21/stage2/json/merge+de-duplicate_format+prompt+loss5_multi_datasets+allava+llava_660k_OCR_deduplicate_loss_less_2.5_len_less_99.json
/share/project/zsy/10_9/filtered/DenseFusion-4V-100K.abs.json
/share/project/zsy/10_9/filtered/gpt4o_60k.json
/share/project/zsy/10_21/stage2/json/only_image/all_data_filter2_sample_10.json
/share/project/zsy/10_21/stage2/json/ocr_vqa_unique_prompt.json
)

image_paths6=(
/share/project/zsy/10_24/json/only_image/zh_filtered_translated.json
/share/project/yt/code/1021/ocr_split_39.json
)


# 迭代每个图片目录
for image_path in "${image_paths6[@]}"; do
    # 运行Python脚本，并将图片目录作为参数传递
    python $PYTHON_SCRIPT "$image_path" &
done

wait 

echo "All processes completed."