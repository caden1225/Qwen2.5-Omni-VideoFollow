cd /share/project/zhaohuxing/qwen2_vl_test
# source /share/project/zhaohuxing/anaconda3/bin/activate

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python compute_loss.py\
#     --split_id 13\
#     --device 5\

# 8 GPUs，并行执行,但每启动一个进程后等待20s

#!/bin/bash

# Set the number of GPUs available
NUM_GPUS=8
START_ID=$1

# Loop through each GPU and start a process
for ((i=0; i<$NUM_GPUS; i++)); do
    # Run the script for each split on a different GPU
    # --split_id will match the GPU number
    CUDA_VISIBLE_DEVICES=$i /share/project/zhaohuxing/anaconda3/bin/python compute_loss.py --split_id $((i + START_ID)) --device 0 &
done

# Wait for all processes to complete
wait

echo "All processes finished."



