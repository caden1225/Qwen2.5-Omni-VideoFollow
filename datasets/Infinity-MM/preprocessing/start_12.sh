HOSTFILE=/share/project/zsy/code/compute_loss/hostfile.a100.17-28
START_ID=0
LINE_COUNT=0

while IFS= read -r line; do

    
    # url=$(echo $line | cut -d':' -f1)
    url=$(echo $line | awk '{print $1}')   # 使用空格作为分隔符提取IP地址
    # url=172.24.222.182

    # 在后台执行 ssh，并且执行完毕后断开
    ssh $url "bash /share/project/zhaohuxing/qwen2_vl_test/start_compute.sh $START_ID" &
    
    START_ID=$((START_ID+8))
    LINE_COUNT=$((LINE_COUNT+1))

    # 启动一个就break
    # if [ $LINE_COUNT -eq 1 ]; then
    #     break
    # fi
done < "$HOSTFILE"

# 等待所有后台任务完成
wait