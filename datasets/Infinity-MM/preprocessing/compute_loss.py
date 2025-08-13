# import pdb
# pdb.set_trace()
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



    with open(f"/share/project/zhaohuxing/qwen2_vl_test/stage2_data_1/chunk_{args.split_id}.json", "r") as f:
        print('loading ' + f"/share/project/zhaohuxing/qwen2_vl_test/stage2_data_1/chunk_{args.split_id}.json")
        data = json.load(f)


    processor = AutoProcessor.from_pretrained("./qwen2_vl_7b")
    tokenizer = AutoTokenizer.from_pretrained("./qwen2_vl_7b")
    split_id = args.split_id


    models = []
    threads = 1
    device=f'cuda:{args.device}'
    for i in range(threads):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "./qwen2_vl_7b",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        models.append(model)


    import numpy as np


    def thread(model, data, device, split_id):
        print('Starting split {} device {}'.format(split_id, device))
        batch_size = 3
        answer_losses = np.zeros(len(data))
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:min(i+batch_size, len(data))]
            try:
                text = []
                image = []
                for b in batch:
                    # print(b['conversations_new'])
                    text.append(processor.apply_chat_template(b["conversations_new"], tokenize=False, add_generation_prompt=True))
                    
                convs = [b["conversations_new"] for b in batch]    
                image_inputs, video_inputs = process_vision_info(convs)
                inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                )

                labels = []
                for in_ids in inputs.input_ids:
                    label = in_ids.clone()
                    # print(in_ids.shape, label.shape)
                    seg_id = tokenizer.encode('<|im_end|>')

                    # 找到所有的 <|im_end|> 在 label 中的位置
                    im_end_pos = (label == seg_id[0]).nonzero(as_tuple=True)[0]
                    # print(im_end_pos)
                    # system <|im_end|> user <|im_end|> assistant <|im_end|> ......
                    # 跳过第一个 <|im_end|>，因为是system
                    im_end_pos = im_end_pos[1:]
                    pre_id = 0
                    for idxx in range(len(im_end_pos)):
                        # mask掉非assistant的部分
                        if idxx%2 == 0:
                            label[pre_id:im_end_pos[idxx]+1] = -100
                            pre_id = im_end_pos[idxx]+1
                        else:
                            pre_id = im_end_pos[idxx]+1

                    labels.append(label)
                # print(labels)
                inputs["labels"] = torch.stack(labels)
                cache_position = torch.tensor([i for i in range(len(inputs.input_ids[0]))])
                inputs = inputs.to(device)
                inputs["cache_position"] = cache_position
                
                prepared_inputs = model.prepare_inputs_for_generation(**inputs)
                prepared_inputs['labels'] = inputs['labels']
                with torch.no_grad():
                    outputs = model(**prepared_inputs)
                    print(model.losses)
                answer_losses[i:i+len(batch)] = model.losses
                np.save(f"/share/project/zhaohuxing/qwen2_vl_test/stage2_data_1_loss/answer_losses_{split_id}.npy", answer_losses)
            except Exception as e:
                print(e)
                print("error")
                # 清理GPU
                torch.cuda.empty_cache()

    for i in range(threads):
        thread(models[i], data, device, split_id)

    print('excute_done')
        

