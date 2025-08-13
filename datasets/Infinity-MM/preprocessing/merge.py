
import json
from tqdm import tqdm
import os

import sys


input = sys.argv[1]
input_ls = input.split("/")
test_set = "--".join(input_ls[-2:]).split(".")[0]
#test_set = 'allava_vflan--ALLaVA-Caption-LAION-4V_dedup'
save_dir = f"./res-test_set/{test_set}"
os.makedirs(save_dir, exist_ok=True)

final_file = os.path.join(save_dir, f"ram-{test_set}.jsonl")

with open(final_file, 'w') as outfile:
    for i in range(0, 8):
        with open(os.path.join(save_dir, f"ram-{test_set}-{i}.jsonl"), 'r') as infile:
            for line in infile:
                outfile.write(json.dumps(json.loads(line), ensure_ascii=False) + "\n")
            
    print(f"Files merged into {final_file}")
    
    try:
        for i in range(0, 8):
            name = os.path.join(save_dir, f"ram-{test_set}-{i}.jsonl")
            os.remove(name)
            print(f"File {name} deleted successfully.")
    except FileNotFoundError:
        print(f"File {name} not found.")
    except Exception as e:
        print(f"Error occurred while deleting file {name}: {e}")