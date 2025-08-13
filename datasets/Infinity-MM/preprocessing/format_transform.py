import json
from tqdm import tqdm
file_path = [
    '/the/jsons/that/you/want/to/process'
]
data_list = []
for file in tqdm(file_path):
    with open(file, 'r') as f:
        data = json.load(f)
        data_list.append(data)
        
def convert_conversations_2_conversations_new(sample):
    sample['conversations_new'] = []
    if 'conversations' not in sample or sample['conversations'] is None:
        return
    for i in range(len(sample['conversations'])):
        if i%2 == 0:
            sample['conversations_new'].append({'role': 'user', 'content': []})
        else:
            sample['conversations_new'].append({'role': 'assistant', 'content': []})
        if i == 0 and 'image' in sample:
            image = sample['image']
            if type(image) == str:
                image = [image]
            for img in image:
                sample['conversations_new'][0]['content'].append(
                    {'type': 'image', 'image': img, "resized_height": 500, "resized_width": 500,})
        if 'image' in sample:
            if len(sample['image']) == 1:
                sample['conversations_new'][i]['content'].append(
                    {'type': 'text', 'text': sample['conversations'][i]['value'].replace('<image>', '').strip()})
            else:
                sample['conversations_new'][i]['content'].append(
                    {'type': 'text', 'text': sample['conversations'][i]['value']})
                for id in range(1, len(image)):
                    sample['conversations_new'][i]['content'][-1]['text'] = sample['conversations_new'][i]['content'][-1]['text'].replace(f'Image{id}: <image>;', '')
                sample['conversations_new'][i]['content'][-1]['text'] = sample['conversations_new'][i]['content'][-1]['text'].replace(f'Image{len(image)}: <image>.', '')
                sample['conversations_new'][i]['content'][-1]['text'] = sample['conversations_new'][i]['content'][-1]['text'].strip()
for data in data_list:
    for sample in tqdm(data):
        convert_conversations_2_conversations_new(sample)

for data in data_list:
    data = [sample for sample in data if 'conversations' in sample and sample['conversations'] is not None]
    
for i in tqdm(range(len(data_list))):
    with open('/the/path/you/want/to/save/json/after/transforming/'+file_path[i].split('/')[-1], 'w') as f:
        json.dump(data_list[i], f, indent=4)