import argparse
import re
from datasets import load_dataset, Dataset

'''
Filter SFT datasets to remove foreign language characters and references to AI companies and models. 
NB: all datasets were deduplicated before filtering. 
'''

filter_strings = [
    "OpenAI", "Open AI",
    "ChatGPT", "Chat GPT",
    "GPT-3", "GPT3", "GPT 3",
    "GPT-4", "GPT4", "GPT 4",
    "GPT-3.5", "GPT3.5", "GPT 3.5",
    "BingChat", "Bing Chat",
    "LAION",
    "Open Assistant", "OpenAssistant",
    "BARD", "PaLM",
    "Gemini", "Gemma",
    "Google AI",
    "Anthropic", "Claude",
    "LLaMA",
    "Meta AI",
    "Mixtral", "Mistral", "NVIDIA", "Nemotron", "Allen AI", "Olmo, Smol"
]
model_pattern = re.compile("|".join(map(re.escape, filter_strings)), re.IGNORECASE)

rpattern = re.compile(r'[А-Яа-яЁё]') #Russian characters

cpattern = re.compile(r'[\u4E00-\u9FFF]') #Chinese characters

apattern = re.compile(r'[\u0600-\u06FF]') #Arabic characters


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()


saved_dialogues = args.output_file

dialogues_path = args.input_file


data = load_dataset('json', data_files={'train': dialogues_path})['train'] #or parquet

elem = {}
elem['messages'] = []

for i, d in enumerate(data):

    content = ''.join([e['content'] for e in d['messages']])

    if rpattern.search(content) or cpattern.search(content) or apattern.search(content):
        continue
    elif model_pattern.search(content):
        continue
    else:
        dlist = []
        for turn in d['messages']:
            if turn['role'] == 'system':
                dlist.append({'role':'system', 'content':'You are a helpful assistant.'})
            else:
                content = turn['content']
                dlist.append({'role': turn['role'], 'content': content})
    
        elem['messages'].append(dlist)

elem = Dataset.from_dict(elem)
elem.to_json(saved_dialogues)



