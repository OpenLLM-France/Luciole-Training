import os
import re
from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA, instruct_adapter
from functools import partial


'''def convert_messages(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1, tokenizer=None
) -> DocumentsPipeline:
    import re
    for document in data:
        messages = []
        for m in document.text:
            del m['function_calls']
            del m['functions']
            messages.append(m)
        #messages = [convert_message(m) for m in document.text]
        document.text = messages
        #document.text = tokenizer.apply_chat_template(messages, tokenize=False)
        # Remove empty think tags, caused by unfinished thoughts
        #if "<think>\n\n</think>\n\n<think>" in document.text:
        #    document.text = document.text.replace(
        #        "<think>\n\n</think>\n\n<think>", "<think>\n"
        #    )
        #    document.text = document.text[:-11]  # remove "<|im_end|>\n" at the end
        # Remove Chinese-heavy lines that can appear in thoughts
        flag = False
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        for message in document.text:
            chinese_chars = chinese_pattern.findall(message)
            if chinese_chars:
                flag = True
        if flag:
            continue
        #document.metadata["conversation"] = messages
        yield document'''

def convert_messages(data: DocumentsPipeline, rank: int = 0, world_size: int = 1, tokenizer=None) -> DocumentsPipeline:
    import re
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
    "Mixtral", "Mistral", "NVIDIA", "Nemotron", "Allen AI", "Olmo"]

    model_pattern = re.compile("|".join(map(re.escape, filter_strings)), re.IGNORECASE)
    rpattern = re.compile(r'[А-Яа-яЁё]') #Russian
    cpattern = re.compile(r'[\u4E00-\u9FFF]') #Chinese
    apattern = re.compile(r'[\u0600-\u06FF]') #Arabic
    
    for document in data:
        domain = document.metadata.get("domain", "unknown")
        document.metadata["domain"] = str(domain).replace(" ", "")
        messages = []
        flag = False
        
        for m in document.text:
            #print('m string?', document.text)
            # Clean up the message dict
            m.pop('function_calls', None)
            m.pop('functions', None)
            
            # Check the actual content string for Chinese chars
            #content = m.get("content", "")
            content = m.get("content") or ""
            if content and (cpattern.search(content) or rpattern.search(content) or apattern.search(content)):
                flag = True
            if content and model_pattern.search(content):
                flag = True
            
            messages.append(m)
        
        if flag:
            continue
            
        #document.text = messages
        document.metadata['messages'] = messages
        
        yield document

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    DATA_PATH = args.data_path
    dataset_name = "dolci_instructsft"
    output_path = os.path.join(DATA_PATH, dataset_name)
    print('datapath', DATA_PATH)
    print(args)

    pipeline = [
        #HuggingFaceDatasetReader(
         #   "allenai/Dolci-Instruct-SFT", 
         #   {"split": "train"}, 
         #   streaming=True, 
         #   adapter=instruct_adapter,
        #),
        ParquetReader(
            "hf://datasets/allenai/Dolci-Instruct-SFT/data",
            glob_pattern="*.parquet",
            #adapter=instruct_adapter,
            text_key="messages",
        ),
        partial(convert_messages),
        JsonlWriter(
            f"{output_path}/data", output_filename="${domain}/rank${rank}.jsonl.gz", 
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
        tasks=5,
    )

    main_processing_executor.run()

