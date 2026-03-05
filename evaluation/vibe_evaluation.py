import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="")
parser.add_argument("model_name", type=str, default="")
parser.add_argument("--chat", action="store_true")
parser.add_argument("--save_path", type=str, default="vibe_evaluation_results.txt")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.9)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    model_name = args.model_name

    vibe_prompts = []
    with open("vibe_prompts.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                vibe_prompts.append(line)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        .eval()
        .to(device)
    )

    with open(args.save_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(vibe_prompts):
            if args.chat:
                # tokenizer.eos_token = "<|im_end|>"
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
            else:
                # # Option 1
                # tokenizer.bos_token = "</s>"
                # tokenizer.bos_token_id = tokenizer.eos_token_id
                # tokenizer.add_bos_token = True
                # Option 2
                # tokenizer.add_bos_token = False

                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=1024,
                    num_return_sequences=1,
                    do_sample=True,
                    # top_k=50,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    eos_token_id=[1, 261],
                )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
            print(f"\n\n#### Example #{i} \n{generated_text}\n")
            f.write(f"\n\n#### Example #{i} \n{generated_text}\n")
