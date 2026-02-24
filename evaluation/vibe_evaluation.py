import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--chat", action="store_true")
parser.add_argument("--save_path", type=str, default="vibe_evaluation_results.txt")
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
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
            else:
                tokenizer.bos_token_id = tokenizer.eos_token_id
                input_ids = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=True
                ).input_ids.to(device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=128,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
            print(f"\n\n#### Example #{i} \n{generated_text}\n")
            f.write(f"\n\n#### Example #{i} \n{generated_text}\n")
