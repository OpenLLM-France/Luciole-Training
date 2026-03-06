import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

parser = argparse.ArgumentParser(description="")
parser.add_argument("model_name", type=str, default="")
parser.add_argument("--chat", action="store_true")
parser.add_argument("--save_path", type=str, default="vibe_evaluation_results.txt")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--eos_token_id", nargs="+", type=int, default=None)
parser.add_argument("--num_sequences", type=int, default=1)
parser.add_argument("--system_prompt", type=str, default=None)
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
                conversation = [{"role": "user", "content": prompt}]
                if args.system_prompt:
                    conversation.insert(
                        0, {"role": "system", "content": args.system_prompt}
                    )
                input_ids = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            args_dict = vars(args)  # convert argparse Namespace → dict

            samples = []

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    num_return_sequences=args.num_sequences,
                    do_sample=True,
                    top_k=50,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    eos_token_id=args.eos_token_id,
                )

            for j, seq in enumerate(output):
                text = tokenizer.decode(seq, skip_special_tokens=False)

                print(f"\n\n#### Example #{i}-{j}\n{text}\n")

                samples.append(
                    {
                        "example_id": i,
                        "sample_id": j,
                        "text": text,
                        "generation_args": args_dict,
                    }
                )

            with open("generations.jsonl", "a") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
