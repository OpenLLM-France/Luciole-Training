import argparse
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- terminal colors (ANSI) ---
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"


def color_prob(prob, threshold):
    """Green = healthy, yellow = a bit low, red = below threshold."""
    if prob < threshold:
        return RED
    if prob < 10 * threshold:
        return YELLOW
    return GREEN


parser = argparse.ArgumentParser(
    description="Inspect per-token probabilities the model assigns to the target "
    "(assistant) tokens of a chat dataset, and flag low-probability tokens."
)
parser.add_argument("model_name", type=str)
parser.add_argument(
    "input_file", type=str, help="Path to a .jsonl file with a 'messages' field"
)
parser.add_argument(
    "--messages_field",
    type=str,
    default="messages",
    help="Name of the dataset column holding the list of chat messages",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.01,
    help="Flag target tokens whose probability is below this value (default: 0.01)",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
    help="Limit the number of examples processed (default: all)",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=None,
    help="Optionally truncate the tokenized sequence to this many tokens",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="target_token_probs.jsonl",
    help="Where to write per-example results",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def build_inputs(tokenizer, messages):
    """Tokenize a conversation, returning input_ids and a mask of the target
    (assistant) tokens. Falls back gracefully if the chat template does not
    support assistant-token masks."""
    try:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            return_tensors="pt",
        )
        assistant_mask = enc.get("assistant_masks")
        if assistant_mask is not None:
            assistant_mask = torch.as_tensor(assistant_mask)
            if assistant_mask.sum() > 0:
                return enc["input_ids"], assistant_mask.bool()
    except (TypeError, ValueError):
        pass

    # Fallback: no assistant mask available from the template. Score every token
    # except the very first (which has no preceding context to be predicted from).
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    mask = torch.ones_like(enc["input_ids"], dtype=torch.bool)
    mask[:, 0] = False
    return enc["input_ids"], mask


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True, torch_dtype="auto"
        )
        .eval()
        .to(device)
    )

    dataset = load_dataset("json", data_files=args.input_file, split="train")
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    all_target_probs = []
    n_low = 0
    n_target = 0

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for i, example in enumerate(dataset):
            messages = example[args.messages_field]
            input_ids, target_mask = build_inputs(tokenizer, messages)

            if args.max_length is not None:
                input_ids = input_ids[:, : args.max_length]
                target_mask = target_mask[:, : args.max_length]

            input_ids = input_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids).logits  # (1, seq, vocab)

            # logits at position t predict the token at position t+1.
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = input_ids[:, 1:]  # (1, seq-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            token_probs = token_log_probs.exp().squeeze(0)  # (seq-1,)

            # Align the target mask: a token at position t+1 is "targeted" if
            # mask[t+1] is set; its prob lives at token_probs[t].
            keep = target_mask[0, 1:]
            target_ids = targets[0][keep]
            target_probs = token_probs[keep]

            low_tokens = []
            for tok_id, prob in zip(target_ids.tolist(), target_probs.tolist()):
                all_target_probs.append(prob)
                n_target += 1
                if prob < args.threshold:
                    n_low += 1
                    low_tokens.append(
                        {
                            "token_id": tok_id,
                            # decoded text: human-readable, but a single token from a
                            # byte-level BPE may hold only part of a multi-byte char
                            # and decode to U+FFFD ('replacement char').
                            "token": tokenizer.decode([tok_id]),
                            # raw piece: never loses bytes (SentencePiece '_', byte-level
                            # 'A©' / '<0xE9>'), so the token is always identifiable.
                            "piece": tokenizer.convert_ids_to_tokens(tok_id),
                            "prob": prob,
                        }
                    )

            n_tgt = target_probs.numel()
            record = {
                "example_id": i,
                "num_target_tokens": n_tgt,
                "mean_prob": target_probs.mean().item() if n_tgt else None,
                "min_prob": target_probs.min().item() if n_tgt else None,
                "perplexity": (-token_log_probs.squeeze(0)[keep].mean()).exp().item()
                if n_tgt
                else None,
                "num_low_prob": len(low_tokens),
                "low_prob_tokens": low_tokens,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if low_tokens:
                print(
                    f"\n{BOLD}{CYAN}[example {i}]{RESET} "
                    f"{RED}{len(low_tokens)}{RESET}/{n_tgt} target tokens "
                    f"below {args.threshold}:"
                )
                for t in low_tokens[:10]:
                    col = color_prob(t["prob"], args.threshold)
                    # repr() escapes whitespace/control chars; the raw piece is shown
                    # alongside so partial-UTF-8 tokens that decode to U+FFFD stay legible.
                    print(
                        f"    {col}{repr(t['token']):<20}{RESET} "
                        f"{DIM}piece={RESET}{repr(t['piece']):<16} "
                        f"{DIM}id={RESET}{t['token_id']:<7} "
                        f"{DIM}p={RESET}{col}{t['prob']:.2e}{RESET}"
                    )
                if len(low_tokens) > 10:
                    print(f"    {DIM}... and {len(low_tokens) - 10} more{RESET}")

    if n_target:
        probs = torch.tensor(all_target_probs)
        low_frac = 100 * n_low / n_target
        low_col = RED if low_frac > 1 else YELLOW if low_frac > 0 else GREEN
        print(f"\n{BOLD}{CYAN}{'=' * 16} Summary {'=' * 16}{RESET}")
        print(f"  examples processed : {BOLD}{len(dataset)}{RESET}")
        print(f"  target tokens      : {BOLD}{n_target}{RESET}")
        print(f"  mean prob          : {GREEN}{probs.mean().item():.4f}{RESET}")
        print(f"  median prob        : {GREEN}{probs.median().item():.4f}{RESET}")
        print(
            f"  low-prob (<{args.threshold}) : {low_col}{n_low}{RESET} "
            f"{low_col}({low_frac:.2f}%){RESET}"
        )
        print(f"  results written to : {DIM}{args.output_file}{RESET}\n")
    else:
        print(
            f"\n{YELLOW}No target tokens were found. "
            f"Check --messages_field and the chat template.{RESET}\n"
        )
