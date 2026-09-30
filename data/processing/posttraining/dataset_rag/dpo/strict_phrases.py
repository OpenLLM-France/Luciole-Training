#!/usr/bin/env python3
"""Imposed refusal phrases and strict-instruction wordings for strict DPO variants (stdlib only).

A strict prompt tells the model which sentence to output when the documents are insufficient.
The model must learn to COPY the sentence given in its instruction, not to reproduce one of a
few memorised sentences: the 5 phrases of hotpotqa_v4 alone do not teach that (after DPO, the
prod model paraphrases the lighteval phrase, which is none of them). So each strict variant
draws, from its pid, one phrase of EN_PHRASES and, independently, one instruction wording of
EN_RULES.

Held out on purpose: the refusal phrases of the lighteval luciole_rag task (HELD_OUT). They
never enter a training prompt, so the eval measures whether the model copies an unseen phrase.
The eval's own wording (backticks, "and nothing else") is one of the rules: the deployer's way
of phrasing the instruction varies too, only the phrase is held out.

Every phrase is at least 40 characters long: build_pairs/build_strict_pairs.is_exact accept up
to 40 extra characters around the phrase, which on a shorter phrase would let a short
explanation count as the exact phrase.

    from strict_phrases import pick_strict
    phrase, rule, rule_id = pick_strict(pid)   # rule = full bullet line of the system prompt
    python3 dpo/strict_phrases.py              # self-check + distribution over 1000 pids
"""
import collections
import hashlib

RULE_PREFIX = "   * If the context is **insufficient** to answer the question, "

# The 5 phrases of hotpotqa_v4 (formatting_sft, add_strict_variants.PHRASES): the SFT saw them.
V4_PHRASES = [
    "The available documents do not contain sufficient information to answer this question. You may want to rephrase it or expand your document base.",
    "I'm unable to answer your question based on the provided context. Please clarify your query or provide additional keywords.",
    "Based on the retrieved context, I cannot provide an answer to your question. Consider rephrasing it or adding more relevant documents.",
    "The context provided does not allow me to answer your question. Could you reformulate your query or add additional relevant documents?",
    "The retrieved documents do not allow me to answer your question. Could you rephrase it or add relevant documents?",
]

# New phrases: varied length, person (I / impersonal / we), vocabulary (documents, sources,
# knowledge base, search results, files), with or without an invitation to rephrase, and some
# without the "do not allow me to answer" core that lighteval's detector looks for.
NEW_PHRASES = [
    "I could not find the answer to this question in the documents I was given.",
    "The information needed to answer this question is not present in the retrieved documents.",
    "Sorry, the sources available to me do not cover this question.",
    "I don't have enough information in the provided documents to answer that.",
    "This question cannot be answered from the documents currently available.",
    "Unfortunately, none of the retrieved passages contain the answer to your question.",
    "The knowledge base does not contain the information required to answer this question.",
    "I cannot answer this question with the documents provided. Please add more relevant sources.",
    "No answer to this question could be found in the search results.",
    "The documents at my disposal are insufficient to answer your question. Try rephrasing it.",
    "I am not able to answer this based on the retrieved context. Could you provide more details?",
    "Your question falls outside the scope of the documents I have access to.",
    "The provided files do not contain an answer to this question.",
    "We could not find this information in the available documents. Please try another query.",
    "Answer not found: the retrieved documents do not address this question.",
    "There is not enough information in the context to answer your question reliably.",
    "The retrieved sources do not mention what you are asking about. Please refine your question.",
    "I'm sorry, but I can't find this information in the documents provided to me.",
    "The context does not provide the information needed to respond to this request.",
    "Based on the documents available, it is not possible to answer this question.",
    "This information is not available in the documents you provided. Consider uploading more files.",
    "The retrieved documents are not sufficient to answer. Please reformulate your question or add documents.",
    "I cannot find any relevant information about this in the provided documents.",
    "None of the documents in my context answer this question. Feel free to ask something else.",
    "The search did not return documents that answer your question. Please try different keywords.",
]
EN_PHRASES = V4_PHRASES + NEW_PHRASES

# Instruction wordings: the part of the bullet after RULE_PREFIX. {phrase} is inserted verbatim.
EN_RULES = {
    "v4": 'respond exactly with: "{phrase}"',
    "eval_style": "reply **exactly** with: `{phrase}` and nothing else.",
    "only_sentence": 'answer only with the following sentence, word for word: "{phrase}"',
    "no_explanation": 'do not explain what is missing; output exactly "{phrase}"',
    "entire_response": 'your entire response must be: "{phrase}"',
    "verbatim": 'reply with this message verbatim and add nothing: "{phrase}"',
    "copy": 'write exactly the following text and nothing more: `{phrase}`',
}

# Refusal phrases of the lighteval luciole_rag task: never in a training prompt.
HELD_OUT = [
    "The provided documents do not allow me to answer this question.",
    "Les documents fournis ne permettent pas de répondre à cette question.",
]


def _norm(s):
    return " ".join(s.lower().split())


def _draw(key, salt, n):
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest(), 16) % n


def strict_rule(phrase, rule_id):
    """Full bullet line of the system prompt for this phrase and wording."""
    return RULE_PREFIX + EN_RULES[rule_id].format(phrase=phrase)


def pick_strict(key, phrases=None, rules=None):
    """(phrase, rule line, rule_id) for a prompt, deterministic in `key` (the pid).

    Phrase and wording are drawn independently (different salts)."""
    phrases = phrases or EN_PHRASES
    rule_ids = rules or list(EN_RULES)
    phrase = phrases[_draw(key, "phrase", len(phrases))]
    rule_id = rule_ids[_draw(key, "rule", len(rule_ids))]
    return phrase, strict_rule(phrase, rule_id), rule_id


def check():
    """Invariants: no held-out phrase (nor a phrase containing one), >= 40 chars, no duplicates,
    no character that would break the quoting of a rule."""
    held = [_norm(h) for h in HELD_OUT]
    seen = set()
    for p in EN_PHRASES:
        n = _norm(p)
        assert not any(h in n or n in h for h in held), f"held-out eval phrase in the pool: {p!r}"
        assert len(p) >= 40, f"phrase shorter than 40 characters: {p!r}"
        assert n not in seen, f"duplicate phrase: {p!r}"
        assert '"' not in p and "`" not in p, f"quote character in phrase: {p!r}"
        seen.add(n)
    for rid, r in EN_RULES.items():
        assert r.count("{phrase}") == 1, rid


check()

if __name__ == "__main__":
    pids = [f"pid{i}" for i in range(1000)]
    ph = collections.Counter(pick_strict(p)[0] for p in pids)
    ru = collections.Counter(pick_strict(p)[2] for p in pids)
    print(f"{len(EN_PHRASES)} phrases ({len(V4_PHRASES)} v4 + {len(NEW_PHRASES)} new), {len(EN_RULES)} wordings; "
          f"over 1000 pids: phrase count {min(ph.values())}-{max(ph.values())}, wordings {dict(ru)}")
    print("example:", strict_rule(EN_PHRASES[7], "eval_style"))
