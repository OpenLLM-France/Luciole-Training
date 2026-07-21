"""
Convert augmented dataset to prompt/completion or message format for fine-tuning.

Context reformatting and citation marker normalization are applied so that the
output matches exactly what OpenRAG sends to the LLM at inference time.

Citation style variation: for each example a citation markup style is randomly
chosen (different begin/end quote markers and cite format) and the system prompt is
updated to match, so the model learns whatever markup is described. Citation
placement (inline after each quote vs a trailing **References** section) is NOT
drawn here: it is decided once at generation time and stored per row
(``citation_end_of_completion``) so the generation prompt matches the inference
prompt; formatting reuses it (see _resolve_end_of_completion). When rebuilding a
References section, a stray header the generator left is stripped first.

No "Final Answer" line: the OpenRAG prompt does not request one, so any trailing
Final Answer block found in a trace is stripped from the completion.

Unanswerable variation: for unanswerable rows, a refusal phrase is randomly
chosen from a pool of semantically equivalent formulations per language. For a
subset of refusal examples, the system prompt specifies that exact phrase as
the response to use when the context is insufficient, so the model learns to
follow a declared refusal wording.
"""

import json
import random
import re
from pathlib import Path
from dataclasses import dataclass

from utils import _strip_final_answer_line


# Context helpers (aligned with OpenRAG inference format)

# 10-dash separator on its own line, matching OpenRAG production format
CONTEXT_SEPARATOR = "\n" + "-" * 10 + "\n\n"


def _is_openrag_format(context: str) -> bool:
    """Detect whether a context string uses the OpenRAG [CHUNK_START]...[CHUNK_END] format."""
    return "[CHUNK_START]" in context and "[CHUNK_END]" in context


def _reformat_openrag_chunks(raw_context: str) -> str:
    """Reformat OpenRAG-style chunks with the standard separator between them.

    Handles both contextualised (``[CONTEXT]...``) and base (``* filename:...``) variants.
    """
    chunk_pattern = re.compile(
        r'((?:\[CONTEXT\].*?)?(?:\*\s*filename:.*?)\[CHUNK_START\].*?\[CHUNK_END\])',
        re.DOTALL,
    )
    chunks = chunk_pattern.findall(raw_context)
    if not chunks:
        return raw_context.strip()
    return CONTEXT_SEPARATOR.join(c.strip() for c in chunks)


def reformat_context_chunks(raw_context: str) -> str:
    """Reformat context chunks to use OpenRAG's separator.

    Supports two input formats:

    1. Classic ``[Title]`` format (chunks separated by double newlines):
        [Title 1]
        Content of chunk 1...

        [Title 2]
        Content of chunk 2...

    2. OpenRAG contextualised format:
        [CONTEXT]
        summary...
        * Title 1

        [CHUNK_START]
        Content of chunk 1...
        [CHUNK_END]

    Output always uses the ``----------`` separator between chunks.
    """
    if _is_openrag_format(raw_context):
        return _reformat_openrag_chunks(raw_context)

    clean_context = raw_context.replace("-" * 10, "\n")  # remove old separators if any
    pattern = r'\n?\[([^\]]+)\]\n'
    parts = re.split(pattern, clean_context)

    chunks = []
    if parts[0].strip():
        chunks.append(parts[0].strip())

    for i in range(1, len(parts), 2):
        title = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunk = f"[{title}]\n{content}" if content else f"[{title}]"
        chunks.append(chunk)

    if not chunks:
        return raw_context.strip()

    return CONTEXT_SEPARATOR.join(chunks)


def normalize_cite_markers(content: str) -> str:
    """Normalize all ##Cite## variants to ##Cite "title"## (double quotes, no extra spaces)."""
    content = re.sub(r'##Cite\s+"([^"]*?)"\s*##', r'##Cite "\1"##', content)
    content = re.sub(r"##Cite\s+'([^']*?)'\s*##", r'##Cite "\1"##', content)
    return content


# Citation style variation

# Probability of grouping citation markers in a trailing References section
# (instead of inline after each quote). This placement is decided ONCE, at
# generation time, and stored per row (``citation_end_of_completion``) so the
# generation prompt matches the inference prompt for that row; formatting reuses
# the stored value (see _resolve_end_of_completion) and only falls back to a fresh
# draw for legacy rows that predate the stored field.
EOC_PROB = 0.30

# Probability of using the English system prompt for non-English data
# when --mixed-system-prompt is enabled (the query, docs, and answer stay in
# the data language; only the system instructions switch to English).
MIXED_SYSTEM_EN_PROB = 0.5

# Probability, among unanswerable examples, of explicitly declaring the exact
# refusal phrase to use in the system prompt.
SPECIFIC_REFUSAL_INSTRUCTION_PROB = 0.30


@dataclass
class CitationStyle:
    """Describes a set of citation markup tags and matching system-prompt instructions."""
    name: str
    begin_quote: str
    end_quote: str
    cite_template: str       # uses literal '{title}' as placeholder for the source name
    inline_instruction: dict  # language → instruction string for inline citations
    eoc_instruction: dict    # language → instruction string for end-of-completion citations
    references_header: dict  # language → section header string


# Regex matching normalized cite markers produced by normalize_cite_markers()
_CITE_RE = re.compile(r'##Cite "([^"]*)"##')


def _make_style(name: str, begin: str, end: str, cite_template: str) -> CitationStyle:
    """Build a CitationStyle, generating instruction strings from the markup parameters."""
    cite_en = cite_template.replace("{title}", "source title")
    cite_fr = cite_template.replace("{title}", "titre de la source")
    inline = {
        "en": (
            f"When quoting from the context, wrap the excerpt with `{begin}` and `{end}`, "
            f"and attribute it with `{cite_en}`."
        ),
        "fr": (
            f"Lorsque vous citez le contexte, encadrez l'extrait avec `{begin}` et `{end}`, "
            f"et attribuez-le avec `{cite_fr}`."
        ),
    }
    eoc = {
        "en": (
            f"When quoting from the context, wrap the excerpt with `{begin}` and `{end}`. "
            f"Do **not** add citation markers after each quote. Instead, list all cited sources "
            f"at the end of your response in a **References** section using `{cite_en}` for each source."
        ),
        "fr": (
            f"Lorsque vous citez le contexte, encadrez l'extrait avec `{begin}` et `{end}`. "
            f"N'ajoutez **pas** de marqueurs de citation après chaque citation. "
            f"Listez toutes les sources citées à la fin de votre réponse dans une section "
            f"**Références** en utilisant `{cite_fr}` pour chaque source."
        ),
    }
    return CitationStyle(
        name=name,
        begin_quote=begin,
        end_quote=end,
        cite_template=cite_template,
        inline_instruction=inline,
        eoc_instruction=eoc,
        references_header={"en": "**References**", "fr": "**Références**"},
    )


CITATION_STYLES = [
    _make_style("default", "##begin_quote##", "##end_quote##", '##Cite "{title}"##'),
    _make_style("xml", "<quote>", "</quote>", '<cite>"{title}"</cite>'),
    _make_style("bracket", "[QUOTE]", "[/QUOTE]", '[Source: "{title}"]'),
    _make_style("double_angle", "<<quote>>", "<</quote>>", '<<cite: "{title}">>'),
]


# Trailing "References"/"Sources" header the generator emits above its cite list.
# Matched so it can be stripped before we rebuild the section in the chosen style
# (otherwise the completion would end up with two References headers).
_REFERENCES_HEADER_RE = re.compile(
    r'(?:\n+|^)[ \t]*#{0,6}[ \t]*\*{0,2}'
    r'(?:references|références|sources|source)'
    # trailing markup: closing ATX hashes, bold asterisks, colon, in any mix
    r'[ \t]*[*#:]*[ \t]*(?=\n|$)',
    re.IGNORECASE,
)


# A line is "list-like" (part of a leftover reference list) if it is blank or a
# bullet/numbered item, possibly emptied once its inline cite marker was removed.
_LIST_LINE_RE = re.compile(r'^[ \t]*(?:[-*•]|\d+[.)])?[ \t]*$')


def _strip_trailing_references_header(text: str) -> str:
    """Drop the generator's trailing References/Sources section from *text*.

    Removes the header and any following reference list, including bullets that
    became empty once their inline cite markers were stripped. Only strips when
    everything after the header is list-like (bullets/blank), so prose is never
    swallowed. Callers rebuild the section from the collected titles afterwards.
    """
    stripped = text.rstrip()
    m = _REFERENCES_HEADER_RE.search(stripped)
    if not m:
        return stripped
    tail = stripped[m.end():]
    if all(_LIST_LINE_RE.match(line) for line in tail.splitlines()):
        stripped = stripped[:m.start()].rstrip()
    return stripped


def apply_citation_style(
    completion: str,
    style: CitationStyle,
    end_of_completion: bool,
    language: str,
) -> str:
    """Replace normalized markers with the chosen style (inline or end-of-completion)."""
    result = completion.replace("##begin_quote##", style.begin_quote)
    result = result.replace("##end_quote##", style.end_quote)

    if end_of_completion:
        # Collect all cited titles in first-seen order, deduplicated
        seen: set = set()
        unique_titles = []
        for m in _CITE_RE.finditer(result):
            title = m.group(1)
            if title not in seen:
                seen.add(title)
                unique_titles.append(title)

        # Remove all inline cite markers (plus any space immediately before them)
        result = re.sub(r'\s*' + _CITE_RE.pattern, '', result)
        # Drop the generator's own References header so we don't duplicate it.
        result = _strip_trailing_references_header(result)
        result = result.rstrip()

        if unique_titles:
            header = style.references_header.get(language, style.references_header["en"])
            refs = "\n".join(
                "- " + style.cite_template.replace("{title}", title)
                for title in unique_titles
            )
            result = f"{result}\n\n{header}\n{refs}"
    else:
        def _replace(match: re.Match) -> str:
            return style.cite_template.replace("{title}", match.group(1))
        result = _CITE_RE.sub(_replace, result)

    return result


# Unanswerable refusal variation

UNANSWERABLE_REFUSALS = {
    "en": [
        "The retrieved documents do not allow me to answer your question. Could you rephrase it or add relevant documents?",
        "I'm unable to answer your question based on the provided context. Please clarify your query or provide additional keywords.",
        "The available documents do not contain sufficient information to answer this question. You may want to rephrase it or expand your document base.",
        "Based on the retrieved context, I cannot provide an answer to your question. Consider rephrasing it or adding more relevant documents.",
        "The context provided does not allow me to answer your question. Could you reformulate your query or add additional relevant documents?",
    ],
    "fr": [
        "Les documents récupérés ne me permettent pas de répondre à votre question. Pourriez-vous la reformuler ou ajouter des documents pertinents ?",
        "Je ne suis pas en mesure de répondre à votre question sur la base du contexte fourni. Veuillez préciser votre requête ou fournir des mots-clés supplémentaires.",
        "Les documents disponibles ne contiennent pas suffisamment d'informations pour répondre à cette question. Vous pouvez reformuler votre requête ou enrichir votre base documentaire.",
        "D'après le contexte récupéré, je ne peux pas répondre à votre question. Envisagez de la reformuler ou d'ajouter des documents plus pertinents.",
        "Le contexte fourni ne me permet pas de répondre à votre question. Pourriez-vous reformuler votre requête ou ajouter des documents pertinents supplémentaires ?",
    ],
}


def _pick_refusal(rng, language: str) -> str:
    """Pick a random refusal phrase from the pool for the given language."""
    refusals = UNANSWERABLE_REFUSALS.get(language, UNANSWERABLE_REFUSALS["en"])
    return rng.choice(refusals)


def _refusal_instruction(refusal: str, language: str) -> str:
    """Build the system-prompt rule line specifying the refusal phrase to use."""
    if language == "fr":
        return (
            f"Si le contexte est **insuffisant** pour répondre à la question, "
            f"répondez exactement par : « {refusal} »"
        )
    return (
        f"If the context is **insufficient** to answer the question, "
        f"respond exactly with: \"{refusal}\""
    )


def _generic_refusal_instruction(language: str) -> str:
    """Build the generic system-prompt rule for insufficient context."""
    if language == "fr":
        return (
            "Si le contexte est **insuffisant** pour répondre à la question, "
            "indiquez que les documents récupérés ne permettent pas de répondre "
            "et invitez l'utilisateur à reformuler sa requête ou à ajouter des documents pertinents."
        )
    return (
        "If the context is **insufficient** to answer the question, state that "
        "the retrieved documents do not allow you to answer and invite the user "
        "to rephrase the query or add relevant documents."
    )


# System prompt templates — prompt/completion format
# (context + question embedded in the prompt; {citation_instruction} filled per-example)

SYSTEM_PROMPT_TEMPLATE = """You are an AI conversational assistant specialized in **information retrieval and synthesis**.
Your goal is to provide **precise, reliable, and well-structured answers** using **only the retrieved documents** (`Context`).
Prioritize **clarity, accuracy, and completeness** in your responses.

## Rules

1. Use only the provided Context
   * Base your answer **exclusively** on the information contained in the `Context`.
   * **Never infer**, assume, or rely on any external knowledge.
   * {refusal_instruction}
   * {citation_instruction}

2. Language Consistency
   * Always respond **in the same language** as the user's query.

3. Structure and Readability
   * Use **headings**, **bullet points**, **numbered lists**, or **tables** to organize information clearly.
   * Ensure responses are **concise yet complete**, avoiding omission of key details.

Here are the retrieved documents: `{context}`

user query '{question}'"""


SYSTEM_PROMPT_TEMPLATE_FR = """Vous êtes un assistant conversationnel IA spécialisé dans la **recherche et la synthèse d'informations**.
Votre objectif est de fournir des **réponses précises, fiables et bien structurées** en utilisant **uniquement les documents récupérés** (`Contexte`).
Privilégiez la **clarté, l'exactitude et l'exhaustivité** dans vos réponses.

## Règles

1. Utilisez uniquement le Contexte fourni
   * Basez votre réponse **exclusivement** sur les informations contenues dans le `Contexte`.
   * **N'inférez jamais**, ne supposez pas et ne vous appuyez pas sur des connaissances externes.
   * {refusal_instruction}
   * {citation_instruction}

2. Cohérence linguistique
   * Répondez toujours **dans la même langue** que la requête de l'utilisateur.

3. Structure et lisibilité
   * Utilisez des **titres**, des **listes à puces**, des **listes numérotées** ou des **tableaux** pour organiser clairement les informations.
   * Assurez-vous que les réponses sont **concises mais complètes**, en évitant d'omettre les détails clés.

Voici les documents récupérés : `{context}`

requête de l'utilisateur '{question}'"""


# System prompt templates — chat format
# (context in system message, question in user message; {citation_instruction} filled per-example)

CHAT_SYSTEM_PROMPT = (
    "You are an AI conversational assistant specialized in **information retrieval and synthesis**.\n"
    "Your goal is to provide **precise, reliable, and well-structured answers** using **only the retrieved documents** (`Context`).\n"
    "Prioritize **clarity, accuracy, and completeness** in your responses.\n"
    "\n"
    "## Rules\n"
    "\n"
    "1. Use only the provided Context\n"
    "   * Base your answer **exclusively** on the information contained in the `Context`.\n"
    "   * **Never infer**, assume, or rely on any external knowledge.\n"
    "   * {refusal_instruction}\n"
    "   * {citation_instruction}\n"
    "\n"
    "2. Language Consistency\n"
    "   * Always respond **in the same language** as the user's query.\n"
    "\n"
    "3. Structure and Readability\n"
    "   * Ensure responses are **concise yet complete**, avoiding omission of key details.\n"
    "\n"
    "Here are the retrieved documents : `{context}`"
)

CHAT_SYSTEM_PROMPT_FR = (
    "Vous êtes un assistant conversationnel IA spécialisé dans la **recherche et la synthèse d'informations**.\n"
    "Votre objectif est de fournir des **réponses précises, fiables et bien structurées** en utilisant **uniquement les documents récupérés** (`Contexte`).\n"
    "Privilégiez la **clarté, l'exactitude et l'exhaustivité** dans vos réponses.\n"
    "\n"
    "## Règles\n"
    "\n"
    "1. Utilisez uniquement le Contexte fourni\n"
    "   * Basez votre réponse **exclusivement** sur les informations contenues dans le `Contexte`.\n"
    "   * **N'inférez jamais**, ne supposez pas et ne vous appuyez pas sur des connaissances externes.\n"
    "   * {refusal_instruction}\n"
    "   * {citation_instruction}\n"
    "\n"
    "2. Cohérence linguistique\n"
    "   * Répondez toujours **dans la même langue** que la requête de l'utilisateur.\n"
    "\n"
    "3. Structure et lisibilité\n"
    "   * Assurez-vous que les réponses sont **concises mais complètes**, en évitant d'omettre les détails clés.\n"
    "\n"
    "Voici les documents récupérés : `{context}`"
)


@dataclass
class ConversionStats:
    total: int = 0
    converted: int = 0
    unanswerable: int = 0
    skipped: int = 0


def format_prompt(
    context: str,
    question: str,
    language: str,
    citation_instruction: str,
    refusal_instruction: str,
) -> str:
    """Format the prompt using the system template for the given language and citation style."""
    templates = {
        "en": SYSTEM_PROMPT_TEMPLATE,
        "fr": SYSTEM_PROMPT_TEMPLATE_FR,
    }
    template = templates.get(language, SYSTEM_PROMPT_TEMPLATE)
    return template.format(
        context=reformat_context_chunks(context),
        question=question,
        citation_instruction=citation_instruction,
        refusal_instruction=refusal_instruction,
    )


def _pick_style(rng) -> CitationStyle:
    """Randomly pick a citation markup style (augmentation applied at formatting)."""
    return rng.choice(CITATION_STYLES)


def _resolve_end_of_completion(row: dict, rng) -> bool:
    """Citation placement (inline vs trailing References section).

    Reuse the value chosen at generation time (``citation_end_of_completion``) so
    the generation prompt matches the inference prompt; fall back to a fresh draw
    for legacy rows that don't carry the field.
    """
    stored = row.get("citation_end_of_completion")
    if stored is not None:
        return bool(stored)
    return rng.random() < EOC_PROB


# The canonical (default) markup style is the one used at generation time; the
# markup variety (xml/bracket/...) is a formatting-only augmentation.
DEFAULT_CITATION_STYLE = CITATION_STYLES[0]


def build_chat_system_prompt(
    context: str,
    language: str,
    style: CitationStyle,
    end_of_completion: bool,
    refusal_instruction: str,
) -> str:
    """Build the OpenRAG chat system prompt shared by generation and inference.

    Single source of truth for the prompt the model is conditioned on: the augment
    scripts call this (with DEFAULT_CITATION_STYLE) to generate each trace, and
    formatting calls it to build the training example's system message.
    """
    template = {"en": CHAT_SYSTEM_PROMPT, "fr": CHAT_SYSTEM_PROMPT_FR}.get(language, CHAT_SYSTEM_PROMPT)
    instruction_dict = style.eoc_instruction if end_of_completion else style.inline_instruction
    return template.format(
        context=reformat_context_chunks(context),
        citation_instruction=instruction_dict.get(language, instruction_dict["en"]),
        refusal_instruction=refusal_instruction,
    )


def _assistant_answer(row: dict, style: CitationStyle, end_of_completion: bool, language: str) -> str:
    """Answer target: the generated trace, restyled, without a Final Answer line."""
    trace = _strip_final_answer_line(normalize_cite_markers(row["reasoning_trace"]))
    return apply_citation_style(trace, style, end_of_completion, language)


def _pick_system_lang(rng, data_language: str, mixed_system_prompt: bool) -> str:
    """Pick the language used for system prompt instructions.

    When mixed_system_prompt is True and data is non-English, returns "en" with
    probability MIXED_SYSTEM_EN_PROB so the model sees English instructions
    alongside non-English data/answers.
    """
    if mixed_system_prompt and data_language != "en" and rng.random() < MIXED_SYSTEM_EN_PROB:
        return "en"
    return data_language


def convert_row(row: dict, language: str = "en", rng=None, mixed_system_prompt: bool = False) -> dict:
    """Convert a single row to prompt/completion format with randomized citation style."""
    if rng is None:
        rng = random

    style = _pick_style(rng)
    end_of_completion = _resolve_end_of_completion(row, rng)
    system_lang = _pick_system_lang(rng, language, mixed_system_prompt)
    is_unanswerable = row.get("is_unanswerable", False)

    instruction_dict = style.eoc_instruction if end_of_completion else style.inline_instruction
    citation_instruction = instruction_dict.get(system_lang, instruction_dict["en"])

    # Refusal phrase stays in data language (assistant must reply in that language);
    # the wrapping rule text follows system_lang.
    refusal = _pick_refusal(rng, language)
    specific_refusal_instruction = (
        is_unanswerable and rng.random() < SPECIFIC_REFUSAL_INSTRUCTION_PROB
    )
    if specific_refusal_instruction:
        refusal_instruction = _refusal_instruction(refusal, system_lang)
    else:
        refusal_instruction = _generic_refusal_instruction(system_lang)

    prompt = format_prompt(
        context=row["context"],
        question=row["question"],
        language=system_lang,
        citation_instruction=citation_instruction,
        refusal_instruction=refusal_instruction,
    )

    if is_unanswerable:
        completion = refusal
    else:
        completion = _assistant_answer(row, style, end_of_completion, language)

    result = {
        "prompt": prompt,
        "completion": completion,
    }

    # Preserve metadata for reference
    metadata_fields = [
        # Original metadata
        "id", "original_dataset", "is_unanswerable", "type", "level", "answer_type", "answer_from", "scale",
        # Chunk composition metadata
        "chunks_relevant", "chunks_background_initial", "chunks_background_removed",
        "chunks_background_final", "chunks_total",
        # Supporting facts metadata
        "supporting_facts_titles",
        # Evaluation metadata (from evaluate_answers_v2.py)
        "eval_answer_correct", "eval_answer_match_type", "eval_extracted_answer",
        "eval_cited_titles", "eval_chunk_precision", "eval_chunk_recall", "eval_chunk_f1",
        "eval_factual_judge_score", "eval_factual_judge_justification",
        "eval_french_quality_judge_score", "eval_french_quality_judge_justification",
    ]
    for field in metadata_fields:
        if field in row:
            result[field] = row[field]

    # Citation variation metadata
    result["citation_style"] = style.name
    result["end_of_completion_citations"] = end_of_completion
    result["system_prompt_language"] = system_lang
    result["specific_refusal_instruction"] = specific_refusal_instruction

    return result


def convert_dataset(
    input_file: str,
    output_file: str,
    language: str = "en",
    include_metadata: bool = True,
    seed: int | None = None,
    mixed_system_prompt: bool = False,
) -> ConversionStats:
    """Convert augmented dataset to prompt/completion format."""
    stats = ConversionStats()
    converted_rows = []
    rng = random.Random(seed)

    with open(input_file, "r") as f:
        for line in f:
            stats.total += 1
            row = json.loads(line)

            try:
                converted = convert_row(row, language, rng, mixed_system_prompt=mixed_system_prompt)

                if not include_metadata:
                    minimal = {
                        "prompt": converted["prompt"],
                        "completion": converted["completion"],
                    }
                    for keep_field in ("supporting_facts_titles", "chunks_total"):
                        if keep_field in converted:
                            minimal[keep_field] = converted[keep_field]
                    converted = minimal

                converted_rows.append(converted)
                stats.converted += 1

                if row.get("is_unanswerable", False):
                    stats.unanswerable += 1

            except Exception as e:
                print(f"Error converting row {row.get('id', 'unknown')}: {e}")
                stats.skipped += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for row in converted_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


def convert_to_chat_format(
    input_file: str,
    output_file: str,
    language: str = "en",
    include_metadata: bool = True,
    seed: int | None = None,
    mixed_system_prompt: bool = False,
) -> ConversionStats:
    """Convert augmented dataset to chat/messages format for instruction fine-tuning."""
    stats = ConversionStats()
    rng = random.Random(seed)

    metadata_fields = [
        # Original metadata
        "id", "original_dataset", "is_unanswerable", "type", "level", "answer_type", "answer_from", "scale",
        # Chunk composition metadata
        "chunks_relevant", "chunks_background_initial", "chunks_background_removed",
        "chunks_background_final", "chunks_total",
        # Supporting facts metadata
        "supporting_facts_titles",
        # Evaluation metadata (from evaluate_answers_v2.py)
        "eval_answer_correct", "eval_answer_match_type", "eval_extracted_answer",
        "eval_cited_titles", "eval_chunk_precision", "eval_chunk_recall", "eval_chunk_f1",
        "eval_factual_judge_score", "eval_factual_judge_justification",
        "eval_french_quality_judge_score", "eval_french_quality_judge_justification",
    ]

    with open(input_file, "r") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            stats.total += 1
            row = json.loads(line)

            try:
                style = _pick_style(rng)
                end_of_completion = _resolve_end_of_completion(row, rng)
                system_lang = _pick_system_lang(rng, language, mixed_system_prompt)
                is_unanswerable = row.get("is_unanswerable", False)

                # Refusal phrase stays in data language; the wrapping rule text follows system_lang.
                refusal = _pick_refusal(rng, language)
                specific_refusal_instruction = (
                    is_unanswerable and rng.random() < SPECIFIC_REFUSAL_INSTRUCTION_PROB
                )
                if specific_refusal_instruction:
                    refusal_instruction = _refusal_instruction(refusal, system_lang)
                else:
                    refusal_instruction = _generic_refusal_instruction(system_lang)

                system_content = build_chat_system_prompt(
                    context=row["context"],
                    language=system_lang,
                    style=style,
                    end_of_completion=end_of_completion,
                    refusal_instruction=refusal_instruction,
                )

                if is_unanswerable:
                    assistant_content = refusal
                else:
                    assistant_content = _assistant_answer(row, style, end_of_completion, language)

                chat_row = {
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }

                if include_metadata:
                    for field in metadata_fields:
                        if field in row:
                            chat_row[field] = row[field]
                    chat_row["citation_style"] = style.name
                    chat_row["end_of_completion_citations"] = end_of_completion
                    chat_row["system_prompt_language"] = system_lang
                    chat_row["specific_refusal_instruction"] = specific_refusal_instruction
                else:
                    for keep_field in ("supporting_facts_titles", "chunks_total"):
                        if keep_field in row:
                            chat_row[keep_field] = row[keep_field]

                f_out.write(json.dumps(chat_row, ensure_ascii=False) + "\n")
                stats.converted += 1

                if is_unanswerable:
                    stats.unanswerable += 1

            except Exception as e:
                print(f"Error converting row {row.get('id', 'unknown')}: {e}")
                stats.skipped += 1

    return stats


def print_stats(stats: ConversionStats, output_file: str):
    """Print conversion statistics."""
    print("\n" + "="*60)
    print("CONVERSION REPORT")
    print("="*60)
    print(f"   Output file:     {output_file}")
    print(f"\n   Total input:     {stats.total}")
    print(f"   Converted:       {stats.converted}")
    print(f"   Skipped:         {stats.skipped}")
    print(f"   Unanswerable:    {stats.unanswerable}")
    print("="*60 + "\n")


def _resolve_input_path(input_path: str) -> Path | None:
    """Resolve existing input path with compatibility for legacy `filtred` naming."""
    path = Path(input_path)
    if path.exists():
        return path

    legacy_candidates: list[Path] = []
    if "filtered" in path.name:
        legacy_candidates.append(path.with_name(path.name.replace("filtered", "filtred")))
    if "filtred" in path.name:
        legacy_candidates.append(path.with_name(path.name.replace("filtred", "filtered")))

    for candidate in legacy_candidates:
        if candidate.exists():
            print(f"Input file not found, using compatible path: {candidate}")
            return candidate
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert dataset to prompt/completion format")
    parser.add_argument("--input", type=str, required=True, help="Input filtered JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: <input>_prompt_completion.jsonl)")
    parser.add_argument("--language", type=str, choices=["en", "fr"], default="en",
                        help="Language for prompt template (default: en)")
    parser.add_argument("--format", type=str, choices=["prompt_completion", "chat"], default="prompt_completion",
                        help="Output format: 'prompt_completion' (default) or 'chat' (messages format)")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Exclude metadata (id, type, level) from output")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible citation style and refusal choices")
    parser.add_argument("--mixed-system-prompt", action="store_true",
                        help=f"For non-English data, use English system-prompt instructions on "
                             f"{int(MIXED_SYSTEM_EN_PROB*100)}%% of rows (query, docs, and answer stay in the data language). "
                             f"Trains the model to handle English instructions paired with non-English content.")

    args = parser.parse_args()

    resolved_input = _resolve_input_path(args.input)
    if resolved_input is None:
        print(f"Error: File not found: {args.input}")
        return

    if args.output:
        output_file = args.output
    else:
        input_path = resolved_input
        suffix = "_chat.jsonl" if args.format == "chat" else "_prompt_completion.jsonl"
        output_file = str(input_path.parent / f"{input_path.stem}{suffix}")

    print(f"Converting: {resolved_input}")
    print(f"Format: {args.format}")
    print(f"Language: {args.language}")
    print(f"Citation styles: {len(CITATION_STYLES)} variants, EOC ratio: {EOC_PROB:.0%}")
    print(f"Refusal variants: {len(UNANSWERABLE_REFUSALS.get(args.language, UNANSWERABLE_REFUSALS['en']))} per language")
    print(f"Specific refusal instruction ratio: {SPECIFIC_REFUSAL_INSTRUCTION_PROB:.0%} of unanswerable rows")
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    if args.format == "chat":
        stats = convert_to_chat_format(
            str(resolved_input), output_file,
            language=args.language,
            include_metadata=not args.no_metadata,
            seed=args.seed,
            mixed_system_prompt=args.mixed_system_prompt,
        )
    else:
        stats = convert_dataset(
            str(resolved_input),
            output_file,
            language=args.language,
            include_metadata=not args.no_metadata,
            seed=args.seed,
            mixed_system_prompt=args.mixed_system_prompt,
        )

    print_stats(stats, output_file)


if __name__ == "__main__":
    main()
