#!/usr/bin/env python3
"""Evaluate whether a model *generates* the correct typographic characters that
the original base training data had normalised away by ``ftfy.fix_text`` (curly
quotes, typographic apostrophes / elisions, full-width punctuation, ligatures,
...).

Two complementary, quantitative probes are run, broken down per *kind* of
character ("straight double quote", "curly double quote", "french guillemets",
"straight apostrophe", "curly apostrophe", "curly single quote",
"fullwidth parenthesis", "ascii parenthesis", "ligature fi/fl", ...):

1. CHARACTER GENERATION
   A prompt sets up a construct that requires a specific character, the model
   generates greedily, and we score the *decoded* continuation at the character
   level (so byte-fallback sequences are handled transparently). Two sub-cases:

   * Quotes / parentheses ("closing" probes). The prompt opens the construct a
     few words in (e.g. a French guillemet ``«`` must later be closed by ``»``);
     we scan for the first closing mark of the family. Per category:
         success : the first closing mark is the expected one
         failure : the first closing mark is a wrong variant of the family
         limit   : none appeared within the token budget (we don't know). The
                   budget (--max-new-tokens) is generous so this stays rare.

   * Apostrophes / elisions. The prompt ENDS just before an elision/contraction
     word (e.g. "… c'est précisément" → aujourd'hui), primed with the style under
     test, so the model generates the whole word itself — the apostrophe is
     bundled into the word's token and cannot be produced as a bare next token.
     We scan the continuation for the first apostrophe and score its glyph:
         success : the apostrophe is exactly the expected glyph
         failure : a different apostrophe-like mark (straight vs curly, backtick,
                   acute accent, left curly quote, ...)
         limit   : no apostrophe within the budget (no elision word produced).
     A unicode escape emitted as literal text (``\\u2019``) is scored by codepoint.

2. PROBABILITY CHECK (no generation, so no "limit")
   Right after a prompt, is the expected glyph the most probable among the
   candidate glyphs of its type?
   * Quotes / parentheses: the prompt is a COMPLETE sentence ending where the
     close belongs; we compare the summed next-token probability of each
     candidate closing glyph and check the expected one wins.
   * Apostrophes: for every vocabulary pair that has both a straight- and a
     curly-apostrophe token (e.g. ``▁jusqu'``/``▁jusqu’``, ``'t``/``’t``), in a
     context ending right before that word, we compare P(straight) vs P(curly).
     The "straight apostrophe" row counts a success when straight wins, "curly
     apostrophe" when curly wins (so the two rows are complementary). This
     sidesteps the fact that the pre-tokenizer bundles the elision apostrophe
     into the preceding token, which makes next-token *generation* of a bare
     apostrophe ill-posed.

The character detection is deliberately done on decoded *text*, not on token ids:
the reference list of ftfy-affected token ids (``Luciole-ftfy-tokens.txt`` next
to this script, overridable with ``--tokens-file``) is loaded only to annotate
which single-character glyphs are affected in the tested model's vocabulary,
because an affected character may also be emitted as a sequence of byte-fallback
tokens that no single id captures.

Models are HuggingFace checkpoints. They are loaded and tested ONE AT A TIME so a
list of models that do not all fit in memory can be compared in a single run.

Outputs (``--output-prefix``, default ``analyze_ftfy_generation``):
    <prefix>.json : every detail (config + per-model raw results + aggregates)
    <prefix>.md   : human-readable report, one column per model

The Markdown report is a pure function of the JSON, so it can be regenerated
without re-running the models:
    python analyze_ftfy_generation.py --from-json <prefix>.json

Each model may be given as a bare PATH (the table column is then the path
basename) or as LABEL=PATH to set a short, readable column name, e.g.
``Base=/lustre/.../step_0005959``. LABEL must not contain ``/``.

Usage
-----
    python analyze_ftfy_generation.py [LABEL=]MODEL1 [[LABEL=]MODEL2 ...] \
        [--output-prefix PATH] [--max-new-tokens N] [--batch-size N] \
        [--tokens-file PATH]
    python analyze_ftfy_generation.py --from-json run1.json [run2.json ...] \
        [--output-prefix PATH]        # merge JSON(s) into one report, no models

The Markdown report is a pure function of the JSON, so several runs (e.g. one
SLURM job per model batch) can be merged into a single table afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata

# Apostrophe generation prompts end just before the elision/contraction word; a
# few dozen tokens is enough for the model to produce that word (whose token
# carries the apostrophe) and to scan it.
APOSTROPHE_MAX_NEW_TOKENS = 24

# Default reference token-id list: looked up next to this script.
DEFAULT_TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Luciole-ftfy-tokens.txt"
)

# ----------------------------------------------------------------------------- #
# Character families and categories
# ----------------------------------------------------------------------------- #
# A "family" is the set of characters whose first appearance in a continuation
# ends the scan. The category's "expected" character is the correct member.
FAMILY_DOUBLE = set('"“”»›')  # closing double-quote-like glyphs
FAMILY_SINGLE = set("'’ʼ")  # apostrophe / single-quote closing glyphs
FAMILY_PAREN = set(")）")  # closing parenthesis glyphs

# Every glyph a model may plausibly emit *in place of* an apostrophe. Any of
# these at the elision site counts as "an apostrophe was produced" (so the
# outcome is success/failure by glyph, never an undetermined "limit"). Beyond the
# true apostrophes this includes common substitutes: backtick, acute/grave
# accents, prime marks, the left curly quote and the full-width apostrophe.
APOSTROPHE_LIKE = set("'’‘‛ʼʹ`´′＇")

# Reference sets (used only for the affected-glyph inventory / annotations).
AFFECTED_SINGLE_Q = set("‘’ʼ‛＇")
AFFECTED_DOUBLE_Q = set("“”„‟＂")
AFFECTED_LIGATURE = set("ﬀﬁﬂﬃﬄ")
AFFECTED_FULLWIDTH = set("　；：，！？．（）＋－０１２３４５６７８９～")


def _c(ch: str) -> str:
    """Readable label for a (possibly non-printing) character."""
    try:
        return f"{ch!r} (U+{ord(ch):04X} {unicodedata.name(ch)})"
    except (ValueError, TypeError):
        return repr(ch)


# --- closing-character generation probes -------------------------------------
# Probes are built so every prompt ends a few words *inside* an open construct:
# the model must continue and then CLOSE it, and the prompt primes the expected
# style. The closing character the model emits (scanned on decoded text) is
# compared with the expected one.
#
# Quotation sentences are SHARED across the four quote styles: the same lead-in +
# opening words are rendered once with straight double quotes, once with French
# guillemets, once with curly double quotes and once with curly single quotes, so
# the only difference between the four categories is the quote glyph itself.

# (lead_in, opening_words) -> prompt = lead_in + OPENER + opening_words
# Mixed French / English, varied topics and registers.
QUOTE_STEMS = [
    # --- French ---
    ("Le président a déclaré : ", "Nous devons agir dès maintenant pour le climat"),
    ("La ministre a ajouté : ", "Cette réforme changera le quotidien des familles"),
    ("Mon grand-père répétait souvent : ", "Le travail finit toujours par payer"),
    (
        "L’entraîneur a lancé à ses joueurs : ",
        "On ne lâche rien jusqu’au dernier coup de sifflet",
    ),
    ("Elle m’a murmuré à l’oreille : ", "Je crois que c’est enfin le bon moment"),
    ("Le guide a expliqué au groupe : ", "Ce château date du seizième siècle"),
    ("Le chef a prévenu en cuisine : ", "Il faut goûter la sauce avant de la servir"),
    (
        "Un passant a crié dans la rue : ",
        "Attention, la route est fermée un peu plus loin",
    ),
    ("La chercheuse a précisé : ", "Nos données montrent une tendance vraiment nette"),
    ("Le vieux marin racontait : ", "La tempête nous a surpris au large des côtes"),
    ("Le journaliste a écrit en une : ", "La décision a surpris tous les observateurs"),
    (
        "La pancarte à l’entrée indiquait : ",
        "Ne pas déranger avant huit heures du matin",
    ),
    (
        "Le médecin a conseillé : ",
        "Buvez beaucoup d’eau et reposez-vous quelques jours",
    ),
    (
        "Elle a soupiré avant de répondre : ",
        "Je n’aurais jamais imaginé une chose pareille",
    ),
    ("Le professeur a rappelé : ", "Lisez le chapitre entier avant le prochain cours"),
    (
        "Le témoin a affirmé aux policiers : ",
        "J’ai tout vu depuis la fenêtre de ma cuisine",
    ),
    (
        "La légende du village raconte : ",
        "Un trésor serait caché sous la vieille colline",
    ),
    ("Le maire a conclu son discours : ", "Ensemble, nous reconstruirons ce quartier"),
    # --- English ---
    ("She turned to me and said, ", "I have never seen anything quite like it"),
    ("The captain announced calmly, ", "We will be landing a little ahead of schedule"),
    (
        "My mother always told me, ",
        "Never put off until tomorrow what you can do today",
    ),
    ("The scientist explained, ", "The results went far beyond what we expected"),
    ("He shrugged and replied, ", "Honestly, it could have been a great deal worse"),
    ("The coach shouted from the bench, ", "Keep your eyes on the ball and just run"),
    ("The old man whispered, ", "There is something you really ought to know"),
    ("The notice on the gate read, ", "Please remember to close the door behind you"),
    ("The review opened with, ", "This is easily the finest film of the year"),
    ("The detective muttered to himself, ", "None of this adds up the way it should"),
    ("The waiter said politely, ", "May I recommend the grilled fish this evening"),
    ("A voice on the radio warned, ", "A powerful storm is approaching from the west"),
    (
        "The teacher reminded the class, ",
        "The exam will cover everything from chapter three",
    ),
    ("The founder insisted, ", "Our priority has always been the customer first"),
    ("Grandma liked to say, ", "A warm kitchen makes for a happy home"),
    ("The tour guide noted, ", "This bridge was built more than a century ago"),
    (
        "The pilot reassured the cabin, ",
        "There is really nothing at all to worry about",
    ),
    ("The poet once wrote, ", "The night is always darkest just before the dawn"),
]

# category -> (opener_string, expected_closing_char, apostrophe_style)
# The apostrophe style restyles any apostrophe inside the shared stems so it
# matches the quote style: straight quotes get a straight apostrophe, the curly /
# guillemet styles get a curly one.
QUOTE_STYLES = {
    "straight double quote": ('"', '"', "'"),
    "french guillemets": ("« ", "»", "’"),
    "curly double quote": ("“", "”", "’"),
    "curly single quote": ("‘", "’", "’"),
}

# Apostrophe / elision probes. Each sentence primes the style with earlier
# apostrophes ("{A}") and ENDS on a strong elision/contraction trigger WITHOUT
# the apostrophe, which the model is expected to generate ("jusqu" -> "jusqu'à",
# "aujourd" -> "aujourd'hui", "wasn" -> "wasn't", ...). "{A}" is substituted with
# a straight (') or curly (’) apostrophe per category.
APO_STEMS = [
    # --- French --- (each ends just before the commented elision word)
    "— Quand a lieu le rendez-vous ? — Je peux te le confirmer, c{A}est très précisément",  # aujourd'hui
    "Je n{A}ai rien pu faire hier, alors je m{A}en occupe dès",  # aujourd'hui
    "Ce dossier ne peut plus attendre : il l{A}a dit, il faut le terminer",  # aujourd'hui
    "Dans la pièce plongée dans le noir, il s{A}arrêta, certain de sentir la présence de",  # quelqu'un
    "On a frappé à la porte : il m{A}a semblé qu{A}il y avait",  # quelqu'un
    "Il ne demandait l{A}aide de personne, et pourtant il espérait toujours",  # quelqu'un
    "Reliée au continent par un isthme étroit, c{A}est en réalité une véritable",  # presqu'île
    "En Bretagne, nous avons passé tout l{A}été sur une magnifique",  # presqu'île
    "Il s{A}est battu avec un courage admirable, il a tenu bon",  # jusqu'au bout
    "Le coureur était à bout de forces, mais il n{A}a pas renoncé et a continué",  # jusqu'à la fin
    "Le roman est si prenant qu{A}on ne le lâche plus",  # jusqu'à la dernière page
    "Ce commerce n{A}observe aucune pause le midi : il reste ouvert",  # jusqu'au soir
    "Il a travaillé sans relâche, c{A}est vrai, et ce",  # jusqu'à l'aube
    "La route, qu{A}on devine à peine, grimpe en lacets serrés",  # jusqu'au sommet
    "Pour me désaltérer, je n{A}ai demandé qu{A}un grand verre",  # d'eau
    "Mourant de soif, l{A}enfant réclamait sans cesse encore un peu",  # d'eau
    "Chaque matin, avant même de s{A}habiller, posément, il ouvre",  # l'armoire
    "Intriguée, elle n{A}a pas hésité : aussitôt elle a décacheté",  # l'enveloppe
    "À mon humble avis, et je l{A}affirme très sincèrement,",  # c'est
    "Ne te fie pas aux apparences : au fond, j{A}en suis sûr,",  # c'est
    "Peu importe ce qu{A}on en pense, moi, je crois vraiment",  # qu'il
    "Il n{A}a rien voulu nous expliquer ; j{A}en déduis donc",  # qu'il
    "Dès la première page de ce roman, on comprend qu{A}il",  # s'agit
    "Près du feu, repu, le vieux chat qu{A}elle adore, peu à peu,",  # s'endort
    "Quand je suis entré, sans un mot, il s{A}est levé ; puis il",  # m'a
    "Je lui ai posé la question et, sans l{A}ombre d{A}une hésitation, il",  # m'a répondu
    "Tant que tu n{A}auras pas vraiment compris, je le répéterai",  # jusqu'à
    "Elle a relu l{A}article, l{A}a corrigé, puis elle l{A}a retravaillé",  # jusqu'à
    # --- English contractions --- (each ends just before the commented word)
    "I{A}ve searched every single drawer, but the key simply",  # wasn't
    "They{A}d all assumed he knew the answer, but it turned out he",  # didn't
    "She{A}d been yawning all morning long; it was clear she",  # couldn't
    "They swore they{A}d call us right back, but so far they",  # haven't
    "We{A}ve really got to hurry now, because otherwise we",  # won't
    "Relax, there{A}s no rush at all; honestly, you",  # don't
    "Grab an umbrella — it{A}s grey outside — because later",  # it'll
    "Come on, let{A}s get going, or the two of us",  # we'll
    "Yes, of course — that{A}s perfectly fine — just give me a second,",  # I'm
    "Leave the dishes where they are, don{A}t worry,",  # I'll
    "After everything that{A}s happened this year, I can finally say",  # I've
    "I know you{A}d rather not hear it, but honestly you",  # you're
]
APO_STYLES = {"straight apostrophe": "'", "curly apostrophe": "’"}

# Parenthesis probes (not shared: distinct scripts). Each prompt opens a
# parenthetical the model is expected to close.
ASCII_PAREN_PROMPTS = [
    "The national museum (originally founded in 1793",
    "He was born in Lyon (a city in central France",
    "This recipe needs two cups of flour (roughly 240 grams",
    "Our latest model (released earlier this spring",
    "La Révolution française (qui débuta en 1789",
    "Le mont Blanc (le plus haut sommet des Alpes",
    "The effect was statistically significant (p below 0.05",
    "She studied molecular biology (with a focus on genetics",
    "Le train part tous les jours à midi (sauf le dimanche",
    "The library stays open late (except on public holidays",
    "Marie Curie (the first woman to win a Nobel Prize",
    "The algorithm runs in linear time (that is, O of n",
    "Il a grandi à Marseille (dans le sud de la France",
    "The treaty was finally signed (after months of negotiation",
    "Add a pinch of salt (about half a teaspoon",
    "The company was founded by two students (both engineers",
    "Victor Hugo (l’auteur des Misérables",
    "The engine produces 300 horsepower (roughly 220 kilowatts",
    "Notre réunion est reportée à jeudi (en fin de journée",
    "The planet Mars (often called the red planet",
    "You should back up your files (ideally every single day",
    "Le musée du Louvre (le plus visité au monde",
    "The experiment was repeated three times (to ensure accuracy",
    "His first novel (written when he was just twenty",
    "La tour Eiffel (achevée en 1889",
    "The software is free to use (under an open licence",
    "Beethoven composed nine symphonies (the last one choral",
    "Cette plante a besoin de lumière (mais pas en plein soleil",
    "The bridge spans two kilometres (making it one of the longest",
    "We met in Berlin (the capital of Germany",
]
FULLWIDTH_PAREN_PROMPTS = [
    "東京（とうきょう）は日本の首都です。大阪（おおさか",
    "富士山（ふじさん）はとても有名です。琵琶湖（びわこ",
    "彼女は数学（すうがく）が得意で、物理（ぶつり",
    "漢字（かんじ）とひらがな（",
    "明治（めいじ）時代のあとは、大正（たいしょう",
    "北海道（ほっかいどう）は広く、沖縄（おきなわ",
    "日本語（にほんご）を勉強し、中国語（ちゅうごくご",
    "桜（さくら）の季節のあとに、梅雨（つゆ",
    "彼の名字は田中（たなか）で、友達は鈴木（すずき",
    "京都（きょうと）には古い寺（てら",
    "試験（しけん）の範囲は、第三章（だいさんしょう",
    "母（はは）と父（ちち",
    "電車（でんしゃ）ではなく、自転車（じてんしゃ",
    "彼は医者（いしゃ）で、弟は弁護士（べんごし",
    "春（はる）と夏（なつ",
    "会議（かいぎ）は火曜日（かようび",
    "東京駅（とうきょうえき）から新大阪（しんおおさか",
    "日本（にほん）の人口（じんこう",
    "犬（いぬ）と猫（ねこ",
    "先生（せんせい）が黒板（こくばん",
    "山田さん（やまだ）は会社員（かいしゃいん",
    "水曜日（すいようび）ではなく、木曜日（もくようび",
    "辞書（じしょ）で意味（いみ",
    "彼女の名前（なまえ）は花子（はなこ",
    "朝食（ちょうしょく）のあと、昼食（ちゅうしょく",
    "図書館（としょかん）と美術館（びじゅつかん",
]


def _family_of(ch: str) -> str:
    """Return the scan-stop set (as a string) matching the expected character."""
    if ch in FAMILY_DOUBLE:
        return "".join(sorted(FAMILY_DOUBLE))
    if ch in FAMILY_SINGLE:
        return "".join(sorted(FAMILY_SINGLE))
    if ch in FAMILY_PAREN:
        return "".join(sorted(FAMILY_PAREN))
    raise ValueError(f"no family for {ch!r}")


def build_gen_probes() -> dict:
    """Assemble the category -> {mode, expected, stop, prompts} mapping, sharing
    the quotation sentences across all quote styles."""
    probes: dict[str, dict] = {}
    # Quote styles (shared stems). mode="quote": intra-word apostrophes inside the
    # quoted text are ignored so only the word-boundary closing mark is scored.
    # Any apostrophe in the stem is restyled to match the quote style.
    for cat, (opener, expected, apo) in QUOTE_STYLES.items():
        prompts = [
            (lead + opener + words).replace("’", apo) for lead, words in QUOTE_STEMS
        ]
        probes[cat] = {
            "mode": "quote",
            "expected": expected,
            "stop": _family_of(expected),
            "prompts": prompts,
        }
    # Apostrophe styles (shared stems). mode="apostrophe": the first apostrophe
    # the model emits (an elision/contraction, so followed by a letter) is scored.
    for cat, apo in APO_STYLES.items():
        prompts = [s.replace("{A}", apo) for s in APO_STEMS]
        probes[cat] = {
            "mode": "apostrophe",
            "expected": apo,
            "stop": _family_of(apo),
            "prompts": prompts,
        }
    # Parentheses.
    probes["fullwidth parenthesis"] = {
        "mode": "paren",
        "expected": "）",
        "stop": _family_of("）"),
        "prompts": list(FULLWIDTH_PAREN_PROMPTS),
    }
    probes["ascii parenthesis"] = {
        "mode": "paren",
        "expected": ")",
        "stop": _family_of(")"),
        "prompts": list(ASCII_PAREN_PROMPTS),
    }
    return probes


GEN_PROBES = build_gen_probes()


def build_prob_close_probes() -> dict:
    """category -> {expected, candidates, prompts} for the closing-glyph part of
    the probability check (complete-sentence prompts; quote styles share stems)."""
    probes: dict[str, dict] = {}
    for cat, (opener, expected, apo) in QUOTE_STYLES.items():
        prompts = [
            (lead + opener + content).replace("’", apo)
            for lead, content in CLOSE_PROBE_STEMS
        ]
        probes[cat] = {
            "expected": expected,
            "candidates": _family_of(expected),
            "prompts": prompts,
        }
    probes["fullwidth parenthesis"] = {
        "expected": "）",
        "candidates": _family_of("）"),
        "prompts": list(FULLWIDTH_PAREN_PROBE_PROMPTS),
    }
    probes["ascii parenthesis"] = {
        "expected": ")",
        "candidates": _family_of(")"),
        "prompts": list(ASCII_PAREN_PROBE_PROMPTS),
    }
    return probes


# ============================================================================= #
# Probability-check probes (replace the old log-likelihood test)
# ============================================================================= #
# The probability check never generates: it reads the next-token distribution
# right after the prompt and asks whether the expected glyph is the most probable
# *among the candidate glyphs of its type*. There is therefore no "limit".
#
# (A) Closing glyphs (quotes / parentheses). The prompt is a COMPLETE sentence
# that ends exactly where the closing mark belongs, so the close is the natural
# next token. Quote sentences are shared across the four styles (apostrophes
# restyled), like QUOTE_STEMS.
CLOSE_PROBE_STEMS = [
    ("Le guide a expliqué : ", "Ce monument date du seizième siècle."),
    ("Elle a répondu calmement : ", "Je reviendrai sans faute demain matin."),
    ("Le président a déclaré : ", "Nous devons agir dès maintenant."),
    ("Mon oncle disait toujours : ", "Le temps finit par tout arranger."),
    ("La note à l’accueil précisait : ", "Le musée est fermé le lundi."),
    ("Le médecin a conseillé : ", "Reposez-vous et buvez beaucoup d’eau."),
    ("Le capitaine a annoncé : ", "Nous atterrirons avec un peu d’avance."),
    ("La chercheuse a conclu : ", "Les résultats sont clairs et cohérents."),
    ("The sign on the door read, ", "Please do not feed the animals."),
    ("She smiled and said, ", "I have never been happier in my life."),
    ("The teacher reminded us, ", "The exam begins at nine o’clock sharp."),
    ("He whispered to me, ", "We really should leave before it gets dark."),
]

# Parenthetical prompts whose content is complete, so the close is the next token.
ASCII_PAREN_PROBE_PROMPTS = [
    "Il est né à Lyon (une ville du centre de la France",
    "Le musée a été fondé il y a longtemps (en 1793",
    "The treaty was finally signed (after months of difficult negotiation",
    "Marie Curie (the first woman ever to win a Nobel Prize",
    "Ajoutez une pincée de sel (environ une demi-cuillère",
    "The planet Mars (often called the red planet",
    "La tour Eiffel (achevée en 1889",
    "He studied molecular biology (with a particular focus on genetics",
    "Notre prochaine réunion est reportée (à jeudi en fin de journée",
    "The library stays open late every day (except on public holidays",
    "Victor Hugo (l’auteur des Misérables",
    "The engine produces three hundred horsepower (about 220 kilowatts",
]
FULLWIDTH_PAREN_PROBE_PROMPTS = [
    "日本の首都は東京（とうきょう",
    "富士山（ふじさん",
    "彼女は数学（すうがく",
    "漢字（かんじ",
    "明治（めいじ",
    "北海道（ほっかいどう",
    "日本語（にほんご",
    "京都（きょうと",
    "彼の名字は田中（たなか",
    "試験の範囲は第三章（だいさんしょう",
    "電車（でんしゃ",
    "彼は医者（いしゃ",
]

# (B) Apostrophe preference. For each vocabulary token that is a word written with
# a curly apostrophe (▁ marks a word-initial token), a context that ends right
# before it, so the elision/contraction word is the natural next token. If the
# straight-apostrophe twin is also a token, we compare P(straight) vs P(curly).
# Keyed by the CURLY token string.
APOSTROPHE_PREF_CONTEXTS = {
    # French word-initial elisions (lowercase)
    "▁l’": "Chaque matin, en arrivant, il ouvre",
    "▁d’": "Au petit déjeuner, je bois toujours un grand verre",
    "▁j’": "Hier soir, après le dîner,",
    "▁m’": "Quand je suis entré, aussitôt il",
    "▁n’": "Malgré tous ses efforts, il",
    "▁s’": "Peu à peu, le vieux chien",
    "▁t’": "Ne bouge pas, je crois qu’il",
    "▁c’": "Franchement, à mon avis,",
    "▁qu’": "Au fond de moi, je crois",
    "▁jusqu’": "Nous avons marché le long de la plage",
    "▁lorsqu’": "Je te donnerai ma réponse",
    "▁puisqu’": "Nous partirons sans lui,",
    "▁aujourd’hui": "Le grand rendez-vous est prévu pour",
    # French word-initial elisions (capitalised, sentence start)
    "▁C’": "Il m’a enfin tout avoué.",
    "▁L’": "Tout le monde attendait dehors.",
    "▁J’": "On m’a posé la question.",
    "▁N’": "Il hésita un long moment.",
    "▁S’": "Le ciel se couvrait lentement.",
    "▁D’": "Le débat était loin d’être clos.",
    "▁Qu’": "Personne ne savait quoi faire.",
    "▁Jusqu’": "La route montait sans cesse.",
    "▁Lorsqu’": "Le silence était total.",
    "▁Aujourd’hui": "Hier, il pleuvait sans arrêt.",
    # English contractions (suffix tokens)
    "’t": "She tried her best but she really wasn",
    "’s": "I honestly think it",
    "’re": "I am quite sure that you",
    "’ve": "After all this time, they",
    "’ll": "Do not worry about it, I",
    "’m": "Yes, of course, I",
    "’d": "If I had known earlier, I",
}


# ----------------------------------------------------------------------------- #
# Affected-token inventory (reference list)
# ----------------------------------------------------------------------------- #
def load_affected_ids(path: str) -> list[int]:
    """Parse the merge-tool token-id file (one int per line, '#' comments)."""
    ids: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(int(line))
    return ids


def affected_glyph_inventory(tok, ids: list[int]) -> dict[str, list[int]]:
    """Map each affected single-character glyph to the ids that decode to it,
    using the tested model's own tokenizer (so byte-fallback is resolved)."""
    inv: dict[str, list[int]] = {}
    for i in ids:
        try:
            s = tok.decode([i])
        except Exception:
            continue
        s = s.replace("▁", " ").replace("▁", " ")
        for ch in s:
            if ch in (
                AFFECTED_SINGLE_Q
                | AFFECTED_DOUBLE_Q
                | AFFECTED_LIGATURE
                | AFFECTED_FULLWIDTH
            ):
                inv.setdefault(ch, []).append(i)
    return inv


# ----------------------------------------------------------------------------- #
# Generation / scoring (require a model)
# ----------------------------------------------------------------------------- #
def _flatten_gen_items() -> list[dict]:
    items = []
    for cat, spec in GEN_PROBES.items():
        for p in spec["prompts"]:
            items.append(
                {
                    "category": cat,
                    "prompt": p,
                    "expected": spec["expected"],
                    "stop": spec["stop"],
                    "mode": spec["mode"],
                }
            )
    return items


def _leading_escape(s: str) -> tuple[str, str] | tuple[None, str]:
    """If `s` starts with a literal unicode escape the model emitted as text
    (e.g. the six characters ``\\u2019`` rather than the character ’), decode it.
    Returns (decoded_char, raw_escape_text) or (None, '')."""
    mobj = re.match(r"\\u([0-9a-fA-F]{4})", s) or re.match(r"\\U([0-9a-fA-F]{8})", s)
    if mobj:
        try:
            return chr(int(mobj.group(1), 16)), mobj.group(0)
        except (ValueError, OverflowError):
            pass
    return None, ""


def classify_apostrophe(continuation: str, expected: str) -> tuple[str, str]:
    """Scan the continuation for the FIRST apostrophe-like character the model
    emits and score its style. The prompt ends just before an elision/contraction
    word, so the model generates the whole word (the apostrophe is bundled into
    the word's token); we read the apostrophe out of that word.

      success : the apostrophe is exactly the expected glyph
      failure : it is a different apostrophe-like mark (straight vs curly, a
                backtick, an acute accent, the left curly quote, ...)
      limit   : no apostrophe appeared within the token budget (the model did not
                produce an elision/contraction word)

    A unicode escape emitted as literal text (``\\u2019``) is decoded and scored
    by its codepoint; the raw escape is returned so the report shows it.
    """
    i, n = 0, len(continuation)
    while i < n:
        if continuation[i] == "\\":
            dec, raw = _leading_escape(continuation[i:])
            if dec is not None:
                if dec in APOSTROPHE_LIKE:
                    return ("success" if dec == expected else "failure", raw)
                i += len(raw)
                continue
        ch = continuation[i]
        if ch in APOSTROPHE_LIKE:
            return ("success" if ch == expected else "failure", ch)
        i += 1
    return ("limit", "")


def classify_closing(
    continuation: str, expected: str, stop: str, skip_intraword: bool
) -> tuple[str, str]:
    """Scan for the first CLOSING mark (quotes / parentheses).

    When `skip_intraword` (quotes), a stop glyph immediately followed by a letter
    is treated as inside the quoted text (e.g. the ' in "c'est", or a nested
    opening quote) and skipped, so only the word-boundary closer is scored.
    """
    stopset = set(stop)
    for i, ch in enumerate(continuation):
        if ch in stopset:
            if skip_intraword:
                nxt = continuation[i + 1] if i + 1 < len(continuation) else ""
                if nxt.isalpha():
                    continue
            return ("success" if ch == expected else "failure", ch)
    return ("limit", "")


def classify_continuation(
    continuation: str, expected: str, stop: str, mode: str
) -> tuple[str, str]:
    """Dispatch to the mode-specific scorer. outcome in {success, failure, limit}
    ("limit" only for quote/paren modes)."""
    if mode == "apostrophe":
        return classify_apostrophe(continuation, expected)
    return classify_closing(
        continuation, expected, stop, skip_intraword=(mode == "quote")
    )


def run_generation(model, tok, device, batch_size, closing_budget):
    """Generate and score every probe, category by category so each category uses
    its own token budget: quotes/parentheses get `closing_budget` (they must
    produce content and then close), apostrophes get APOSTROPHE_MAX_NEW_TOKENS
    (scored at the start of the continuation)."""
    import torch

    results = []
    # Preserve the tokenizer's padding config, restore afterwards.
    saved_side, saved_pad = tok.padding_side, tok.pad_token
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    total = sum(len(spec["prompts"]) for spec in GEN_PROBES.values())
    done = 0
    for cat, spec in GEN_PROBES.items():
        budget = (
            APOSTROPHE_MAX_NEW_TOKENS
            if spec["mode"] == "apostrophe"
            else closing_budget
        )
        prompts = spec["prompts"]
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=budget,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id,
                )
            gen = out[:, enc["input_ids"].shape[1] :]
            for prompt, g in zip(chunk, gen):
                cont = tok.decode(g, skip_special_tokens=True)
                outcome, first = classify_continuation(
                    cont, spec["expected"], spec["stop"], spec["mode"]
                )
                results.append(
                    {
                        "category": cat,
                        "prompt": prompt,
                        "expected": spec["expected"],
                        "continuation": cont,
                        "first_close": first,
                        "outcome": outcome,
                    }
                )
            done += len(chunk)
            print(f"    generation {done}/{total}", flush=True)

    tok.padding_side, tok.pad_token = saved_side, saved_pad
    return results


def _build_glyph_token_ids(tok, glyphs):
    """glyph -> set of vocab ids whose decoded surface, with the leading
    metaspace/space stripped, STARTS WITH that glyph (i.e. tokens that would make
    the glyph the next visible character)."""
    glyphset = set(glyphs)
    out = {g: set() for g in glyphset}
    for tokstr, tid in tok.get_vocab().items():
        s = tokstr.replace("▁", " ").lstrip()
        if s and s[0] in glyphset:
            out[s[0]].add(tid)
    return out


def _next_token_probs(model, tok, device, prompts, batch_size):
    """Next-token probability distribution right after each prompt (left-padded
    batches). Returns a list of 1-D CPU tensors, one per prompt."""
    import torch

    saved_side, saved_pad = tok.padding_side, tok.pad_token
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        out.extend(list(torch.softmax(logits, dim=-1).cpu()))
    tok.padding_side, tok.pad_token = saved_side, saved_pad
    return out


def run_probability_check(model, tok, device, batch_size):
    """Right after each prompt, is the expected glyph the most probable among the
    candidate glyphs of its type? (no generation, so no 'limit'.)

      * quotes / parentheses: complete-sentence prompts; compare the summed
        next-token probability of each candidate closing glyph.
      * apostrophes: compare P(straight-apostrophe token) vs P(curly token) for
        every vocabulary pair that has both, in a context ending right before the
        elision/contraction word. The 'straight apostrophe' row counts a success
        when straight wins, 'curly apostrophe' when curly wins.

    Returns (results_list, by_cat_dict)."""
    vocab = tok.get_vocab()
    results = []

    # (A) closing glyphs
    close_probes = build_prob_close_probes()
    all_glyphs = set()
    for spec in close_probes.values():
        all_glyphs |= set(spec["candidates"])
    glyph_ids = _build_glyph_token_ids(tok, all_glyphs)
    for cat, spec in close_probes.items():
        cands = list(spec["candidates"])
        for prompt, p in zip(
            spec["prompts"],
            _next_token_probs(model, tok, device, spec["prompts"], batch_size),
        ):
            gp = {
                g: (float(p[list(glyph_ids[g])].sum()) if glyph_ids[g] else 0.0)
                for g in cands
            }
            best = max(gp, key=gp.get)
            results.append(
                {
                    "category": cat,
                    "kind": "close",
                    "prompt": prompt,
                    "expected": spec["expected"],
                    "top_glyph": best,
                    "success": bool(best == spec["expected"]),
                    "glyph_probs": gp,
                }
            )

    # (B) apostrophe preference (only pairs that exist in this vocab)
    pairs = []
    for curly_str, context in APOSTROPHE_PREF_CONTEXTS.items():
        straight_str = curly_str.replace("’", "'")
        if curly_str in vocab and straight_str in vocab:
            pairs.append(
                (
                    context,
                    straight_str,
                    vocab[straight_str],
                    curly_str,
                    vocab[curly_str],
                )
            )
    if pairs:
        probs = _next_token_probs(
            model, tok, device, [c for c, *_ in pairs], batch_size
        )
        for (context, sstr, sid, cstr, cid), p in zip(pairs, probs):
            ps, pc = float(p[sid]), float(p[cid])
            for cat, want_curly in (
                ("straight apostrophe", False),
                ("curly apostrophe", True),
            ):
                ok = (pc > ps) if want_curly else (ps > pc)
                results.append(
                    {
                        "category": cat,
                        "kind": "apostrophe",
                        "prompt": context,
                        "pair": [sstr, cstr],
                        "p_straight": ps,
                        "p_curly": pc,
                        "success": bool(ok),
                    }
                )
    return results, aggregate_probcheck(results)


def aggregate_probcheck(results):
    by_cat = {}
    for r in results:
        d = by_cat.setdefault(r["category"], {"success": 0, "total": 0})
        d["success"] += int(r["success"])
        d["total"] += 1
    for d in by_cat.values():
        d["success_rate"] = d["success"] / d["total"] if d["total"] else float("nan")
    return by_cat


# ----------------------------------------------------------------------------- #
# Aggregation
# ----------------------------------------------------------------------------- #
def aggregate_generation(gen_results):
    by_cat = {}
    for r in gen_results:
        d = by_cat.setdefault(
            r["category"], {"success": 0, "failure": 0, "limit": 0, "total": 0}
        )
        d[r["outcome"]] += 1
        d["total"] += 1
    for d in by_cat.values():
        decided = d["success"] + d["failure"]
        d["success_rate"] = d["success"] / decided if decided else float("nan")
    overall = {"success": 0, "failure": 0, "limit": 0, "total": 0}
    for d in by_cat.values():
        for k in ("success", "failure", "limit", "total"):
            overall[k] += d[k]
    decided = overall["success"] + overall["failure"]
    overall["success_rate"] = overall["success"] / decided if decided else float("nan")
    return by_cat, overall


# ----------------------------------------------------------------------------- #
# Markdown rendering (pure function of the JSON payload)
# ----------------------------------------------------------------------------- #
def _fmt_pct(x):
    return "n/a" if x != x else f"{100 * x:.0f}%"


def _fmt_num(x, nd=3):
    return "n/a" if x != x else f"{x:.{nd}f}"


def _probcheck_ratio(r):
    """For one probability-check probe, P(expected) / max P(other candidates).
    >1 means the expected glyph/token beats its strongest rival. Returns None if
    the best competitor has zero probability (ratio undefined)."""
    if r["kind"] == "close":
        gp = r["glyph_probs"]
        exp = r["expected"]
        others = [v for g, v in gp.items() if g != exp]
        best = max(others) if others else 0.0
        return gp.get(exp, 0.0) / best if best > 0 else None
    # apostrophe: the only competitor is the opposite-style token
    if r["category"] == "curly apostrophe":
        num, den = r["p_curly"], r["p_straight"]
    else:
        num, den = r["p_straight"], r["p_curly"]
    return num / den if den > 0 else None


def _mean_ratio(probes, category):
    """Geometric mean of the per-probe ratios (robust to a single probe whose
    best competitor has near-zero probability)."""
    vals = [
        v
        for r in probes
        if r["category"] == category and (v := _probcheck_ratio(r)) is not None
    ]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(max(v, 1e-12)) for v in vals) / len(vals))


def _gen_rate(d, limit_as_failure):
    """Generation success rate for one category dict. limit_as_failure=False
    ignores limit cases (success / decided); True counts them as failures
    (success / total)."""
    if not d:
        return float("nan")
    denom = d["total"] if limit_as_failure else (d["success"] + d["failure"])
    return d["success"] / denom if denom else float("nan")


def build_markdown(payload: dict) -> str:
    order = payload["model_order"]
    models = payload["models"]
    cfg = payload["config"]
    gen_cats = cfg["generation_categories"]
    prob_cats = cfg.get("probcheck_categories", [])
    lines = []
    a = lines.append

    a("# ftfy typographic-generation analysis\n")
    a(f"- closing-probe token budget: **{cfg['max_new_tokens']}**")
    a(
        f"- generation prompts: **{cfg['n_generation_items']}**, "
        f"probability-check probes: **{cfg.get('n_probcheck_items', 0)}**\n"
    )
    a("Models:\n")
    for m in order:
        a(f"- **{m}** — `{models[m]['path']}`")
    a("")

    # --- overall ---
    a("## 1. Overall generation\n")
    a(
        "`success` = expected glyph produced; `failure` = wrong variant; `limit` = "
        "nothing of the family within the token budget (for apostrophes, the model "
        "produced no elision/contraction word).\n"
    )
    a("| metric | " + " | ".join(order) + " |")
    a("|" + "---|" * (len(order) + 1))
    for key, label in [
        ("success", "success"),
        ("failure", "failure"),
        ("limit", "limit"),
    ]:
        row = [label] + [str(models[m]["generation_overall"][key]) for m in order]
        a("| " + " | ".join(row) + " |")
    row = ["**rate (ignoring limit)**"] + [
        _fmt_pct(_gen_rate(models[m]["generation_overall"], False)) for m in order
    ]
    a("| " + " | ".join(row) + " |")
    row = ["**rate (limit = failure)**"] + [
        _fmt_pct(_gen_rate(models[m]["generation_overall"], True)) for m in order
    ]
    a("| " + " | ".join(row) + " |")
    a("")

    # --- section 2: three per-category tables ---
    a("## 2. By character kind\n")

    def gen_table(title, limit_as_failure, note):
        a(f"### {title}\n")
        a("| category | " + " | ".join(order) + " |")
        a("|" + "---|" * (len(order) + 1))
        for cat in gen_cats:
            row = [cat]
            for m in order:
                d = models[m]["generation_by_category"].get(cat)
                row.append(_fmt_pct(_gen_rate(d, limit_as_failure)))
            a("| " + " | ".join(row) + " |")
        a(f"\n_{note}_\n")

    gen_table(
        "2a. Generation success rate — ignoring limit cases",
        False,
        "success / (success + failure); prompts where no glyph of the family was "
        "produced in the token budget are excluded.",
    )
    gen_table(
        "2b. Generation success rate — limit counted as failure",
        True,
        "success / total; not closing within the budget counts against the model.",
    )

    # 2c. probability check
    a("### 2c. Probability check — expected glyph most probable among candidates\n")
    a(
        "No generation: right after a complete prompt, is the expected glyph the "
        "most probable among the candidate glyphs of its type? For apostrophes this "
        "is a straight-vs-curly token preference (the two rows are complementary).\n"
    )
    a("| category | " + " | ".join(order) + " |")
    a("|" + "---|" * (len(order) + 1))
    for cat in prob_cats:
        row = [cat]
        for m in order:
            d = models[m].get("probcheck_by_category", {}).get(cat)
            row.append(_fmt_pct(d["success_rate"]) if d else "n/a")
        a("| " + " | ".join(row) + " |")
    a("")

    # 2d. probability margin
    a("### 2d. Probability margin — geometric mean of P(expected) / best competitor\n")
    a(
        "Geometric mean over probes of the expected glyph's probability divided by "
        "the highest probability among the other candidate glyphs of its type (for "
        "apostrophes, the opposite-style token). **> 1** = the expected glyph is on "
        "average more probable than its strongest rival; **< 1** = a rival wins. "
        "(Geometric so one near-zero-competitor probe can't dominate.)\n"
    )
    a("| category | " + " | ".join(order) + " |")
    a("|" + "---|" * (len(order) + 1))
    for cat in prob_cats:
        row = [cat]
        for m in order:
            row.append(_fmt_num(_mean_ratio(models[m].get("probcheck", []), cat), 2))
        a("| " + " | ".join(row) + " |")
    a("\n_Per-probe detail and the affected-glyph inventory are in the JSON._\n")
    return "\n".join(lines)


def merge_payloads(payloads: list[dict]) -> dict:
    """Combine several result payloads into one (models concatenated in order).

    The category lists are the union across files (first-seen order), so reports
    produced by different script versions still line up. Scalar config values
    (max_new_tokens, counts, ...) are taken from the first file. Colliding model
    labels are disambiguated with a '#N' suffix.
    """
    if len(payloads) == 1:
        return payloads[0]

    gen_cats, prob_cats = [], []
    merged_models: dict[str, dict] = {}
    order, seen = [], {}
    for pl in payloads:
        for cat in pl["config"].get("generation_categories", []):
            if cat not in gen_cats:
                gen_cats.append(cat)
        for cat in pl["config"].get("probcheck_categories", []):
            if cat not in prob_cats:
                prob_cats.append(cat)
        for label in pl["model_order"]:
            lab = label
            if lab in seen:
                seen[lab] += 1
                lab = f"{lab}#{seen[lab]}"
            else:
                seen[lab] = 0
            order.append(lab)
            merged_models[lab] = pl["models"][label]

    cfg = dict(payloads[0]["config"])
    cfg["generation_categories"] = gen_cats
    cfg["probcheck_categories"] = prob_cats
    cfg["merged_from"] = len(payloads)
    return {"config": cfg, "model_order": order, "models": merged_models}


# ----------------------------------------------------------------------------- #
# Driver
# ----------------------------------------------------------------------------- #
def analyze_one_model(path, label, device, max_new_tokens, batch_size, affected_ids):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[loading {label}] {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    # Newer transformers use `dtype=`, older ones `torch_dtype=`; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    print("  running closing-character generation ...", flush=True)
    gen = run_generation(model, tok, device, batch_size, max_new_tokens)
    gen_by_cat, gen_overall = aggregate_generation(gen)

    print("  running probability check ...", flush=True)
    prob, prob_by_cat = run_probability_check(model, tok, device, batch_size)

    inventory = {}
    if affected_ids:
        inv = affected_glyph_inventory(tok, affected_ids)
        inventory = {g: ids for g, ids in inv.items()}

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[done {label}] gen rate (ignoring limit) = "
        f"{_fmt_pct(_gen_rate(gen_overall, False))}",
        flush=True,
    )

    return {
        "path": path,
        "generation": gen,
        "generation_by_category": gen_by_cat,
        "generation_overall": gen_overall,
        "probcheck": prob,
        "probcheck_by_category": prob_by_cat,
        "affected_inventory": inventory,
    }


def split_label(arg: str) -> tuple[str | None, str]:
    """Parse a ``label=path`` model argument into (label, path).

    A leading ``label=`` is recognised only when the text before the first ``=``
    contains no ``/``; otherwise the whole argument is a path (so a path that
    itself contains ``=``, e.g. ``.../step=0005959``, is left intact and gets no
    label). Returns (None, path) when no label is given.
    """
    head, sep, tail = arg.partition("=")
    if sep and "/" not in head and head and tail:
        return head, tail
    return None, arg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "models",
        nargs="*",
        help="Model(s) as PATH or LABEL=PATH (LABEL becomes the table column; "
        "a bare PATH falls back to its basename). LABEL must not contain '/'.",
    )
    ap.add_argument(
        "--output-prefix",
        default="analyze_ftfy_generation",
        help="Output path prefix for <prefix>.json and <prefix>.md",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Token budget for the closing (quote/parenthesis) generation probes, "
        "which must generate content and then close (default 400). Apostrophe "
        f"generation probes always use {APOSTROPHE_MAX_NEW_TOKENS} tokens.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts generated per batch (default 8)",
    )
    ap.add_argument(
        "--tokens-file",
        default=DEFAULT_TOKENS_FILE,
        help="Reference list of ftfy-affected token ids (for the glyph inventory). "
        "Defaults to Luciole-ftfy-tokens.txt next to this script.",
    )
    ap.add_argument(
        "--from-json",
        nargs="+",
        default=None,
        help="Skip the models; (re)build the .md report from one or more existing "
        ".json files. Several files are merged into a single report, one column "
        "per model, in the order given.",
    )
    args = ap.parse_args()

    # --- regenerate / merge markdown only ---
    if args.from_json:
        payloads = []
        for p in args.from_json:
            with open(p) as f:
                payloads.append(json.load(f))
        payload = merge_payloads(payloads)
        json_path = args.output_prefix + ".json"
        with open(json_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        md = build_markdown(payload)
        md_path = args.output_prefix + ".md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"Merged {len(payloads)} file(s) -> {json_path} and {md_path}")
        return

    if not args.models:
        ap.error("provide at least one model path (or --from-json)")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device; running on CPU (slow).", flush=True)

    affected_ids = []
    if args.tokens_file and os.path.exists(args.tokens_file):
        affected_ids = load_affected_ids(args.tokens_file)
        print(f"Loaded {len(affected_ids)} affected token ids from {args.tokens_file}")
    else:
        print(f"No tokens file at {args.tokens_file}; skipping glyph inventory.")

    # Resolve (label, path) for each model. Explicit LABEL=PATH wins; otherwise
    # fall back to the path basename. Disambiguate any collisions.
    paths, labels, seen = [], [], {}
    for arg in args.models:
        label, path = split_label(arg)
        base = (
            label if label is not None else (os.path.basename(path.rstrip("/")) or path)
        )
        if base in seen:
            seen[base] += 1
            base = f"{base}#{seen[base]}"
        else:
            seen[base] = 0
        paths.append(path)
        labels.append(base)

    close_probes = build_prob_close_probes()
    n_probcheck_items = sum(len(v["prompts"]) for v in close_probes.values()) + len(
        APOSTROPHE_PREF_CONTEXTS
    )
    payload = {
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "tokens_file": args.tokens_file if affected_ids else None,
            "n_affected_ids": len(affected_ids),
            "generation_categories": list(GEN_PROBES.keys()),
            "probcheck_categories": list(GEN_PROBES.keys()),
            "n_generation_items": len(_flatten_gen_items()),
            "n_probcheck_items": n_probcheck_items,
            "generation_probes": {k: v["prompts"] for k, v in GEN_PROBES.items()},
            "probcheck_close_probes": {
                k: v["prompts"] for k, v in close_probes.items()
            },
            "apostrophe_pref_contexts": APOSTROPHE_PREF_CONTEXTS,
        },
        "model_order": labels,
        "models": {},
    }

    for path, label in zip(paths, labels):
        payload["models"][label] = analyze_one_model(
            path, label, device, args.max_new_tokens, args.batch_size, affected_ids
        )

    json_path = args.output_prefix + ".json"
    with open(json_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {json_path}")

    md = build_markdown(payload)
    md_path = args.output_prefix + ".md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
