"""Tool schemas for the ReAct/HotpotQA agent, selected by react_hotpot's --tool_type.

Pure data: the OpenAI-format function schemas sent to vLLM, plus the TOOL_SETS
registry mapping each --tool_type to its tool list. No environment logic and no
heavy dependencies live here -- the backends that execute these tools are:

  wiki_api   `search` + `lookup`                 -> WikiEnv      (react_wiki_env.py)
  retriever  `wikipedia_retriever` + `next_results` -> RetrieverEnv (react_retriever_env.py)

Both tool sets share the same `submit_answer` finish tool and the same F1
grading (GradedEnv, react_wiki_env.py); only the retrieval half differs. react_hotpot
imports TOOL_SETS to expose the right schemas and picks the matching env factory.

The `search` tools' descriptions depend on the run's `--auto_disambiguation`
mode (manual: lists the ambiguous title's options; auto: opens the top one), so
`build_tool_sets(auto_disambiguation)` resolves that clause for the run. TOOL_SETS
itself carries the base descriptions (no disambiguation clause) and is the source
of the tool *names* used elsewhere (allowed_tool_names, --tool_types choices).
"""

import copy

# Shared finish tool -- identical for both tool sets, so defined once.
SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Finish the current task and submit the final answer to the question. The answer must be short and exact (a name, date, or short phrase -- not a sentence).",
        "parameters": {
            "type": "object",
            "properties": {
                "detailed_answer": {
                    "type": "string",
                    "description": "A complete, self-contained answer written as one or more full sentences, explaining the answer in context.",
                },
                "supporting_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The list of evidence sentences retrieved with the tools that support the answer. If there is no supporting fact, it should be an empty list.",
                },
                "short_answer": {
                    "type": "string",
                    "description": "The final answer to the question.",
                },
            },
            "required": ["detailed_answer", "supporting_facts", "short_answer"],
        },
    },
}


# FEVER is a 3-way fact-check, not open-ended QA: the answer is a verdict, not a
# name/date/phrase. Reuse submit_answer's structure (detailed_answer,
# supporting_facts) but redefine the task and the short_answer field, which is
# constrained to the three FEVER labels (matching doc.metadata["answer"], set to
# the lowercased FEVER label in react's fever reader).
FEVER_SUBMIT_ANSWER_TOOL = copy.deepcopy(SUBMIT_ANSWER_TOOL)
FEVER_SUBMIT_ANSWER_TOOL["function"]["description"] = (
    "Finish the fact-checking task and submit your verdict on whether the "
    "evidence you found in Wikipedia supports the claim."
)
FEVER_SUBMIT_ANSWER_TOOL["function"]["parameters"]["properties"]["short_answer"] = {
    "type": "string",
    "description": (
        "The verdict: 'supports' if Wikipedia's evidence supports the claim, "
        "'refutes' if it contradicts the claim, or 'not enough info' if the "
        "evidence is insufficient to decide."
    ),
    "enum": ["supports", "refutes", "not enough info"],
}

# Datasets whose submit_answer differs from the default open-ended QA one. Any
# dataset absent here uses SUBMIT_ANSWER_TOOL. Resolved per run by build_tool_sets.
SUBMIT_ANSWER_TOOLS = {"fever": FEVER_SUBMIT_ANSWER_TOOL}


# ---------------------------------------------------------------------------
# wiki_api tool set: browse Wikipedia by exact title (search) then Ctrl+F within
# the loaded page (lookup). Backed by WikiEnv (react_wiki_env.py).
# ---------------------------------------------------------------------------
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search Wikipedia for an entity. Returns the first 5 sentences from the corresponding entity wiki page if it exists, otherwise returns the top-5 similar entities suggested by the Wikipedia search engine.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "The exact name (page title) of the Wikipedia entity to search for.",
                }
            },
            "required": ["entity"],
        },
    },
}

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Return the next passage in the current page containing the given string, simulating the Ctrl+F functionality of a browser. Call it again with the same string to jump to the next occurrence (results are labelled '(Result i / N)'). Must be called after a search has loaded a page.",
        "parameters": {
            "type": "object",
            "properties": {
                "string": {
                    "type": "string",
                    "description": "The string to look up within the currently loaded Wikipedia page.",
                }
            },
            "required": ["string"],
        },
    },
}

WIKI_API_TOOLS = [SEARCH_TOOL, LOOKUP_TOOL, SUBMIT_ANSWER_TOOL]


# ---------------------------------------------------------------------------
# retriever tool set: semantic `wikipedia_retriever` over a pre-built embedding
# index, plus `next_results` to page further down the same ranked list (the
# retriever's analog of wiki_api's `lookup`). Backed by RetrieverEnv
# (react_retriever_env.py).
# ---------------------------------------------------------------------------
WIKIPEDIA_RETRIEVER_TOOL = {
    "type": "function",
    "function": {
        "name": "wikipedia_retriever",
        "description": (
            "Semantic search over Wikipedia: returns the passages most relevant "
            "to your query, ranked by meaning (dense text embeddings), not by "
            "keyword. The query can be a natural-language question or a "
            "description of the fact you need -- you do not need an exact page "
            "title. Returns the top matching passages, each with its article "
            "title. Starts a new ranked list; use `next_results` to see further "
            "results for the same query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language question or description of the information you are looking for.",
                },
                "k": {
                    "type": "integer",
                    "description": "Optional number of passages to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        },
    },
}

NEXT_RESULTS_TOOL = {
    "type": "function",
    "function": {
        "name": "next_results",
        "description": (
            "Return the next passages for your most recent `wikipedia_retriever` "
            "query, continuing further down the ranked list (like scrolling past "
            "the first page of results). Must be called after `wikipedia_retriever`. "
            "Use it when the top results did not contain the answer, before trying "
            "a different query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "k": {
                    "type": "integer",
                    "description": "Optional number of additional passages to return (default 5, max 10).",
                },
            },
            "required": [],
        },
    },
}

RETRIEVER_TOOLS = [WIKIPEDIA_RETRIEVER_TOOL, NEXT_RESULTS_TOOL, SUBMIT_ANSWER_TOOL]


# ---------------------------------------------------------------------------
# wiki_structured tool set: browse Wikipedia by section. `search` resolves a
# title and loads the page, returning its introduction (abstract);
# `get_page_structure` lists the page's section outline, and `get_section` reads
# one section. (The abstract is not a separate tool: it is shown by `search`.)
# Backed by WikiStructuredEnv (react_wiki_structured_env.py).
# ---------------------------------------------------------------------------
# search here returns the page's abstract (WikiStructuredEnv._landing_obs),
# unlike wiki_api's search (first 5 sentences), so it keeps the same `search`
# function name but carries its own description.
STRUCTURED_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Load the Wikipedia page for an entity and return its introduction "
            "(abstract). If no such page exists, returns similar entity titles."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Name (page title) of the Wikipedia entity to search for.",
                }
            },
            "required": ["entity"],
        },
    },
}

GET_PAGE_STRUCTURE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_page_structure",
        "description": (
            "Return the table of contents (section and subsection titles) of the "
            "currently loaded page."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

GET_SECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_section",
        "description": (
            "Return the text of a section of the currently loaded page. Long "
            "sections are returned one page at a time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Exact title of the section to read.",
                }
            },
            "required": ["section"],
        },
    },
}

CONTINUE_READING_TOOL = {
    "type": "function",
    "function": {
        "name": "continue_reading",
        "description": (
            "Return the next page of the introduction or section currently being "
            "read one page at a time."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

WIKI_STRUCTURED_TOOLS = [
    STRUCTURED_SEARCH_TOOL,
    GET_PAGE_STRUCTURE_TOOL,
    GET_SECTION_TOOL,
    CONTINUE_READING_TOOL,
    SUBMIT_ANSWER_TOOL,
]


# Registry: --tool_type selects the tool schema list exposed to the agent. The
# matching env factory is chosen in react_hotpot (its args differ per tool set).
# These carry the BASE search descriptions (no disambiguation clause); use
# build_tool_sets() to get the run's flag-aware version. Names/keys here are
# authoritative (allowed_tool_names, --tool_types choices).
TOOL_SETS = {
    "wiki_api": WIKI_API_TOOLS,
    "wiki_structured": WIKI_STRUCTURED_TOOLS,
    "retriever": RETRIEVER_TOOLS,
}


# Disambiguation clause appended to a `search` tool's description, per run mode.
# The behaviour is WikiEnv._render_page's, shared by wiki_api and wiki_structured.
_DISAMBIG_CLAUSE_MANUAL = (
    " If the title is ambiguous, lists the pages it may refer to."
)
_DISAMBIG_CLAUSE_AUTO = (
    " If the title is ambiguous, opens the most likely page and names the alternatives."
)


def build_tool_sets(auto_disambiguation=False, dataset_name=None):
    """Return the tool sets with the run's `search` and `submit_answer` resolved.

    Two things depend on the run and so are resolved here rather than baked into
    the module-level schemas:
    - the `search` description depends on `auto_disambiguation` (manual lists the
      ambiguous title's options; auto opens the top one). Only wiki_api and
      wiki_structured have a `search`; the retriever set is left untouched.
    - the `submit_answer` description depends on `dataset_name`: FEVER submits a
      3-way verdict, the QA datasets submit a short exact answer. Every tool set
      ends in the shared SUBMIT_ANSWER_TOOL, swapped here for the dataset's.
    """
    clause = _DISAMBIG_CLAUSE_AUTO if auto_disambiguation else _DISAMBIG_CLAUSE_MANUAL
    submit_tool = SUBMIT_ANSWER_TOOLS.get(dataset_name, SUBMIT_ANSWER_TOOL)

    def resolved(tools, has_search):
        tools = list(tools)
        # Swap the shared submit_answer (always the final tool) for the dataset's.
        if tools and tools[-1] is SUBMIT_ANSWER_TOOL:
            tools = tools[:-1] + [submit_tool]
        if has_search:
            # The search tool leads both wiki sets; copy it so the appended
            # clause does not mutate the shared module-level schema.
            head = copy.deepcopy(tools[0])
            head["function"]["description"] += clause
            tools = [head] + tools[1:]
        return tools

    return {
        "wiki_api": resolved(WIKI_API_TOOLS, has_search=True),
        "wiki_structured": resolved(WIKI_STRUCTURED_TOOLS, has_search=True),
        "retriever": resolved(RETRIEVER_TOOLS, has_search=False),
    }
