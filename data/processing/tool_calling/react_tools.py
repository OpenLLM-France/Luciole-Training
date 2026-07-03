"""Tool schemas for the ReAct/HotpotQA agent, selected by react_hotpot's --tool_type.

Pure data: the OpenAI-format function schemas sent to vLLM, plus the TOOL_SETS
registry mapping each --tool_type to its tool list. No environment logic and no
heavy dependencies live here -- the backends that execute these tools are:

  wiki_api   `search` + `lookup`                 -> WikiEnv      (react_wiki_env.py)
  retriever  `wikipedia_retriever` + `next_results` -> RetrieverEnv (react_retriever_env.py)

Both tool sets share the same `submit_answer` finish tool and the same F1
grading (GradedEnv, react_wiki_env.py); only the retrieval half differs. react_hotpot
imports TOOL_SETS to expose the right schemas and picks the matching env factory.
"""

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


# Registry: --tool_type selects the tool schema list exposed to the agent. The
# matching env factory is chosen in react_hotpot (its args differ per tool set).
TOOL_SETS = {
    "wiki_api": WIKI_API_TOOLS,
    "retriever": RETRIEVER_TOOLS,
}
