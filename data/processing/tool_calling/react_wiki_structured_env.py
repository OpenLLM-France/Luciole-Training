"""Structured Wikipedia browser backing the `wiki_structured` tool set.

The section-aware counterpart to WikiEnv (react_wiki_env.py). Where WikiEnv's
`wiki_api` tools browse a page as a flat blob (first 5 sentences + Ctrl+F
`lookup`), WikiStructuredEnv parses each loaded page into its section tree. Its
reading tools are:

  search              ->  loads the page; returns its abstract (introduction)
  get_page_structure  ->  the table of contents (section outline)
  get_section         ->  the text of one named section (with its subsections)
  continue_reading    ->  the next page of the abstract / section being read

The abstract is not a separate tool: it is always shown by `search` (built from
the get_abstract helper, kept for that reuse but not exposed as a tool). Long
text is paged -- the abstract at MAX_ABSTRACT_CHARS per page, a section at
MAX_SECTION_CHARS -- and walked with continue_reading, which continues whichever
was read most recently. This lets the agent read a page's summary, see its
outline, and jump straight to the relevant section instead of scrolling with
`lookup`, which is far more token-efficient on long articles.

Page loading, title resolution, disambiguation, the offline ZIM / online HTTP
backends, and `submit_answer` F1 grading are all inherited from WikiEnv; this
module only adds the section parse and the three tools. The parse walks the
article in document order and works on both the offline mwoffliner ZIM markup
and live Wikipedia (both wrap section headings as
`<div class="mw-heading mw-heading{N}"><h{N} id="...">Title</h{N}></div>`; the
content root differs -- a flat `mw-parser-output` in the ZIM, Parsoid
`<section>` wrappers under `mw-body-content` live -- so `_pick_root` selects the
subtree that actually contains the headings and `_walk` recurses through the
wrapper divs/sections).
"""

import re

from react_wiki_env import WikiEnv, clean_str

# Heading tags and the block-level content tags whose text we collect under the
# current heading. Anything else (tables/infoboxes, figures, navboxes) is skipped
# -- the same "prose only" stance as WikiEnv's paragraph flatten.
_HEAD_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_CONTENT_TAGS = {"p", "ul", "ol", "dl", "blockquote"}

# Element classes that never hold article prose we want: page chrome,
# navigation, hatnotes, edit links, the rendered ToC, and reference lists.
# `_walk` does not recurse into an element carrying any of these, so their
# boilerplate is not swept into a section.
_SKIP_DIV_CLASSES = {
    "navbox", "vertical-navbox", "hatnote", "thumb", "mw-editsection", "toc",
    "reflist", "mw-references-wrap", "noprint", "navigation-not-searchable",
    "sistersitebox", "metadata",
}

# Tags `_walk` never recurses into: tables (infoboxes/navboxes), media, and
# non-prose leaves. Everything else is recursed into, so headings and paragraphs
# survive whatever wrapper the renderer nests them in -- Parsoid `<section>`s and
# the `<meta>` markers that Python's html.parser leaves around live-Wikipedia
# heading divs, as well as the flat `mw-parser-output` of the offline ZIM.
_SKIP_TAGS = {
    "table", "figure", "figcaption", "style", "script", "nav", "aside", "math",
    "form", "input", "img", "audio", "video",
}

# Inline citation markers like "[12]" -- Wikipedia superscript references that
# survive get_text(). Stripped so the model reads clean prose.
_CITATION_RE = re.compile(r"\[\d+\]")

# Minimum words for a block to count as prose (mirrors WikiEnv's >2-word filter,
# which drops stray one/two-word list items and fragments).
_MIN_BLOCK_WORDS = 2

# The lead pseudo-section's marker: level 0 so it sorts before every real
# heading (h1-h6 -> levels 1-6) and is easy to identify.
_LEAD_LEVEL = 0
_LEAD_TITLE = "(introduction)"

# Characters of a section shown per get_section / continue_reading call. A long
# section is split into pages of at most this many characters (broken on a word
# boundary); the agent walks the pages with continue_reading. Raise for fewer,
# longer pages.
MAX_SECTION_CHARS = 2500

# Characters of the abstract shown by search. The lead is usually short; if it is
# longer than this it is truncated (the sections hold the detail anyway).
MAX_ABSTRACT_CHARS = 1500


def _heading_info(el):
    """Return (level, title) if `el` is a section heading, else None.

    Handles both markups seen in the wild: a bare `<h2>` and the modern
    `<div class="mw-heading mw-heading2"><h2 id="...">` wrapper. `mw-editsection`
    edit links and an optional legacy `mw-headline` span are handled so the
    title is clean text.
    """
    tag = None
    if el.name in _HEAD_TAGS:
        tag = el
    elif el.name == "div" and el.get("class") and "mw-heading" in el.get("class"):
        tag = el.find(list(_HEAD_TAGS))
    if tag is None:
        return None
    for edit in tag.find_all(class_="mw-editsection"):
        edit.extract()
    headline = tag.find(class_="mw-headline")
    title = " ".join((headline or tag).get_text().split())
    return int(tag.name[1]), title


def _clean_block(text):
    """Strip inline citation markers and collapse whitespace in a block's text."""
    return " ".join(_CITATION_RE.sub("", text).split())


def _walk(node, emit_heading, emit_content):
    """Walk `node` in document order, emitting headings and prose blocks.

    Content tags (`_CONTENT_TAGS`) are captured as leaf text (not descended
    into, so a nested list inside a paragraph is not double-counted). Every other
    element is recursed into -- so headings and paragraphs are found whatever
    wrapper the renderer nests them in -- except non-prose tags (`_SKIP_TAGS`,
    e.g. infobox tables) and chrome carrying a `_SKIP_DIV_CLASSES` class.
    """
    for child in node.children:
        name = getattr(child, "name", None)
        if name is None:  # NavigableString
            continue
        head = _heading_info(child)
        if head is not None:
            emit_heading(*head)
            continue
        if name in _CONTENT_TAGS:
            text = _clean_block(child.get_text())
            if len(text.split()) > _MIN_BLOCK_WORDS:
                emit_content(text)
            continue
        if name in _HEAD_TAGS:  # a bare heading was already handled by _heading_info
            continue
        if name in _SKIP_TAGS:
            continue
        if set(child.get("class") or []) & _SKIP_DIV_CLASSES:
            continue
        _walk(child, emit_heading, emit_content)


def _pick_root(soup):
    """Return the subtree that actually holds the article's section headings.

    The offline ZIM populates the first `mw-parser-output`; live Wikipedia
    leaves that one nearly empty and puts the real content in a second
    `mw-parser-output` under `mw-body-content`. Pick the candidate containing the
    most `<div class="mw-heading">` wrappers, falling back to the whole document.
    """
    candidates = soup.find_all(class_="mw-body-content") + soup.find_all(class_="mw-parser-output")
    best, best_n = None, -1
    for cand in candidates:
        n = len(cand.find_all("div", class_="mw-heading"))
        if n > best_n:
            best, best_n = cand, n
    return best or soup


def parse_sections(soup):
    """Parse a page's soup into an ordered list of [level, title, blocks].

    The first element is always the lead/introduction (level `_LEAD_LEVEL`); each
    subsequent element is a section heading with its own prose blocks (subsection
    text belongs to the subsection's own element, reconstructed on demand by
    `WikiStructuredEnv._find_section`). Block text is cleaned with `clean_str`
    (the same unicode fix WikiEnv applies) so it matches the rest of the pipeline.
    """
    sections = [[_LEAD_LEVEL, _LEAD_TITLE, []]]

    def emit_heading(level, title):
        sections.append([level, title, []])

    def emit_content(text):
        sections[-1][2].append(clean_str(text))

    _walk(_pick_root(soup), emit_heading, emit_content)
    return sections


class WikiStructuredEnv(WikiEnv):
    """Section-aware Wikipedia browser (see module docstring).

    Reuses every WikiEnv mechanism (backends, disambiguation, grading) and only
    overrides `_render_article` to parse the loaded page into `self.sections`,
    adding the three navigation tools. `max_section_chars` / `max_abstract_chars`
    cap a single tool's output so a long section cannot flood the episode
    context; `landing_abstract_chars` bounds the lead shown when a page loads.
    """

    def __init__(self, ground_truth=None, backend="online", zim_path=None,
                 accept_threshold=None, auto_disambiguation=False,
                 max_section_chars=MAX_SECTION_CHARS,
                 max_abstract_chars=MAX_ABSTRACT_CHARS, scorer=None):
        super().__init__(ground_truth=ground_truth, backend=backend,
                         zim_path=zim_path, accept_threshold=accept_threshold,
                         auto_disambiguation=auto_disambiguation, scorer=scorer)
        self.sections = None  # parsed [level, title, blocks] list for the loaded page
        self.max_section_chars = max_section_chars
        self.max_abstract_chars = max_abstract_chars
        self._reset_reader()

    def _reset_reader(self):
        """Clear the paginated-reader state (label + pages + next page index).

        The reader backs continue_reading: it holds whatever text was last shown
        one page at a time -- the introduction (from search) or a section (from
        get_section) -- so continue_reading can serve the next page of it.
        """
        self._reader_label = None
        self._reader_pages = None
        self._reader_page_idx = 0

    def reset(self):
        super().reset()
        self.sections = None
        self._reset_reader()

    # ------------------------------------------------------------------
    # Page loading: parse the article into sections instead of a flat blob.
    # ------------------------------------------------------------------
    def _render_article(self, soup, entity):
        self.sections = parse_sections(soup)
        # Keep a flat `self.page` too (join of every block) so inherited
        # machinery that inspects it still works; the structured tools read
        # `self.sections`, not this.
        self.page = "\n".join(
            block for _, _, blocks in self.sections for block in blocks
        )
        if not self.page.strip():
            self.sections = None
            self.obs = (
                f"The page '{entity}' has no readable text. Try a more "
                f"specific title or a different entity."
            )
            return self.obs
        self.obs = self._landing_obs(entity)
        return self.obs

    def _landing_obs(self, entity):
        """The observation returned by `search` once a page is loaded: the page's
        introduction (abstract). A long introduction is paged like a section --
        this shows the first page and continue_reading walks the rest. The tools'
        own descriptions cover usage, so no hint is appended here.
        """
        text = self._abstract_text()
        if not text:
            self._reset_reader()
            return f"Loaded page '{entity}'.\n(This page has no introduction text.)"
        first = self._start_reader("Introduction", text, self.max_abstract_chars)
        return f"Loaded page '{entity}'.\n{first}"

    # ------------------------------------------------------------------
    # Helpers over self.sections
    # ------------------------------------------------------------------
    def _no_page(self):
        return "Error: no page loaded. Call `search` first to load a Wikipedia page."

    def _abstract_text(self):
        return "\n".join(self.sections[0][2]) if self.sections else ""

    def _toc_lines(self):
        """Indented outline of the real (h2+) headings, one per line."""
        lines = []
        for level, title, _ in self.sections:
            if level >= 2:
                lines.append("  " * (level - 2) + f"- {title}")
        return lines

    def _find_section(self, name):
        """Return (body_text, matched_title, subtitles) for `name`, or (None, None, []).

        Matches a heading title case-insensitively (exact first, then a
        substring match so a partial title still resolves). The body includes the
        section's own prose followed by every subsection until the next heading of
        the same or higher level, subsections marked with `== Title ==`.
        `subtitles` lists the immediate (direct-child) subsection titles, so the
        caller can tell an empty leaf (content was a skipped table/list) apart
        from a parent that only holds subsections.
        """
        target = name.strip().lower()
        idx = None
        for i, (level, title, _) in enumerate(self.sections):
            if level >= 2 and title.lower() == target:
                idx = i
                break
        if idx is None:
            for i, (level, title, _) in enumerate(self.sections):
                if level >= 2 and target in title.lower():
                    idx = i
                    break
        if idx is None:
            return None, None, []
        level, title, blocks = self.sections[idx]
        parts = list(blocks)
        subtitles = []
        for sub_level, sub_title, sub_blocks in self.sections[idx + 1:]:
            if sub_level <= level:
                break
            if sub_level == level + 1:
                subtitles.append(sub_title)
            parts.append(f"== {sub_title} ==")
            parts.extend(sub_blocks)
        return "\n".join(parts).strip(), title, subtitles

    # ------------------------------------------------------------------
    # Page views. search, get_page_structure, get_section and continue_reading
    # are tools (see react_tools); get_abstract is NOT a tool -- it is the
    # helper that supplies the abstract shown (paginated) by search.
    # ------------------------------------------------------------------
    def get_page_structure(self):
        """get_page_structure[]: the table of contents (section outline) of the loaded page."""
        if not self.sections:
            return self._no_page()
        lines = self._toc_lines()
        if not lines:
            return "Table of contents: (this page has no section headings)."
        return "Table of contents:\n" + "\n".join(lines)

    def get_abstract(self):
        """The lead paragraphs (introduction) of the loaded page, in full.
        Internal helper (not a tool): `search` shows this text, paginated."""
        if not self.sections:
            return self._no_page()
        return self._abstract_text() or "(This page has no introduction text.)"

    def get_section(self, section):
        """get_section[section]: the text of the named section (with its subsections).

        A section longer than `max_section_chars` is split into pages; this
        returns the first page and `continue_reading` walks the rest.
        """
        if not self.sections:
            return self._no_page()
        if not isinstance(section, str) or not section.strip():
            return ("Error: 'section' must be a non-empty section title. Call "
                    "get_page_structure to see the available section titles.")
        body, title, subtitles = self._find_section(section)
        if body is None:
            titles = [t for level, t, _ in self.sections if level >= 2]
            return (f"Error: no section titled '{section}'. Available sections: "
                    f"{titles}. Use an exact title from get_page_structure.")
        if not body:
            self._reset_reader()
            if subtitles:
                return (f"Section '{title}' has no text of its own; it only "
                        f"contains subsections: {subtitles}. Request one of them "
                        "by title.")
            return (f"Section '{title}' has no readable prose text -- its content "
                    "is a table, list, or media that this reader does not "
                    "capture. Use get_page_structure to choose another section.")
        return self._start_reader(f"Section '{title}'", body, self.max_section_chars)

    def continue_reading(self):
        """continue_reading[]: the next page of what you are reading (the
        introduction from search, or a section from get_section)."""
        if self._reader_pages is None:
            return ("Error: nothing to continue. Load a page with search or open a "
                    "section with get_section first.")
        if self._reader_page_idx >= len(self._reader_pages):
            return f"No more text in {self._reader_label}."
        return self._serve_next_page()

    def _start_reader(self, label, text, limit):
        """Begin a paginated read of `text` under `label`; return its first page.

        `label` names the text in the page marker and the continue_reading
        messages (e.g. "Introduction" or "Section 'History'").
        """
        self._reader_label = label
        self._reader_pages = self._paginate(text, limit)
        self._reader_page_idx = 0
        return self._serve_next_page()

    @staticmethod
    def _paginate(text, limit):
        """Split `text` into pages of at most `limit` chars, broken on whitespace.

        Each break is taken at the last whitespace at or before the limit (so
        words are never cut mid-token); a run with no whitespace is hard-cut at
        the limit as a fallback.
        """
        if len(text) <= limit:
            return [text]
        pages, i, n = [], 0, len(text)
        while i < n:
            if n - i <= limit:
                pages.append(text[i:].strip())
                break
            cut = text.rfind(" ", i, i + limit)
            nl = text.rfind("\n", i, i + limit)
            cut = max(cut, nl)
            if cut <= i:  # no whitespace in the window -> hard cut
                cut = i + limit
            pages.append(text[i:cut].strip())
            i = cut
        return [p for p in pages if p]

    def _serve_next_page(self):
        """Return the current reader page, advancing the page cursor.

        A single-page read is returned as-is; a multi-page one is prefixed with a
        `(<label>, page i of N)` marker and, unless it is the last page, notes
        that continue_reading yields the rest.
        """
        pages = self._reader_pages
        i = self._reader_page_idx
        total = len(pages)
        self._reader_page_idx += 1
        if total == 1:
            return pages[i]
        text = f"({self._reader_label}, page {i + 1} of {total})\n{pages[i]}"
        if i + 1 < total:
            text += "\n[Call continue_reading for the next page.]"
        return text


def new_structured_env(doc, backend="online", zim_path=None, accept_threshold=None,
                       auto_disambiguation=False, scorer=None):
    """Create a fresh WikiStructuredEnv for one conversation and alias its scores.

    Mirrors react.new_env (the wiki_api factory): the ground-truth answer is
    taken from the document so `submit_answer` can grade the agent, and the env's
    score list is aliased into the doc metadata so each submit_answer score lands
    in the output (same list object -> appends persist). `scorer` overrides the
    default token-F1 grading (e.g. FEVER exact match).
    """
    env = WikiStructuredEnv(
        ground_truth=doc.metadata["answer"], backend=backend, zim_path=zim_path,
        accept_threshold=accept_threshold, auto_disambiguation=auto_disambiguation,
        scorer=scorer,
    )
    doc.metadata["submit_answer_scores"] = env.scores
    return env
