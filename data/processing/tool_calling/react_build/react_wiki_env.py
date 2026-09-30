# Wikipedia web API tools, ported from the ReAct repo (wikienv.py).
# https://github.com/ysymyth/ReAct
#
# The three retrieval actions from the paper plus the auxiliary `thoughts`
# action. `lookup` depends on the page loaded by the last `search`, so the
# state is held on a WikiEnv instance and the tool functions operate on it.

import os
import random
import re
import string
import threading
import time
from collections import Counter

import requests
from bs4 import BeautifulSoup
from loguru import logger

# Wikipedia returns HTTP 403 to requests without a descriptive User-Agent.
# https://w.wiki/4wJS  https://phabricator.wikimedia.org/T400119
_HEADERS = {
    "User-Agent": "Luciole_tool_calling (research dataset generation) "
    "contact ogouvert@linagora.com)"
}
_TIMEOUT = 15  # seconds; fail fast instead of hanging on a stuck socket

# Live Wikipedia rate-limits scrapers: with many concurrent agent episodes
# each firing a search, we get HTTP 429 ("too many requests"). Two defences,
# applied process-wide (the runner executes tools via asyncio.to_thread, so
# all WikiEnv instances share these):
#   1. a semaphore capping how many requests hit Wikipedia at once;
#   2. retry with exponential backoff + jitter on 429/503, honouring the
#      server's Retry-After header when present.
# For large-scale generation, prefer a local Wikipedia mirror over live hits.
_MAX_CONCURRENT_REQUESTS = 8
_request_semaphore = threading.Semaphore(_MAX_CONCURRENT_REQUESTS)
_RETRY_STATUSES = (429, 503)
_MAX_RETRIES = 6
_MAX_BACKOFF = 30  # seconds

_session = requests.Session()
_session.headers.update(_HEADERS)


def _get(url):
    """GET `url` with concurrency capping and backoff on rate-limit responses.

    Returns the response text on success. Raises on a non-retryable HTTP error
    or after exhausting retries, so the caller (call_tool) surfaces it to the
    model as an error observation rather than parsing a 429 error page as if it
    were a Wikipedia article.
    """
    last_status = None
    for attempt in range(_MAX_RETRIES):
        with _request_semaphore:  # released before we sleep, so retries don't hold a slot
            resp = _session.get(url, timeout=_TIMEOUT)
        last_status = resp.status_code
        if resp.status_code == 200:
            return resp.text
        if resp.status_code not in _RETRY_STATUSES:
            resp.raise_for_status()
        # Rate-limited / temporarily unavailable: back off and retry.
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = min(2 ** attempt, _MAX_BACKOFF)
        time.sleep(delay + random.uniform(0, 1))  # jitter avoids thundering herd
    raise RuntimeError(
        f"Wikipedia request failed after {_MAX_RETRIES} retries (last status {last_status})"
    )


# Offline backend: a local Kiwix ZIM snapshot read in-process via libzim. No
# network, no rate limits, safe for high concurrency. The Archive is opened
# once per process and shared across all WikiEnv instances (opening it per
# episode would be wasteful). libzim is imported lazily so the online backend
# has no dependency on it.
_archive = None
_archive_path = None
_archive_lock = threading.Lock()


def _open_archive(zim_path):
    """Open (once per process) and return the ZIM Archive at `zim_path`.

    Falls back to the WIKI_ZIM_PATH env var when no path is passed, so the
    Slurm worker (which sources the env) is authoritative even if the path was
    not resolvable on the submit node.
    """
    global _archive, _archive_path
    if _archive is None:
        with _archive_lock:
            if _archive is None:
                zim_path = zim_path or os.environ.get("WIKI_ZIM_PATH")
                if not zim_path:
                    raise ValueError(
                        "offline backend requires a .zim path (pass zim_path or "
                        "set WIKI_ZIM_PATH)"
                    )
                from libzim.reader import Archive  # lazy: only needed offline

                _archive = Archive(zim_path)
                _archive_path = zim_path
    return _archive


# How often (in tool calls) to emit a wiki_api profiling line; 0 disables it.
WIKI_LOG_EVERY = int(os.environ.get("REACT_WIKI_LOG_EVERY", "50"))
# Average search latency above which the backend is flagged [SLOW]; a ZIM page
# fetch+parse should be tens of ms, so hundreds of ms signals a real bottleneck.
WIKI_SLOW_MS = float(os.environ.get("REACT_WIKI_SLOW_MS", "500"))


class _WikiStats:
    """Process-wide latency/concurrency profiler for the wiki_api backend.

    Counterpart to react_retriever_env._EmbedStats, but wiki_api's suspect is
    I/O, not CPU: each `search` fetches and parses a page from the ZIM archive
    (or Wikipedia over HTTP); `lookup` is in-memory Ctrl+F and should be nearly
    free. Every WIKI_LOG_EVERY tool calls this logs -- into datatrove's own
    loguru stream, beside the inference metrics -- average search/lookup latency,
    throughput, and current/peak concurrency, tagging [SLOW backend] if a page
    fetch averages over WIKI_SLOW_MS. That is the signature of a strange
    bottleneck in the backend rather than in vLLM generation.
    """

    def __init__(self, report_every=WIKI_LOG_EVERY):
        self.lock = threading.Lock()
        self.report_every = report_every
        self.n = 0
        self.counts = {"search": 0, "lookup": 0}
        self.times = {"search": 0.0, "lookup": 0.0}
        self.inflight = 0
        self.peak = 0
        self.first_t = None

    def begin(self):
        """Mark a tool call as started (bumps the in-flight gauge)."""
        if not self.report_every:
            return
        with self.lock:
            if self.first_t is None:
                self.first_t = time.perf_counter()
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)

    def end(self, op, seconds):
        """Mark a `search`/`lookup` call as finished; log every report_every calls."""
        if not self.report_every:
            return
        with self.lock:
            self.inflight -= 1
            self.n += 1
            self.counts[op] += 1
            self.times[op] += seconds
            if self.n % self.report_every:
                return
            counts, times = dict(self.counts), dict(self.times)
            n, inflight, peak = self.n, self.inflight, self.peak
            wall = time.perf_counter() - self.first_t

        def avg_ms(o):
            return 1000 * times[o] / counts[o] if counts[o] else 0.0

        tag = "SLOW backend" if avg_ms("search") >= WIKI_SLOW_MS else "ok"
        logger.info(
            "wiki_api [{}]: {} calls | search {:.0f} ms x{}, lookup {:.0f} ms x{} "
            "| {:.1f} calls/s | in-flight {} (peak {}) | wall {:.0f}s",
            tag, n, avg_ms("search"), counts["search"], avg_ms("lookup"),
            counts["lookup"], n / wall if wall else 0.0, inflight, peak, wall,
        )


_STATS = _WikiStats()


def clean_str(p):
    # Undo the double-escaping in Wikipedia's raw HTML (e.g. "\\u00e9" -> "é").
    # This byte round-trip is fragile: some pages contain backslash sequences
    # that are not valid unicode/latin1 escapes and raise UnicodeDecodeError.
    # A crash here would abort the whole page load (turned into an error
    # observation by call_tool), so fall back to the original text on failure --
    # leaving it un-unescaped is far better than losing the page.
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return p


# A Wikipedia disambiguation page leads with "<title> may refer to:" (or
# "commonly refers to:"). We detect that and return the listed options for the
# model to choose from, rather than guessing a page on its behalf.
_DISAMBIG_RE = re.compile(r"refers? to:")
_MAX_OPTIONS = 15

# In automatic-disambiguation mode (auto_disambiguation=True) a disambiguation
# page is resolved by opening its top option instead of asking the model to pick.
# The chosen option can itself be a disambiguation page, so cap the chain to
# avoid an unbounded loop; past the cap we fall back to listing the options.
MAX_AUTO_DISAMBIG_HOPS = 2

# Context window for a single `lookup` hit: this many sentences on EACH side of
# the matched sentence are included (so LOOKUP_CONTEXT=1 -> previous + match +
# next). Raise for more surrounding context per lookup.
LOOKUP_CONTEXT = 1


def normalize_answer(s):
    """Normalize a QA answer for exact-match scoring.

    Standard HotpotQA/SQuAD normalization: lowercase, drop punctuation, drop
    the articles a/an/the, and collapse whitespace. This is what "exact match"
    means for these benchmarks (raw string equality is far too brittle:
    "The Beatles." vs "beatles" should match).
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    """Token-level F1 between two answers (HotpotQA/SQuAD metric), in [0, 1].

    Both are normalized first. Exact match scores 1.0; a partial token overlap
    scores between 0 and 1; no shared token scores 0.0. Graded (unlike exact
    match) so the caller can react to a near-miss differently from a total miss.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        # e.g. yes/no answers reduced to empty: score 1.0 only if both empty.
        return float(pred_tokens == gt_tokens)
    num_same = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def fever_score(prediction, ground_truth):
    """Exact-match scoring for FEVER verdicts: 1.0 if they match, else 0.0.

    A FEVER verdict is one of a fixed label set, so grading is strict exact match
    after normalization -- not token-F1. Every label (including 'not enough
    info') is graded the same way.
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


# dataset_name -> answer scorer for submit_answer. Datasets absent here use the
# default token-F1 (f1_score). Resolved per run by react's env factory.
SCORERS = {"fever": fever_score}


class GradedEnv:
    """Answer submission + grading shared by every tool-set environment.

    Every tool set (Wikipedia browse, dense retrieval, ...) exposes the same
    `submit_answer` finish tool and grades the agent's final answer against the
    question's ground truth with token-level F1. That state and logic live here
    so each concrete env (WikiEnv, RetrieverEnv, ...) only has to implement its
    own retrieval tools. Subclasses must set `self.obs` (the current
    observation) and call `super().__init__(...)`.

    `ground_truth` is the reference answer for the current question;
    `accept_threshold` is the F1 an answer must strictly exceed to be accepted
    (negative = accept any answer on the first submit).
    """

    ACCEPT_THRESHOLD = 0.0  # F1 the answer must strictly exceed to be accepted (episode ends)

    def __init__(self, ground_truth=None, accept_threshold=None, scorer=None):
        self.obs = None  # current observation
        self.answer = None  # current answer from the agent
        self.detailed_answer = None  # full-sentence answer from the agent
        self.supporting_facts = None  # evidence sentences backing the answer
        self.ground_truth = ground_truth  # reference answer for grading
        # How submit_answer grades short_answer against ground_truth. Defaults to
        # token-F1 (QA); a dataset may inject its own (e.g. FEVER exact match).
        # A scorer may return None to mean "do not score" (episode recorded but
        # left ungraded), handled alongside the no-ground-truth case.
        self.scorer = scorer if scorer is not None else f1_score
        self.score = None  # score of the last submitted answer (None = unscored)
        self.scores = []  # score of every submit_answer call, in order
        self.episode_done = False  # set by submit_answer; ends the agent loop
        # F1 the answer must strictly exceed to be accepted (negative = accept
        # any answer); falls back to the class default.
        self.accept_threshold = self.ACCEPT_THRESHOLD if accept_threshold is None else accept_threshold

    def reset_answer(self):
        """Clear the answer/score state (shared by every subclass' reset())."""
        self.answer = None
        self.detailed_answer = None
        self.supporting_facts = None
        self.score = None
        self.scores = []
        self.episode_done = False

    def submit_answer(self, short_answer, detailed_answer=None, supporting_facts=None):
        """submit_answer[short_answer, detailed_answer, supporting_facts]: finish the task and grade `short_answer`.

        `short_answer` is the short, exact answer that is graded. `detailed_answer`
        (a full-sentence answer) and `supporting_facts` (the evidence sentences
        that back it) are recorded on the env but do not affect grading.

        Grades the answer against `self.ground_truth` with token-level F1 (see
        `f1_score`), stored on the env as `self.score`. The observation and
        whether the episode ends depend on the score:
          - threshold < 0       : accepted unconditionally, episode ends;
          - score > threshold   : accepted, episode ends;
          - 0 < score <= threshold: a near miss -- the model is asked to
            reformulate its answer, and the loop stays open so it can;
          - == 0                : wrong -- the model is asked to gather more
            evidence, loop stays open;
          - no ground truth     : recorded, episode ends (nothing to grade).
        """
        # supporting_facts is a list whose entries are either plain strings
        # (wiki_api / wiki_structured evidence sentences) or {id, quote} objects
        # (the retriever variant). Reject anything else so the model gets a chance
        # to correct the call rather than silently recording a malformed value.
        def _ok_fact(f):
            if isinstance(f, str):
                return True
            return (isinstance(f, dict)
                    and isinstance(f.get("id"), str)
                    and isinstance(f.get("quote"), str))

        if supporting_facts is not None and not (
            isinstance(supporting_facts, list) and all(_ok_fact(f) for f in supporting_facts)
        ):
            return ("Error: 'supporting_facts' must be a list of strings or of "
                    "{\"id\": ..., \"quote\": ...} objects.")
        self.answer = short_answer
        self.detailed_answer = detailed_answer
        self.supporting_facts = supporting_facts
        # score is None when there is nothing to grade against, or when the
        # scorer declines to score this pair (e.g. FEVER 'not enough info').
        if self.ground_truth is None:
            self.score = None
        else:
            self.score = self.scorer(short_answer, self.ground_truth)
        if self.score is None:
            self.episode_done = True
            self.obs = "Final answer recorded."
        elif self.accept_threshold < 0 or self.score > self.accept_threshold:
            # A negative threshold accepts any answer (episode ends after the
            # first submit); otherwise the score must strictly exceed it.
            self.episode_done = True
            self.obs = f"Final answer recorded. Score: {self.score:.2f}."
        elif self.score > 0:
            self.episode_done = False
            self.obs = (
                f"Final answer recorded. "
                "This answer is close but not quite right; reformulate it "
                "(e.g. more concise) and "
                "submit again."
            )
        else:
            self.episode_done = False
            self.obs = (
                f"Final answer recorded. "
                "This answer seems to be wrong; re-examine the evidence with "
                "the tools and submit a different answer."
            )
        self.scores.append(self.score)
        return self.obs

    def thoughts(self, thought):
        """thoughts[thought]: record a reasoning step; does not change the env."""
        self.obs = "Nice thought."
        return self.obs


class WikiEnv(GradedEnv):
    """Stateful Wikipedia browser backing the search/lookup tools.

    `ground_truth` is the reference answer for the current question; it is used
    by `submit_answer` (inherited from GradedEnv) to grade the agent's final
    answer with exact match.

    `backend` selects where `search` gets its data:
      - "online": live Wikipedia over HTTP (rate-limited; see `_get`);
      - "offline": a local Kiwix ZIM snapshot at `zim_path` (read via libzim).
    """

    def __init__(self, ground_truth=None, backend="online", zim_path=None,
                 accept_threshold=None, auto_disambiguation=False, scorer=None):
        if backend not in ("online", "offline"):
            raise ValueError(f"backend must be 'online' or 'offline', got {backend!r}")
        super().__init__(ground_truth=ground_truth, accept_threshold=accept_threshold,
                         scorer=scorer)
        self.page = None  # current Wikipedia page
        self.lookup_keyword = None  # current lookup keyword
        self.lookup_list = None  # paragraphs containing current lookup keyword
        self.lookup_cnt = None  # current lookup index
        self.backend = backend
        self.zim_path = zim_path
        # When True, a disambiguation page is auto-resolved to its top option
        # instead of returning the option list for the model to choose from.
        self.auto_disambiguation = auto_disambiguation
        self._auto_disambig_hops = 0  # guards the auto-resolution chain per search
        self.search_time = 0
        self.num_searches = 0

    def reset(self):
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self._auto_disambig_hops = 0
        self.reset_answer()

    @staticmethod
    def _split_sentences(text):
        """Split page text into cleaned sentences.

        Paragraphs (newline-separated) are split on '. '; each piece gets a
        single trailing period, but only if it does not already end in sentence
        punctuation. That guard fixes the ".." that appeared when a
        paragraph-final fragment kept its own period and a second one was
        appended (e.g. "...from 2007 to 2015.." ).
        """
        sentences = []
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                continue
            for s in para.split(". "):
                s = s.strip()
                if not s:
                    continue
                if s[-1] not in ".!?":
                    s += "."
                sentences.append(s)
        return sentences

    @staticmethod
    def get_page_obs(page):
        return " ".join(WikiEnv._split_sentences(page)[:5])

    def construct_lookup_list(self, keyword):
        if self.page is None:
            return []
        sentences = self._split_sentences(self.page)
        kw = keyword.lower()
        # Each hit is the matched sentence with LOOKUP_CONTEXT sentences of
        # context on each side, so the model sees the surrounding text.
        return [
            " ".join(sentences[max(0, i - LOOKUP_CONTEXT):i + LOOKUP_CONTEXT + 1])
            for i, s in enumerate(sentences)
            if kw in s.lower()
        ]

    def _set_page_from_paragraphs(self, paragraphs):
        """Build `self.page` from a page's paragraph texts and set the obs.

        Shared by both backends: keeps only paragraphs longer than two words,
        cleans them, and exposes the first 5 sentences as the observation.
        """
        self.page = ""
        for p in paragraphs:
            if len(p.split(" ")) > 2:
                self.page += clean_str(p)
                if not p.endswith("\n"):
                    self.page += "\n"
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        return self.obs

    @staticmethod
    def _list_items(soup):
        """Content list-item texts (capped at _MAX_OPTIONS).

        Scoping to `mw-parser-output` keeps page chrome (navigation, footers)
        out so only the article's own list items are returned.
        """
        content = soup.find(class_="mw-parser-output") or soup
        options = []
        for li in content.find_all("li"):
            text = " ".join(li.get_text().split())
            if text:
                options.append(text)
            if len(options) >= _MAX_OPTIONS:
                break
        return options

    @staticmethod
    def _disambiguation_options(soup):
        """Return the listed options if `soup` is a disambiguation page, else [].

        Gated on the lead text ("... may refer to:") so normal articles that
        merely contain the phrase elsewhere are not misclassified.
        """
        content = soup.find(class_="mw-parser-output") or soup
        lead = " ".join(
            p.get_text().strip() for p in content.find_all("p")[:2]
        ).lower()
        if not _DISAMBIG_RE.search(lead):
            return []
        return WikiEnv._list_items(soup)

    def _render_page(self, soup, entity):
        """Turn a fetched page into an observation.

        Disambiguation pages become a list of options for the model to pick
        from (it must then search the exact title it wants); any other page
        becomes its first few sentences.
        """
        options = self._disambiguation_options(soup)
        if not options:
            # Even when the "... may refer to:" lead is absent (some ZIM dumps
            # render it differently), a page with list items but no real prose
            # paragraphs is an index/disambiguation page in practice -> list its
            # options instead of mashing them into pseudo-prose.
            has_prose = any(len(p.get_text().split()) > 2 for p in soup.find_all("p"))
            if not has_prose:
                options = self._list_items(soup)
        if options:
            # Automatic mode: open the top real option instead of asking the
            # model to choose. The picked page is loaded by re-running search on
            # its title; the hop guard stops a disambiguation-of-disambiguation
            # chain, after which we fall through to listing the options.
            if self.auto_disambiguation and self._auto_disambig_hops < MAX_AUTO_DISAMBIG_HOPS:
                targets = self._disambiguation_targets(soup)
                if targets:
                    target, others = targets[0], targets[1:]
                    self._auto_disambig_hops += 1
                    self.search_step(target)
                    note = (f"'{entity}' could refer to several pages; automatically "
                            f"opened '{target}'.")
                    if others:
                        # Name the alternatives so the model can re-search one of
                        # them if the top option is not the page it needs.
                        note += (f" Other options: {', '.join(others)}. Search one of "
                                 f"these exact titles if you need a different page.")
                    self.obs = f"{note}\n{self.obs}"
                    return self.obs
            self.obs = (
                f"{entity} may refer to several pages. Search the exact title "
                f"of the one you want: {options}"
            )
            return self.obs
        return self._render_article(soup, entity)

    def _disambiguation_targets(self, soup, limit=_MAX_OPTIONS):
        """Ordered page titles a disambiguation list points to (top option first).

        Takes the primary (first real article) link from each list item, skipping
        footnote/citation fragments and external links, and preferring the link's
        `title` attribute -- set to the exact page title by both the ZIM and live
        Wikipedia -- over its anchor text. Duplicates are dropped and the list is
        capped at `limit`. The first element is what auto-disambiguation opens;
        the rest are the alternatives shown to the model.
        """
        content = soup.find(class_="mw-body-content") or soup.find(class_="mw-parser-output") or soup
        titles, seen = [], set()
        for li in content.find_all("li"):
            for a in li.find_all("a", href=True):
                href = a.get("href", "")
                if not href or href.startswith("#") or "cite_note" in href or "cite_ref" in href:
                    continue  # in-page/citation anchor, not an article
                if href.startswith(("http://", "https://", "//")) and "wikipedia.org/wiki/" not in href:
                    continue  # external link
                title = " ".join((a.get("title") or a.get_text()).split())
                if title and title.lower() not in seen:
                    seen.add(title.lower())
                    titles.append(title)
                break  # one target per option row (its primary link)
            if len(titles) >= limit:
                break
        return titles

    def _first_disambiguation_target(self, soup):
        """Title of the top option in a disambiguation list, or None."""
        targets = self._disambiguation_targets(soup)
        return targets[0] if targets else None

    def _render_article(self, soup, entity):
        """Turn a non-disambiguation page's soup into an observation.

        Split out from `_render_page` (which handles title resolution and
        disambiguation, common to every wiki tool set) so subclasses that browse
        the same pages differently -- e.g. WikiStructuredEnv, which parses the
        page into sections -- can override just the article rendering while
        reusing the disambiguation handling.
        """
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        self._set_page_from_paragraphs(paragraphs)
        if not self.page.strip():
            # Never return an empty observation: the model would otherwise
            # fabricate a result. Give an actionable message instead.
            self.obs = (
                f"The page '{entity}' has no readable text. Try a more "
                f"specific title or a different entity."
            )
        return self.obs

    def search_step(self, entity):
        old_time = time.time()
        try:
            if self.backend == "offline":
                self._search_offline(entity)
            else:
                self._search_online(entity)
        finally:
            self.search_time += time.time() - old_time
            self.num_searches += 1
        return self.obs

    def _not_found_obs(self, entity):
        """Message for a failed search, shaped by whether we have suggestions.

        `self.result_titles` must already be set. When the search engine returns
        no near-titles at all, we say so plainly instead of printing an empty
        `Similar: []`, which reads like a bug and gives the model nothing to act
        on.
        """
        similar = self.result_titles[:5]
        if not similar:
            return (
                f"Could not find '{entity}', and no similar page titles were "
                "found. Check the spelling or try a different or more general title."
            )
        return f"Could not find {entity}. Similar: {similar}."

    def _search_online(self, entity):
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = _get(search_url)
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # no exact page -> show similar titles
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = self._not_found_obs(entity)
            return self.obs
        return self._render_page(soup, entity)

    def _search_offline(self, entity):
        archive = _open_archive(self.zim_path)
        entry = self._get_entry(archive, entity)
        if entry is None:  # no exact page -> suggest similar titles
            from libzim.suggestion import SuggestionSearcher  # lazy: only needed offline

            suggestion = SuggestionSearcher(archive).suggest(entity)
            paths = list(suggestion.getResults(0, 5))
            titles = []
            for path in paths:
                try:
                    titles.append(archive.get_entry_by_path(path).title)
                except KeyError:
                    titles.append(path)
            self.result_titles = titles
            self.obs = self._not_found_obs(entity)
            return self.obs
        while entry.is_redirect:  # follow redirects to the real article
            entry = entry.get_redirect_entry()
        html = bytes(entry.get_item().content).decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, features="html.parser")
        return self._render_page(soup, entity)

    @staticmethod
    def _get_entry(archive, entity):
        """Return the ZIM entry for `entity`, or None if absent.

        Tries title lookup first, then a path lookup (namespace-free ZIMs key
        articles by a path that mirrors the title), so we are robust to either
        libzim accessor being the one that resolves the page.
        """
        getters = []
        by_title = getattr(archive, "get_entry_by_title", None)
        if by_title is not None:
            getters.append(by_title)
        getters.append(archive.get_entry_by_path)
        for getter in getters:
            try:
                return getter(entity)
            except KeyError:
                continue
        return None

    # ------------------------------------------------------------------
    # Tool entry points (names match the TOOLS definitions)
    # ------------------------------------------------------------------

    def search(self, entity):
        """search[entity]: first 5 sentences of the page, or top-5 similar entities."""
        _STATS.begin()
        t0 = time.perf_counter()
        self._auto_disambig_hops = 0  # fresh auto-disambiguation budget per search
        try:
            return self.search_step(entity)
        finally:
            _STATS.end("search", time.perf_counter() - t0)

    def lookup(self, string):
        """lookup[string]: next occurrence of `string` in the page (matched sentence +
        one on each side). Call repeatedly with the same string to step through matches."""
        _STATS.begin()
        t0 = time.perf_counter()
        try:
            return self._lookup_step(string)
        finally:
            _STATS.end("lookup", time.perf_counter() - t0)

    def _lookup_step(self, string):
        if self.lookup_keyword != string:  # reset lookup
            self.lookup_keyword = string
            self.lookup_list = self.construct_lookup_list(string)
            self.lookup_cnt = 0
        if self.lookup_cnt >= len(self.lookup_list):
            self.obs = "No more results.\n"
        else:
            self.obs = (f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                        + self.lookup_list[self.lookup_cnt])
            self.lookup_cnt += 1
        return self.obs
