"""Iterative RAG search with function calling and chunk accumulation.

Architecture (as specified):
1. The user question is sent to the LLM together with the search_rag tool.
2. The LLM returns a function call with a search query (and, after the first
   round, keep_chunk_ids - which of the shown chunks to keep in context).
3. The system searches the RAG base (top_n=10 chunks with ids) and returns
   the chunk texts to the LLM.
4. The LLM either answers with [#id] citations or calls search_rag again
   with a new query and the ids of useful chunks to keep.
5. The loop repeats up to max_iterations (default 10), accumulating useful
   chunks in context.

LLM access follows project conventions: chat/completions endpoint,
Api-Key / x-project headers, model URI gpt://{folder_id}/<model>.
"""

import json
import re

import requests

from src.answerer import parse_cited_ids

GEN_URL = "https://ai.api.cloud.yandex.net/v1/chat/completions"

NOT_FOUND_TEXT = "Не найдено в документации."

SYSTEM_PROMPT = (
    'Ты - ассистент по документации Yandex Cloud (база YaAgentAI). '
    'Отвечай ТОЛЬКО на основе фрагментов документации, которые тебе предоставят. '
    'Для ответа на вопрос сначала вызови функцию search_rag с поисковым запросом. '
    'Система вернёт до 10 фрагментов документации с индексами (#id). '
    'После получения фрагментов оцени, достаточно ли их для полного и точного ответа: '
    '- Если достаточно - сформулируй ответ по-русски, помечая каждый содержательный '
    'факт ссылкой [#id] сразу после предложения, к которому он относится. '
    '- Если НЕдостаточно - вызови search_rag снова: query (новый поисковый запрос, '
    'который найдёт недостающие фрагменты) и keep_chunk_ids (список ID фрагментов '
    'из показанных, которые полезны для ответа и должны остаться в контексте). '
    'В документации используются технические термины: max_tool_calls, tool_choice, '
    'parallel_tool_calls, allowed_domains, search_context_size, max_num_results, '
    'vector_store_ids, reasoning_effort, temperature, top_p, max_output_tokens. '
    'Включай их в поисковые запросы, когда вопрос касается соответствующих параметров. '
    'Если в предоставленных фрагментах нет ответа на вопрос, напиши ровно: '
    'Не найдено в документации.'
)

FINAL_ANSWER_PROMPT = (
    'Ты - ассистент по документации Yandex Cloud (база YaAgentAI). '
    'Отвечай ТОЛЬКО на основе предоставленных фрагментов. Каждый содержательный '
    'факт помечай ссылкой [#<id>] сразу после предложения, к которому он '
    'относится, где <id> - номер фрагмента. Если в предоставленных фрагментах '
    'нет ответа на вопрос, напиши ровно: Не найдено в документации. '
    'Не выдумывай факты и не используй знания вне фрагментов.\n\n'
    'Вопрос: {question}\n\nФрагменты документации:\n\n{context}'
)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_rag",
        "description": (
            "Поиск по базе документации Yandex Cloud. "
            "Возвращает до 10 фрагментов с индексами."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос к базе документации",
                },
                "keep_chunk_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "ID фрагментов из показанных, которые нужно сохранить "
                        "в контексте (пустой список на первом вызове)"
                    ),
                },
            },
            "required": ["query", "keep_chunk_ids"],
            "additionalProperties": False,
        },
    },
}

_WHITESPACE_RE = re.compile(r"\s+")


def _chunk_line(chunk, max_chars=1200):
    """One context line for LLM prompts."""
    text = _WHITESPACE_RE.sub(" ", chunk["text"])[:max_chars]
    return "#{} | {}\n{}".format(chunk["id"], chunk.get("source", ""), text)


def _parse_tool_arguments(raw_calls):
    """Parse raw tool_calls from the API into (name, arguments) pairs.

    Returns a list of dicts with 'name' and 'arguments' (parsed JSON object).
    Malformed calls are skipped.
    """
    parsed = []
    for call in raw_calls or []:
        if not isinstance(call, dict) or call.get("type") != "function":
            continue
        fn = call.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments")
        if not name:
            continue
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (TypeError, ValueError):
            continue
        if not isinstance(args, dict):
            continue
        parsed.append({"name": name, "arguments": args})
    return parsed


class IterativeSearch:
    """Iterative RAG pipeline driven by LLM function calls.

    Parameters
    ----------
    retriever : object with .by_id and .search_hybrid(query, top_n)
    api_key, folder_id : Yandex Cloud credentials.
    model : str, optional
        Model URI for the LLM (default deepseek-v4-flash).
    max_iterations : int, cap for search rounds (default 10).
    top_n : int, chunks retrieved per query (default 10).
    max_chunks : int, chunks passed to the final fallback prompt (default 10).
    temperature, max_tokens, timeout, attempts : generation parameters.
    """

    def __init__(
        self,
        retriever,
        api_key,
        folder_id,
        model=None,
        max_iterations=10,
        top_n=10,
        max_chunks=10,
        temperature=0.2,
        max_tokens=10000,
        timeout=120,
        attempts=3,
        reasoning_effort=None,
    ):
        self.retriever = retriever
        self.api_key = api_key
        self.folder_id = folder_id
        self.model = model or "gpt://{}/deepseek-v4-flash".format(folder_id)
        self.max_iterations = max(1, max_iterations)
        self.top_n = top_n
        self.max_chunks = max_chunks
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.attempts = attempts
        self.reasoning_effort = reasoning_effort

    def _call_llm(self, messages, tools=None, max_tokens=10000):
        """Single chat-completions call.

        Returns (content, raw_tool_calls) where content is the assistant text
        (possibly empty) and raw_tool_calls is the original API list of tool
        calls (possibly empty). Returns (None, None) on failure.
        """
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
        headers = {
            "Authorization": "Api-Key " + self.api_key,
            "x-project": self.folder_id,
        }
        for _ in range(self.attempts):
            try:
                resp = requests.post(
                    GEN_URL, json=body, headers=headers, timeout=self.timeout
                )
            except requests.RequestException:
                continue
            if resp.status_code != 200:
                continue
            try:
                msg = resp.json()["choices"][0]["message"]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            content = msg.get("content") or ""
            raw_calls = msg.get("tool_calls") or []
            return content, raw_calls
        return None, None

    def _search(self, query):
        """Return ordered chunk dicts for one query (vector search)."""
        ranked = self.retriever.search_vector(query, top_n=self.top_n)
        chunks = []
        for cid, _ in ranked:
            chunk = self.retriever.by_id.get(cid)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _context_text(self, chunks):
        """Render context block for prompts."""
        return "\n\n".join(_chunk_line(c) for c in chunks)

    def _answer_from_chunks(self, question, chunks):
        """Generate a final answer from chunks without tools (fallback path).

        Returns {"text", "citations": [chunk dicts], "fallback": bool}.
        """
        if not chunks:
            return {"text": NOT_FOUND_TEXT, "citations": [], "fallback": False}
        prompt = FINAL_ANSWER_PROMPT.replace("{question}", question).replace(
            "{context}", self._context_text(chunks[: self.max_chunks])
        )
        text, _ = self._call_llm(
            [{"role": "user", "content": prompt}], max_tokens=self.max_tokens
        )
        if text is None:
            return {"text": NOT_FOUND_TEXT, "citations": [], "fallback": True}
        cited_ids = parse_cited_ids(text)
        cited = []
        for cid in cited_ids:
            chunk = next((c for c in chunks if c["id"] == cid), None)
            if chunk is not None and chunk not in cited:
                cited.append(chunk)
        return {"text": text, "citations": cited, "fallback": False}

    def search(self, question, progress_callback=None):
        """Run the full iterative function-calling pipeline.

        Parameters
        ----------
        question : str
            The user question.
        progress_callback : callable, optional
            Called as progress_callback(iteration, queries) before each search
            round so the UI can show live progress.

        Returns a dict with:
        - result: {"text", "citations", "fallback"}
        - chunks: chunks shown in the final round (accumulated + new)
        - iterations: [{"iteration", "queries", "keep_ids", "new_ids",
          "accumulated_ids"}]
        - sufficient: bool (True when the LLM produced the final answer)
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        accumulated = []
        shown = []
        iterations = []
        sufficient = False
        final_result = None

        for iteration in range(1, self.max_iterations + 1):
            content, raw_calls = self._call_llm(
                messages, tools=[SEARCH_TOOL], max_tokens=self.max_tokens
            )
            if content is None and not raw_calls:
                final_result = {
                    "text": NOT_FOUND_TEXT,
                    "citations": [],
                    "fallback": True,
                }
                break  # API failure

            if raw_calls:
                calls = _parse_tool_arguments(raw_calls)
                queries = []
                keep_ids = []
                new_chunks = []
                seen_new = set()
                for call in calls:
                    if call["name"] != "search_rag":
                        continue
                    args = call["arguments"]
                    query = args.get("query")
                    if not isinstance(query, str) or not query.strip():
                        continue
                    queries.append(query.strip())
                    kid = args.get("keep_chunk_ids") or []
                    if isinstance(kid, list):
                        for x in kid:
                            if isinstance(x, bool):
                                continue
                            if isinstance(x, (int, float)):
                                keep_ids.append(int(x))
                    for chunk in self._search(query.strip()):
                        if chunk["id"] not in seen_new:
                            seen_new.add(chunk["id"])
                            new_chunks.append(chunk)

                if not queries:
                    break  # model called the tool without a valid query

                if progress_callback is not None:
                    progress_callback(iteration, queries)

                if keep_ids:
                    keep_set = set(keep_ids)
                    accumulated = [c for c in shown if c["id"] in keep_set]

                acc_ids = {c["id"] for c in accumulated}
                new_chunks = [c for c in new_chunks if c["id"] not in acc_ids]
                shown = accumulated + new_chunks

                iterations.append(
                    {
                        "iteration": iteration,
                        "queries": queries,
                        "keep_ids": keep_ids,
                        "new_ids": [c["id"] for c in new_chunks],
                        "accumulated_ids": [c["id"] for c in accumulated],
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": raw_calls,
                    }
                )
                results_text = (
                    "Результаты поиска по запросу: {}\n\n{}".format(
                        "; ".join(queries), self._context_text(shown)
                    )
                )
                messages.append({"role": "user", "content": results_text})
                continue

            # The model produced a text answer.
            sufficient = True
            cited_ids = parse_cited_ids(content)
            cited = []
            for cid in cited_ids:
                chunk = next((c for c in shown if c["id"] == cid), None)
                if chunk is not None and chunk not in cited:
                    cited.append(chunk)
            final_result = {"text": content, "citations": cited, "fallback": False}
            break

        if final_result is None:
            final_result = self._answer_from_chunks(question, shown)

        return {
            "result": final_result,
            "chunks": shown,
            "iterations": iterations,
            "sufficient": sufficient,
        }
#Developed by YaAgent / SagaAI Platform, 2026. https://github.com/TVD2100/sagaai-platform