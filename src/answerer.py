"""Answer generation with citations over retrieved documentation chunks.

The module wraps the Yandex Cloud Foundation Models chat-completions API
(AI Studio) and produces an answer whose facts are marked with [#<chunk_id>]
citations referring to the supplied chunks. On API failure a degraded
fallback answer is returned together with the raw chunks.
"""

import re

import requests

GEN_URL = "https://ai.api.cloud.yandex.net/v1/chat/completions"
NOT_FOUND_TEXT = "Не найдено в документации."
FALLBACK_TEXT = (
    "Не удалось получить ответ модели. "
    "Ниже приведены наиболее релевантные фрагменты документации."
)

_SYSTEM_PROMPT = (
    "Ты - ассистент по документации Yandex Cloud (база YaAgentAI). "
    "Отвечай ТОЛЬКО на основе предоставленных фрагментов. "
    "Каждый содержательный факт помечай ссылкой [#<id>] сразу после "
    "предложения, к которому он относится, где <id> - номер фрагмента. "
    "Если в предоставленных фрагментах нет ответа на вопрос, напиши ровно: "
    "Не найдено в документации. Не выдумывай факты и не используй знания "
    "вне фрагментов."
)

_CITATION_RE = re.compile(r"#(\d+)")
_WHITESPACE_RE = re.compile(r"\s+")


def parse_cited_ids(text):
    """Return unique chunk ids referenced as #<id> in the generated text."""
    return [int(x) for x in dict.fromkeys(_CITATION_RE.findall(text or ""))]


def chunk_context_line(chunk):
    """One context line for the generator prompt."""
    text = _WHITESPACE_RE.sub(" ", chunk["text"])[:1200]
    return "#{} | {}\n{}".format(chunk["id"], chunk.get("source", ""), text)


class Answerer:
    """LLM answer generator with citation tracking and graceful fallback.

    Parameters
    ----------
    api_key : str
        Yandex Cloud API key (SAGAAI_YANDEXAI_KEY).
    folder_id : str
        Yandex Cloud folder id (SAGAAI_YANDEXAI_KEY2).
    model : str, optional
        Model URI, default gpt://{folder_id}/aliceai-llm-flash (YandexGPT Lite).
    temperature, max_tokens, timeout : generation parameters.
    attempts : int, number of API retries before the fallback answer.
    """

    def __init__(
        self,
        api_key,
        folder_id,
        model=None,
        temperature=0.2,
        max_tokens=10000,
        timeout=120,
        attempts=3,
    ):
        self.api_key = api_key
        self.folder_id = folder_id
        self.model = model or "gpt://{}/aliceai-llm-flash".format(folder_id)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.attempts = attempts

    def answer(self, question, chunks, max_chunks=5):
        """Generate an answer with citations.

        chunks: list of chunk dicts ({id, source, text}) ordered by relevance.
        Returns {"text", "citations": [chunks actually cited], "fallback": bool}.
        """
        chunks = chunks[:max_chunks]
        if not chunks:
            return {"text": NOT_FOUND_TEXT, "citations": [], "fallback": False}
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(question, chunks)},
        ]
        text = self._call_llm(messages)
        if text is None:
            return {"text": FALLBACK_TEXT, "citations": chunks, "fallback": True}
        if text.strip().rstrip(".,!? \t") == NOT_FOUND_TEXT.rstrip("."):
            return {"text": text, "citations": [], "fallback": False}
        cited_ids = parse_cited_ids(text) or [chunks[0]["id"]]
        cited = []
        for cid in cited_ids:
            chunk = next((c for c in chunks if c["id"] == cid), None)
            if chunk is not None and chunk not in cited:
                cited.append(chunk)
        return {"text": text, "citations": cited, "fallback": False}

    def _build_prompt(self, question, chunks):
        blocks = [chunk_context_line(c) for c in chunks]
        return (
            "Вопрос: {}\n\nФрагменты документации:\n\n{}\n\n"
            "Ответь на вопрос по-русски, цитируя источники."
        ).format(question, "\n\n".join(blocks))

    def _call_llm(self, messages):
        """POST to the chat-completions endpoint; return text or None."""
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
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
                content = resp.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if content:
                return content
        return None
#Developed by YaAgent / SagaAI Platform, 2026. https://github.com/TVD2100/sagaai-platform