"""Tests for the forced final answer on the last search iteration.

When the model exhausts max_iterations while still requesting search_rag,
the pipeline must append the new chunks, tell the model this is the last
iteration, and force a final answer without further tool calls.
"""

from src.iterative_search import FINAL_ITERATION_PROMPT, IterativeSearch


class FakeRetriever:
    """Minimal retriever with fixed vector results."""

    def __init__(self):
        self.by_id = {
            1: {"id": 1, "source": "doc1", "text": "chunk one"},
            2: {"id": 2, "source": "doc2", "text": "chunk two"},
            3: {"id": 3, "source": "doc3", "text": "chunk three"},
        }

    def search_vector(self, query, top_n=10):
        return [(1, 0.9), (2, 0.8), (3, 0.7)]


def _tool_call(query, keep=None):
    """Build a raw search_rag tool call as returned by the API."""
    return {
        "type": "function",
        "function": {
            "name": "search_rag",
            "arguments": '{"query": "%s", "keep_chunk_ids": %s}'
            % (query, keep or []),
        },
    }


def test_last_iteration_forces_final_answer_without_tools():
    retriever = FakeRetriever()
    searcher = IterativeSearch(
        retriever, api_key="k", folder_id="f", max_iterations=3
    )

    calls = []

    def fake_call_llm(messages, tools=None, max_tokens=10000):
        calls.append({"tools": tools, "messages": messages})
        if tools is not None:
            # First three rounds request another search.
            return "", [_tool_call("query %d" % len(calls))]
        # Final forced call returns the answer.
        return "Ответ с цитатой [#1]", []

    searcher._call_llm = fake_call_llm

    result = searcher.search("Вопрос")

    assert result["sufficient"] is True
    assert result["result"]["text"] == "Ответ с цитатой [#1]"
    assert result["result"]["fallback"] is False
    assert [c["id"] for c in result["result"]["citations"]] == [1]
    # 3 search rounds + 1 forced final call.
    assert len(calls) == 4
    # The final call must not expose tools.
    assert calls[-1]["tools"] is None
    # The final call must include the forced final prompt.
    assert any(
        m.get("content") == FINAL_ITERATION_PROMPT for m in calls[-1]["messages"]
    )
    # All three search rounds must have happened.
    assert len(result["iterations"]) == 3


def test_last_iteration_falls_back_when_final_call_fails():
    retriever = FakeRetriever()
    searcher = IterativeSearch(
        retriever, api_key="k", folder_id="f", max_iterations=2
    )

    calls = []

    def fake_call_llm(messages, tools=None, max_tokens=10000):
        calls.append({"tools": tools})
        if tools is not None:
            return "", [_tool_call("query %d" % len(calls))]
        # Final forced call fails (API error).
        return None, None

    searcher._call_llm = fake_call_llm

    result = searcher.search("Вопрос")

    assert result["sufficient"] is False
    assert result["result"]["text"] == "Не найдено в документации."
    assert result["result"]["fallback"] is True
    assert len(calls) == 3  # 2 search rounds + 1 failed final call
