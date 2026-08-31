"""Settings persistence for the YaAgentAI search app.

Settings are stored as JSON in ~/.yaagent_search_settings.json.
The module provides DEFAULT_SETTINGS, load_settings() and save_settings().
"""

import json
import os

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".yaagent_search_settings.json")

MODELS = [
    "deepseek-v4-flash",
    "yandexgpt-5-lite",
    "yandexgpt-5-pro",
    "yandexgpt-5.1",
    "aliceai-llm",
    "aliceai-llm-flash",
    "qwen3-235b-a22b-fp8",
    "qwen3.6-35b-a3b",
    "gpt-oss-120b",
    "gpt-oss-20b",
]

REASONING_OPTIONS = {
    "yandexgpt-5-lite": ["none", "minimal", "low", "medium", "high", "xhigh"],
    "yandexgpt-5-pro": ["none", "minimal", "low", "medium", "high", "xhigh"],
    "yandexgpt-5.1": ["none", "minimal", "low", "medium", "high", "xhigh"],
    "aliceai-llm": ["none", "minimal", "low", "medium", "high"],
    "aliceai-llm-flash": ["none", "minimal", "low", "medium", "high"],
    "deepseek-v4-flash": ["none", "minimal", "low", "medium", "high", "xhigh"],
    "qwen3-235b-a22b-fp8": ["low", "medium", "high"],
    "qwen3.6-35b-a3b": ["none", "minimal", "low", "medium", "high", "xhigh"],
    "gpt-oss-120b": ["low", "medium", "high"],
    "gpt-oss-20b": ["low", "medium", "high"],
}

DEFAULT_SETTINGS = {
    "base": "rag_base",
    "model": "deepseek-v4-flash",
    "reasoning_effort": "high",
    "max_iterations": 7,
    "top_n": 10,
    "max_chunks": 15,
    "temperature": 0.2,
    "max_tokens": 10000,
    "show_debug": False,
}


def load_settings():
    """Load settings from disk, merging with defaults."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            settings.update(data)
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings):
    """Persist settings to disk."""
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
#Developed by YaAgent / SagaAI Platform, 2026. https://github.com/TVD2100/sagaai-platform