"""Streamlit UI for the YaAgentAI documentation search agent.

Pages: Новый запрос / Настройки (sidebar).
Settings are persisted via src.settings (JSON in home dir).
Secrets are read from SAGAAI_YANDEXAI_KEY / SAGAAI_YANDEXAI_KEY2.
"""

import json
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retriever import Retriever
from src.iterative_search import IterativeSearch
from src.settings import (
    DEFAULT_SETTINGS,
    MODELS,
    REASONING_OPTIONS,
    load_settings,
    save_settings,
)

BASE_LABELS = {
    "rag_base": "Yandex AI Studio docs",
    "yandex_cloud_docs": "Yandex Cloud docs",
}
QUERY_CACHE_PATH = os.path.join(PROJECT_ROOT, "rag", "query_cache.json")


def discover_bases():
    """Discover available RAG bases: data/<name>/index.db.

    Returns {name: {"path": str, "label": str}} sorted by name.
    """
    bases = {}
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            db_path = os.path.join(data_dir, name, "index.db")
            if os.path.isfile(db_path):
                label = BASE_LABELS.get(name, name.replace("_", " ").title())
                bases[name] = {"path": db_path, "label": label}
    return bases

st.set_page_config(page_title="Yandex Docs Search", page_icon="🔎", layout="wide")


def get_credentials():
    """Return (api_key, folder_id) or (None, None) when missing."""
    api_key = os.environ.get("SAGAAI_YANDEXAI_KEY")
    folder_id = os.environ.get("SAGAAI_YANDEXAI_KEY2")
    return api_key, folder_id


@st.cache_resource

def get_retriever(api_key, folder_id, db_path):
    """Build the production vector-only retriever for the selected base."""
    return Retriever(
        db_path,
        api_key,
        folder_id,
        query_cache_path=QUERY_CACHE_PATH,
        rps=8,
        use_db_vectors=True,
    )


def get_searcher(api_key, folder_id, settings, db_path):
    """Build the iterative search pipeline from settings."""
    retriever = get_retriever(api_key, folder_id, db_path)
    return IterativeSearch(
        retriever,
        api_key,
        folder_id,
        model="gpt://{}/{}".format(folder_id, settings["model"]),
        max_iterations=settings["max_iterations"],
        top_n=settings["top_n"],
        max_chunks=settings["max_chunks"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        reasoning_effort=settings["reasoning_effort"],
    )


def apply_light_theme():
    """Inject light-theme CSS and enlarge the sidebar menu label."""
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; }
        .stApp, .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2,
        .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
        .stCaption, .stText, label { color: #111111 !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            font-size: 1.1em;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(text):
    """Render a Copy button that copies text to the clipboard via JS."""
    payload = json.dumps(text, ensure_ascii=False)
    components.html(
        """
        <script>
        const text = {payload};
        function copy() {{
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.getElementById('copy-btn');
                btn.textContent = 'Скопировано!';
                setTimeout(() => {{ btn.textContent = 'Copy'; }}, 1500);
            }});
        }}
        </script>
        <button id="copy-btn" onclick="copy()" style="padding:0.5em 1em;border-radius:0.5em;border:1px solid #ccc;background:transparent;cursor:pointer;">Copy</button>
        """.format(payload=payload),
        height=50,
    )


def render_chunk(chunk):
    """Render one chunk as an expander with source and text."""
    with st.expander("Фрагмент #{} - {}".format(chunk["id"], chunk.get("source", ""))):
        st.write(chunk["text"])


def render_citations(citations):
    """Render cited chunks."""
    if not citations:
        return
    st.markdown("### Цитированные фрагменты")
    for chunk in citations:
        render_chunk(chunk)


def render_iterations(iterations):
    """Render the iterative search trace."""
    if not iterations:
        return
    st.markdown("### Итерации поиска")
    for item in iterations:
        keep = item.get("keep_ids") or "-"
        st.markdown(
            "**Итерация {}**: запросы: {}; сохранено чанков: {}; новых чанков: {}".format(
                item["iteration"],
                ", ".join(item["queries"]),
                keep,
                item["new_ids"],
            )
        )


def render_settings_page():
    """Settings form: model, reasoning, iterations, top_n, max_chunks, temperature, max_tokens, debug."""
    st.header("Настройки")
    settings = load_settings()

    with st.form("settings_form"):
        model = st.selectbox(
            "Модель",
            options=MODELS,
            index=MODELS.index(settings["model"]) if settings["model"] in MODELS else 0,
        )
        reasoning_options = REASONING_OPTIONS.get(model, ["medium"])
        reasoning_default = settings["reasoning_effort"]
        if reasoning_default not in reasoning_options:
            reasoning_default = "medium" if "medium" in reasoning_options else reasoning_options[0]
        reasoning_effort = st.selectbox(
            "Уровень рассуждений",
            options=reasoning_options,
            index=reasoning_options.index(reasoning_default),
        )
        max_iterations = st.slider("Глубина итеративного поиска (1-10)", 1, 10, settings["max_iterations"])
        top_n = st.slider("Фрагментов на запрос (3-20)", 3, 20, settings["top_n"])
        max_chunks = st.slider("Максимум фрагментов в контексте (3-20)", 3, 20, settings["max_chunks"])
        temperature = st.slider("Температура (0.0-1.0)", 0.0, 1.0, settings["temperature"], 0.05)
        max_tokens = st.slider("Максимум токенов ответа (500-10000)", 500, 10000, settings["max_tokens"], 100)
        show_debug = st.checkbox("Показывать служебную информацию", settings["show_debug"])
        submitted = st.form_submit_button("Сохранить")

    if submitted:
        new_settings = {
            "base": settings["base"],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_iterations": max_iterations,
            "top_n": top_n,
            "max_chunks": max_chunks,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "show_debug": show_debug,
        }
        save_settings(new_settings)
        st.session_state["settings_saved"] = True
        st.rerun()

    if st.session_state.get("settings_saved"):
        st.success("Настройки сохранены")
        st.session_state["settings_saved"] = False

    st.markdown("---")
    st.markdown("### Как задать ключи для работы Yandex моделей")
    st.markdown(
        "Для работы приложения необходимы два значения из Yandex AI Studio: "
        "**API-ключ** и **идентификатор каталога (Folder ID)**. "
        "Задайте их как переменные окружения до запуска приложения:\n\n"
        "**macOS / Linux (bash):**\n"
        "```bash\n"
        "export SAGAAI_YANDEXAI_KEY=\"ваш_api_ключ\"\n"
        "export SAGAAI_YANDEXAI_KEY2=\"ваш_folder_id\"\n"
        "streamlit run app.py\n"
        "```\n\n"
        "**Windows (PowerShell):**\n"
        "```powershell\n"
        "$env:SAGAAI_YANDEXAI_KEY=\"ваш_api_ключ\"\n"
        "$env:SAGAAI_YANDEXAI_KEY2=\"ваш_folder_id\"\n"
        "streamlit run app.py\n"
        "```\n"
    )


def render_query_page(settings):
    """Query page: base selector, question input, live progress, answer, export buttons."""
    st.header("Вопрос-ответ")
    bases = discover_bases()
    if not bases:
        st.error(
            "Не найдено ни одной базы документации. Поместите базу в папку "
            "data/<имя_базы>/index.db и перезапустите приложение."
        )
        return
    base_names = list(bases)
    default_base = settings["base"] if settings["base"] in bases else base_names[0]
    base = st.selectbox(
        "Поиск по документации",
        options=base_names,
        format_func=lambda x: bases[x]["label"],
        index=base_names.index(default_base),
    )
    question = st.text_area(
        "Ваш вопрос по документации",
        placeholder="Например: Как ограничить число вызовов web_search в моделях?",
        height=100,
    )
    if st.button("Получить ответ", type="primary"):
        if not question.strip():
            st.warning("Введите вопрос.")
            return

        api_key, folder_id = get_credentials()
        if not api_key or not folder_id:
            st.error(
                "Не заданы переменные окружения SAGAAI_YANDEXAI_KEY и "
                "SAGAAI_YANDEXAI_KEY2. Задайте их и перезапустите приложение."
            )
            return

        with st.status("Ищу и анализирую документацию...", expanded=True) as status:
            searcher = get_searcher(api_key, folder_id, settings, bases[base]["path"])

            def on_progress(iteration, queries):
                status.update(
                    label="Итерация {}: запросы: {}".format(iteration, ", ".join(queries)),
                    state="running",
                )

            out = searcher.search(question.strip(), progress_callback=on_progress)
            status.update(label="Готово", state="complete")
        st.session_state["last_out"] = out

    if "last_out" in st.session_state:
        out = st.session_state["last_out"]
        st.markdown("### Ответ")
        answer_text = out["result"]["text"]
        st.write(answer_text)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Скачать Markdown",
                data=answer_text.encode("utf-8"),
                file_name="answer.md",
                mime="text/markdown",
            )
        with col2:
            render_copy_button(answer_text)

        if out["result"]["fallback"]:
            st.warning("Не удалось получить ответ модели; показаны найденные фрагменты.")

        if settings["show_debug"]:
            render_iterations(out["iterations"])
            render_citations(out["result"]["citations"])
            st.markdown("### Найденные фрагменты ({})".format(len(out["chunks"])))
            for chunk in out["chunks"][:10]:
                render_chunk(chunk)


def main():
    settings = load_settings()
    apply_light_theme()

    st.title("🔎 Yandex Docs Search")
    bases = discover_bases()
    if bases:
        labels = ", ".join(bases[x]["label"] for x in bases)
        st.caption("Поиск по документации: {}".format(labels))
    else:
        st.caption("Поиск по документации: базы не найдены")

    api_key, folder_id = get_credentials()
    if not api_key or not folder_id:
        st.error(
            "Не заданы переменные окружения SAGAAI_YANDEXAI_KEY и "
            "SAGAAI_YANDEXAI_KEY2. Задайте их и перезапустите приложение."
        )
        return

    page = st.sidebar.radio("Меню", ["Новый запрос", "Настройки"])
    st.sidebar.markdown(
        "Developed by YaAgent / <a href=\"https://github.com/TVD2100/sagaai-platform\" "
        "target=\"_blank\">SagaAI Platform</a>, 2026",
        unsafe_allow_html=True,
    )
    if page == "Настройки":
        render_settings_page()
    else:
        render_query_page(settings)


if __name__ == "__main__":
    main()
#Developed by YaAgent / SagaAI Platform, 2026. https://github.com/TVD2100/sagaai-platform