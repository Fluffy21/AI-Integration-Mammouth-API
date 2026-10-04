import json
import os
import time
import uuid
from pathlib import Path
import io
import base64

from pypdf import PdfReader
from docx import Document
import requests
import streamlit as st

# ---------- Konfiguration ----------

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent / "data")))
CHATS_DIR = DATA_DIR / "chats"
FILES_DIR = DATA_DIR / "files"

PROMPTS_FILE = DATA_DIR / "prompts.json"
MODELS_CACHE_FILE = DATA_DIR / "models_cache.json"

CHATS_DIR.mkdir(parents=True, exist_ok=True)
FILES_DIR.mkdir(parents=True, exist_ok=True)

MODELS_URL = "https://api.mammouth.ai/public/models"
API_URL = "https://api.mammouth.ai/v1/chat/completions"
API_KEY = os.environ.get("MAMMOUTH_API_KEY", "")

CAVEMAN_PROMPT = (
    "Du bist im Caveman Mode. Antworte extrem direkt und kurz. "
    "Keine Hoeflichkeitsfloskeln, keine Wiederholung der Frage, keine Fuellwoerter, "
    "keine ungefragten Erklaerungen. Nur die Antwort, inhaltlich korrekt, minimaler Tokenverbrauch."
)
DEFAULT_PROMPTS = {"Caveman": CAVEMAN_PROMPT}

PROVIDER_PREFIXES = [
    ("claude", "Anthropic"),
    ("gpt", "OpenAI"),
    ("o1", "OpenAI"),
    ("o3", "OpenAI"),
    ("text-embedding", "OpenAI"),
    ("gemini", "Google"),
    ("grok", "xAI"),
    ("deepseek", "DeepSeek"),
    ("qwen", "Alibaba"),
    ("glm", "Zhipu"),
    ("llama", "Meta"),
    ("mistral", "Mistral AI"),
    ("codestral", "Mistral AI"),
    ("devstral", "Mistral AI"),
    ("kimi", "Moonshot"),
    ("minimax", "MiniMax"),
    ("sonar", "Perplexity"),
    ("mammouth", "Mammouth"),
]


def provider_of(model_id: str) -> str:
    low = model_id.lower()
    for prefix, name in PROVIDER_PREFIXES:
        if low.startswith(prefix):
            return name
    return "Unbekannt"


# ---------- Persistenz: Prompts ----------

def load_prompts() -> dict:
    if PROMPTS_FILE.exists():
        try:
            data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
            for k, v in DEFAULT_PROMPTS.items():
                data.setdefault(k, v)
            return data
        except Exception:
            pass
    return dict(DEFAULT_PROMPTS)


def save_prompts(prompts: dict):
    try:
        PROMPTS_FILE.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        st.error(f"Presets konnten nicht gespeichert werden ({PROMPTS_FILE}): {e}")
        raise


# ---------- Persistenz: Chats (je 1 JSON-Datei) ----------

def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def list_chats() -> list:
    chats = []
    for f in CHATS_DIR.glob("*.json"):
        try:
            chats.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    chats.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return chats


def save_chat(chat: dict):
    chat["updated"] = time.time()
    try:
        chat_path(chat["id"]).write_text(json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        st.error(f"Chat konnte nicht gespeichert werden ({chat_path(chat['id'])}): {e}")
        raise


def delete_chat(chat_id: str):
    p = chat_path(chat_id)
    if p.exists():
        p.unlink()


def new_chat() -> dict:
    prompts = load_prompts()

    chat = {
        "id": str(uuid.uuid4()),
        "name": "Neuer Chat",
        "model": "mammouth-recommended",
        "system_prompt": prompts.get("Caveman", CAVEMAN_PROMPT),
        "messages": [],
        "files": [],
        "created": time.time(),
        "updated": time.time(),
    }

    save_chat(chat)

    return chat


# ---------- Modellliste (stuendlich gecacht) ----------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_models() -> list:
    try:
        r = requests.get(MODELS_URL, timeout=10)
        r.raise_for_status()
        raw = r.json().get("data", [])
        models = []
        for m in raw:
            info = m.get("model_info", {}) or {}
            in_cost = info.get("input_cost_per_token")
            out_cost = info.get("output_cost_per_token")
            models.append({
                "id": m["id"],
                "provider": provider_of(m["id"]),
                "input_per_m": round(in_cost * 1_000_000, 4) if in_cost is not None else None,
                "output_per_m": round(out_cost * 1_000_000, 4) if out_cost is not None else None,
            })
        models.sort(key=lambda x: (x["provider"], x["id"]))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_CACHE_FILE.write_text(
            json.dumps({"ts": time.time(), "models": models}, ensure_ascii=False), encoding="utf-8"
        )
        return models
    except Exception as e:
        if MODELS_CACHE_FILE.exists():
            try:
                return json.loads(MODELS_CACHE_FILE.read_text(encoding="utf-8"))["models"]
            except Exception:
                pass
        st.warning(f"Modelle konnten nicht geladen werden: {e}")
        return [{"id": "mammouth-recommended", "provider": "Mammouth", "input_per_m": None, "output_per_m": None}]


def format_model_option(m: dict) -> str:
    inp = f"${m['input_per_m']}" if m.get("input_per_m") is not None else "?"
    out = f"${m['output_per_m']}" if m.get("output_per_m") is not None else "?"
    return f"{m['provider']} | {m['id']} | In: {inp}/1M | Out: {out}/1M"

# ---------- Datei-Handling ----------

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".kts",
    ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".xml", ".yaml", ".yml",
    ".json", ".csv", ".log", ".ini", ".conf",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
}


def extract_file_content(uploaded_file) -> str:
    """Extrahiert Text aus einer hochgeladenen Datei. Bilder liefern '' (werden als Bild, nicht als Text, an die API gegeben)."""
    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return ""

    if extension in TEXT_EXTENSIONS:
        return uploaded_file.getvalue().decode("utf-8", errors="replace")

    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)

    if extension == ".docx":
        document = Document(io.BytesIO(uploaded_file.getvalue()))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    raise ValueError(f"Dateityp '{extension}' wird nicht unterstuetzt.")


def save_uploaded_file(chat_id: str, uploaded_file) -> dict:
    """Speichert die Originaldatei und den extrahierten Text."""
    chat_file_dir = FILES_DIR / chat_id
    chat_file_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    extension = Path(uploaded_file.name).suffix.lower()

    original_path = chat_file_dir / f"{file_id}{extension}"
    text_path = chat_file_dir / f"{file_id}.txt"

    file_bytes = uploaded_file.getvalue()
    original_path.write_bytes(file_bytes)

    content = extract_file_content(uploaded_file)
    text_path.write_text(content, encoding="utf-8")

    return {
        "id": file_id,
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "size": len(file_bytes),
        "original_path": str(original_path),
        "text_path": str(text_path),
    }


def load_file_content(file_info: dict) -> str:
    """Laedt den extrahierten Text einer gespeicherten Datei."""
    path = Path(file_info["text_path"])
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def build_image_content_parts(files: list) -> list:
    """Baut OpenAI-kompatible image_url Content-Parts aus gespeicherten Bild-Dateien."""
    parts = []
    for f in files or []:
        ext = Path(f["name"]).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        img_path = Path(f["original_path"])
        if not img_path.exists():
            continue
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        mime = f.get("type") or IMAGE_MIME_BY_EXT.get(ext, "image/png")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


# ---------- Mammouth API ----------

def call_mammouth(model: str, system_prompt: str, messages: list, files: list | None = None) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    text_files = [f for f in (files or []) if Path(f["name"]).suffix.lower() not in IMAGE_EXTENSIONS]
    image_files = [f for f in (files or []) if Path(f["name"]).suffix.lower() in IMAGE_EXTENSIONS]

    context_messages = []

    if text_files:
        file_context = []
        for file_info in text_files:
            content = load_file_content(file_info)
            if not content:
                continue
            file_context.append(
                f"===== DATEI: {file_info['name']} =====\n\n{content}\n\n===== ENDE DATEI: {file_info['name']} ====="
            )

        if file_context:
            context_messages.append({
                "role": "system",
                "content": (
                    "Dem Nutzer stehen folgende Dateien als Kontext zur Verfuegung. "
                    "Verwende sie bei der Beantwortung der Anfrage, wenn relevant.\n\n"
                    + "\n\n".join(file_context)
                ),
            })

    final_messages = list(messages)
    image_parts = build_image_content_parts(image_files)
    if image_parts and final_messages and final_messages[-1].get("role") == "user":
        last = final_messages[-1]
        final_messages[-1] = {
            "role": "user",
            "content": [{"type": "text", "text": last["content"]}] + image_parts,
        }

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + context_messages + final_messages,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ---------- UI ----------

st.set_page_config(page_title="Mammouth Multi-Chat", layout="wide")

if not API_KEY:
    st.error("MAMMOUTH_API_KEY ist nicht gesetzt (Umgebungsvariable bzw. .env).")

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

with st.sidebar:
    st.header("Chats")
    st.caption(f"Datenordner: {DATA_DIR}")
    if st.button("+ Neuer Chat", use_container_width=True):
        c = new_chat()
        st.session_state.current_chat_id = c["id"]
        st.rerun()
    all_chats = list_chats()
    for c in all_chats:
        active = c["id"] == st.session_state.current_chat_id
        label = ("-> " if active else "") + (c.get("name") or "Chat")
        if st.button(label, key=f"select_{c['id']}", use_container_width=True):
            st.session_state.current_chat_id = c["id"]
            st.rerun()
    if not all_chats:
        st.caption("Noch keine Chats.")

if st.session_state.current_chat_id is None and all_chats:
    st.session_state.current_chat_id = all_chats[0]["id"]

if st.session_state.current_chat_id is None:
    st.info("Links einen neuen Chat anlegen.")
    st.stop()

cp = chat_path(st.session_state.current_chat_id)
if not cp.exists():
    st.session_state.current_chat_id = None
    st.rerun()

chat = json.loads(cp.read_text(encoding="utf-8"))
cid = chat["id"]

# --- Name / Speichern / Loeschen ---
c1, c2, c3 = st.columns([4, 1, 1])
with c1:
    new_name = st.text_input(
        "Name", value=chat["name"], key=f"name_{cid}",
        label_visibility="collapsed", placeholder="Chat-Name",
    )
with c2:
    if st.button("Speichern", key=f"save_name_{cid}", use_container_width=True):
        chat["name"] = new_name
        save_chat(chat)
        st.rerun()
with c3:
    if st.button("Chat loeschen", key=f"delete_{cid}", use_container_width=True):
        delete_chat(cid)
        st.session_state.current_chat_id = None
        st.rerun()

# --- Modell-Dropdown ---
models = fetch_models()
model_lookup = {m["id"]: m for m in models}
options = [m["id"] for m in models]
if chat["model"] not in model_lookup:
    options = [chat["model"]] + options
    model_lookup[chat["model"]] = {"id": chat["model"], "provider": "?", "input_per_m": None, "output_per_m": None}

selected_model = st.selectbox(
    "Modell (Provider | Modell | Input $/1M | Output $/1M)",
    options,
    index=options.index(chat["model"]),
    format_func=lambda mid: format_model_option(model_lookup[mid]),
    key=f"model_{cid}",
)
if selected_model != chat["model"]:
    chat["model"] = selected_model
    save_chat(chat)

# --- System Prompt / Presets ---
# Merkt sich, ob der Expander offen bleiben soll (sonst klappt er bei jedem
# rerun wieder zu und Erfolgs-/Fehlermeldungen "verschwinden" sofort wieder).
exp_key = f"sp_expanded_{cid}"
if exp_key not in st.session_state:
    st.session_state[exp_key] = False

with st.expander("System Prompt", expanded=st.session_state[exp_key]):

    prompts = load_prompts()
    if not prompts:
        prompts = dict(DEFAULT_PROMPTS)
        save_prompts(prompts)

    preset_names = list(prompts.keys())

    def delete_preset(preset_name: str) -> bool:
        """Preset aus prompts.json loeschen (Standard-Presets geschuetzt)."""
        if preset_name not in prompts or preset_name in DEFAULT_PROMPTS:
            return False
        del prompts[preset_name]
        save_prompts(prompts)
        return True

    def apply_preset(cid_: str):
        """Ausgewaehltes Preset auf den Chat anwenden (Callback)."""
        picked = st.session_state.get(f"preset_pick_{cid_}")
        fresh_prompts = load_prompts()
        if picked not in fresh_prompts:
            # Preset existiert nicht (mehr) -> nicht crashen, sondern
            # auf ein gueltiges Preset zurueckfallen.
            fallback = "Caveman" if "Caveman" in fresh_prompts else next(iter(fresh_prompts))
            st.session_state[f"preset_pick_{cid_}"] = fallback
            picked = fallback
        chat_now = json.loads(chat_path(cid_).read_text(encoding="utf-8"))
        chat_now["system_prompt"] = fresh_prompts[picked]
        save_chat(chat_now)
        st.session_state[f"sp_text_{cid_}"] = chat_now["system_prompt"]
        st.session_state[f"sp_expanded_{cid_}"] = True

    # Welches Preset passt zum aktuellen Chat-Prompt?
    current_preset = None
    for preset_name, preset_prompt in prompts.items():
        if preset_prompt == chat["system_prompt"]:
            current_preset = preset_name
            break
    if current_preset is None:
        current_preset = preset_names[0]
    preset_index = preset_names.index(current_preset)

    pc1, pc2 = st.columns([3, 1])
    with pc1:
        preset_pick = st.selectbox(
            "Preset",
            preset_names,
            index=preset_index,
            key=f"preset_pick_{cid}",
            on_change=apply_preset,
            args=(cid,),
        )
    with pc2:
        st.write("")
        if st.button("Preset loeschen", key=f"delete_preset_{cid}", use_container_width=True):
            if preset_pick in DEFAULT_PROMPTS:
                st.warning(f"Das Standard-Preset '{preset_pick}' kann nicht geloescht werden.")
                st.session_state[exp_key] = True
                st.rerun()
            else:
                delete_preset(preset_pick)
                # Wichtig: den alten (jetzt geloeschten) Wert aus dem
                # Dropdown-State entfernen, sonst crasht die Selectbox
                # beim naechsten Rerun ("Wert ist nicht mehr in den Optionen").
                st.session_state.pop(f"preset_pick_{cid}", None)
                st.session_state[exp_key] = True
                st.rerun()

    sp_text = st.text_area(
        "Aktueller System Prompt",
        value=chat["system_prompt"],
        height=150,
        key=f"sp_text_{cid}",
    )

    sc1, sc2, sc3 = st.columns([1, 2, 1])
    with sc1:
        if st.button("Prompt speichern", key=f"save_sp_{cid}", use_container_width=True):
            chat["system_prompt"] = sp_text
            save_chat(chat)
            st.session_state[exp_key] = True
            st.success("System Prompt gespeichert.")
    with sc2:
        new_preset_name = st.text_input(
            "Neuer Preset-Name",
            key=f"preset_name_{cid}",
            label_visibility="collapsed",
            placeholder="Preset-Name zum Speichern",
        )
    with sc3:
        if st.button("Als Preset speichern", key=f"save_preset_{cid}", use_container_width=True):
            preset_name = new_preset_name.strip()
            if not preset_name:
                st.warning("Bitte einen Preset-Namen eingeben.")
                st.session_state[exp_key] = True
            elif preset_name in prompts:
                st.warning(f"Preset '{preset_name}' existiert bereits.")
                st.session_state[exp_key] = True
            else:
                prompts[preset_name] = sp_text
                save_prompts(prompts)
                st.session_state.pop(f"preset_name_{cid}", None)
                st.session_state[exp_key] = True
                st.success(f"Preset '{preset_name}' gespeichert.")
                st.rerun()

# --- Datei-Upload ---
st.divider()
st.subheader("Dateien")

uploaded_files = st.file_uploader(
    "Dateien hochladen",
    type=[
        "txt", "md", "py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h",
        "hpp", "cs", "go", "rs", "php", "rb", "swift", "kt", "kts", "sh",
        "bash", "zsh", "sql", "html", "css", "xml", "yaml", "yml", "json",
        "csv", "log", "ini", "conf", "pdf", "docx",
        "jpg", "jpeg", "png", "gif", "webp",
    ],
    accept_multiple_files=True,
    key=f"file_uploader_{cid}",
)

files_changed = False
if uploaded_files:
    for uploaded_file in uploaded_files:
        already_exists = any(
            f["name"] == uploaded_file.name and f["size"] == len(uploaded_file.getvalue())
            for f in chat.get("files", [])
        )
        if already_exists:
            continue
        try:
            file_info = save_uploaded_file(cid, uploaded_file)
            chat.setdefault("files", []).append(file_info)
            files_changed = True
        except Exception as e:
            st.error(f"Fehler beim Hochladen von '{uploaded_file.name}': {e}")

if files_changed:
    save_chat(chat)
    st.rerun()

if chat.get("files"):
    st.write("Hochgeladene Dateien:")
    for file_info in chat["files"]:
        fc1, fc2 = st.columns([5, 1])
        with fc1:
            size_kb = file_info["size"] / 1024
            ext = Path(file_info["name"]).suffix.lower()
            if ext in IMAGE_EXTENSIONS and Path(file_info["original_path"]).exists():
                st.image(file_info["original_path"], width=120, caption=f"{file_info['name']} ({size_kb:.1f} KB)")
            else:
                st.caption(f"Datei: {file_info['name']} ({size_kb:.1f} KB)")
        with fc2:
            if st.button("Loeschen", key=f"delete_file_{cid}_{file_info['id']}", use_container_width=True):
                try:
                    original_path = Path(file_info["original_path"])
                    text_path = Path(file_info["text_path"])
                    if original_path.exists():
                        original_path.unlink()
                    if text_path.exists():
                        text_path.unlink()
                except OSError as e:
                    st.warning(f"Dateien konnten nicht vollstaendig geloescht werden: {e}")

                chat["files"] = [f for f in chat["files"] if f["id"] != file_info["id"]]
                save_chat(chat)
                st.rerun()

# --- Chatverlauf ---
st.subheader("Chat")
chat_box = st.container(height=500)
with chat_box:
    if not chat["messages"]:
        st.caption("Noch keine Nachrichten. Starte den Chat unten.")
    else:
        for msg in chat["messages"]:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))

# --- Eingabe ---
user_input = st.chat_input("Nachricht...")
if user_input:
    chat["messages"].append({"role": "user", "content": user_input})

    if not API_KEY:
        chat["messages"].append({"role": "assistant", "content": "[Fehler] Kein MAMMOUTH_API_KEY gesetzt."})
    else:
        try:
            with st.spinner("Mammouth denkt..."):
                answer = call_mammouth(
                    model=chat["model"],
                    system_prompt=chat["system_prompt"],
                    messages=chat["messages"],
                    files=chat.get("files", []),
                )
            chat["messages"].append({"role": "assistant", "content": answer})
        except requests.exceptions.Timeout:
            chat["messages"].append({"role": "assistant", "content": "[Fehler] Anfrage an Mammouth ist abgelaufen."})
        except requests.exceptions.RequestException as e:
            chat["messages"].append({"role": "assistant", "content": f"[API-Fehler] {e}"})
        except Exception as e:
            chat["messages"].append({"role": "assistant", "content": f"[Fehler] {e}"})

    save_chat(chat)
    st.rerun()