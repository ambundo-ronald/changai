from langgraph.graph import StateGraph, END
from collections import OrderedDict
from typing_extensions import TypedDict
from typing import Any, Dict, List, Tuple, Union, Optional, Set
import boto3
import requests
import json
from changai.changai.api.v2.non_erp_handler import handle_non_erp_query
import yaml
import re
from frappe.utils.jinja import render_template
import os
import pickle
import numpy as np
import time
import base64
import sqlglot
from functools import lru_cache
from sqlglot import exp
from rapidfuzz import fuzz, process
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types
from changai.changai.api.v2.schema_utils import validate_sql_schema,_load_mapping_data,check_file_updates
from google.oauth2 import service_account
import threading
from werkzeug.wrappers import Response
from changai.changai.api.v2.helpdesk_api import(
    create_helpdesk_ticket,
    get_user_tickets
)
from typing import Any, Dict, Optional
import jinja2
import frappe
from google.api_core import exceptions as google_exceptions
from changai.changai.api.v2.store_chats import (
    save_turn_2,
    inject_prompt,
)
# from changai.changai.api.v2.non_erp_handler import IntelligentStaticResponder
from huggingface_hub import snapshot_download
from frappe.desk.reportview import build_match_conditions
import shutil
from frappe import _
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
# from symspellpy.symspellpy import SymSpell
sym_spell = None
_GEMINI_CLIENT = None
_GEMINI_CONFIG = None
_FIELD_DOCS_CACHE = None
_FIELD_EMBS_CACHE = None
_TABLE_TO_IDX_CACHE = None
_VS_REPORT=None
_KEYWORDS_SET=None
_KEYWORDS_LIST=None
ERPGULF_LINK = "https://app.erpgulf.com/en/products/chang-ai-an-ai-agent"
_ASSETS_DIR = Path(frappe.get_app_path("changai", "changai", "api", "v2", "assets")).resolve()
_PROMPTS_DIR = Path(frappe.get_app_path("changai", "changai", "prompts")).resolve()
CHANGAI_SETTINGS = "ChangAI Settings"
_ALLOWED_EXT = {".json", ".yaml",".j2", ".yml", ".txt", ".md"}
# _warmup_done=False
# def warm_up():
#     global _warmup_done
#     try:
#         import time
#         time.sleep(3)  # wait for frappe to fully initialize first
#         frappe.logger().info("ChangAI: background warmup starting...")
#         load_on_startup()
#         _warmup_done = True
#         frappe.logger().info("ChangAI: background warmup complete ✅")
#     except Exception as e:
#         frappe.logger().error(f"ChangAI: background warmup failed: {e}")


SQL_REWRITE_PROMPT = """You are an ERP query rewriter and entity detector.
Return ONLY valid JSON:
{{"standalone_question":"...","contains_values":true/false,"entity_word":["..."]}}

TASK 1 — FOLLOW-UP
- If the query depends on previous messages, rewrite it as a complete standalone question.
- Otherwise keep it unchanged.

TASK 2 — ENTITY DETECTION
contains_values = TRUE: Any noun that refers to a specific named master record
(item name, customer name, supplier name, warehouse name, employee name)
If not sure, also set contains_values = TRUE, otherwise contains_values = FALSE.
if contains_values = TRUE, then entity_word = the most relevant noun/entity name from the question and place it here and return only if contains_values is TRUE.
if multiple entites are present,place all of them in entity_word list.
TASK 3 — ERP CONTEXTUAL REWRITE

1. Normalize:
- Fix typos, clear English
- Do NOT change entity values

2. Complete intent:
- Never change the question's intent — only fix grammar and map ERP terms.

3. ERP mapping:
- Map generic terms to standard ERPNext concepts based on intent
- Avoid vague words if clearer business terms exist
- Do NOT invent documents or use report names.
Examples:
  stock       → Bin / Stock Ledger Entry
  production  → Work Order
  finance/profit → GL Entry

4. Field hints (max 1–2):
Use natural phrasing ("based on", "using"):
  sales       → grand_total
  qty         → qty
  stock       → actual_qty
  production  → produced_qty
  finance     → debit / credit
  status      → status

5. Time fields:
  Sales/Stock/Finance → posting_date
  Work Order          → actual_start_date / actual_end_date
  Timesheet           → start_date / end_date
  Timesheet Detail    → from_time / to_time
- NEVER use posting_date for Timesheet
- NEVER use creation unless asked

6. Relationships:
- Include linked entities if required

STYLE:
- Natural business language
- No SQL, no tab* names

EXAMPLES:
"total sales amount last month"
→ What is the total sales amount from Sales Invoices last month based on grand_total and posting_date?

"stock in warehouse a"
→ What is the stock quantity in Warehouse A based on actual_qty from Bin?

"who worked today"
→ Which employees logged time today based on Timesheet start_date or Timesheet Detail from_time?

STRICT RULES:
- If the query mentions Draft, Submitted, or Cancelled, explicitly include docstatus in the rewritten question.
- Do not add a specific document type unless clearly implied by the user query or required by standard ERPNext business meaning.
- For vague money questions, clarify the business meaning as actual, ordered, quoted, paid, or outstanding — do not guess the document type incorrectly.
- If the user says "spend", treat it as actual purchase/expense, not quotation or order commitment, unless the user explicitly mentions order, quotation, or planned purchase.
- Preserve all filter conditions, status values, and keywords from the original question — never drop them during rewriting.
- Do NOT add dates, filters, entities, statuses, or assumptions unless explicitly present in the user question or clearly inferred from conversation memory.
- Use chat history only when the current query clearly implies continuation or follow-up context. Never assume dates, filters, entities, or conditions from previous messages unless strongly indicated.
- Use only the most relevant tables and fields required for the user query.
- Use only valid tables and fields from the provided schema context, regardless of retrieval ranking order.
- Choose fields based on business meaning and user intent, not rank position.
- Never invent schema elements.
- Always return any one clear user-readable business field, not only technical IDs, unless explicitly requested.
- If the query is ambiguous, ask for clarification and set "clarify": true."""
# def get_symspell():
#     global sym_spell

#     if sym_spell is not None:
#         frappe.logger().info(f"SymSpell already loaded, skipping PID: {os.getpid()}")
#         return sym_spell

#     frappe.logger().error(f"SymSpell loading NOW in PID: {os.getpid()}") 

#     sym_spell = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)

#     dictionary_path = frappe.get_app_path(
#         "changai",
#         "changai",
#         "api",
#         "v2",
#         "assets",
#         "frequency_dictionary_en_82_765.txt"
#     )

#     sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

#     for kw in BUSINESS_KEYWORDS:
#         sym_spell.create_dictionary_entry(kw.lower(), 1000)

#     return sym_spell

@lru_cache(maxsize=512)
def is_child_table(table: str) -> bool:
    doctype = table.replace("tab", "", 1) if table.startswith("tab") else table

    try:
        meta = frappe.get_meta(doctype, cached=True)
        return bool(getattr(meta, "istable", 0))
    except Exception:
        return False

CHILD_GENERIC_FIELDS = ["parent", "parenttype", "parentfield", "idx"]
MAIN_GENERIC_FIELDS = ["name", "docstatus"]
def enrich_fields_for_sql_context(table: str, fields: list[str]) -> list[str]:
    out = list(fields)

    if is_child_table(table):
        for f in reversed(CHILD_GENERIC_FIELDS):
            if f not in out:
                out.insert(0, f)
    else:
        for f in reversed(MAIN_GENERIC_FIELDS):
            if f not in out:
                out.insert(0, f)

    return out


def format_schema_context(grouped: dict[str, list[str]]) -> str:
    parts = []

    for table, raw_fields in grouped.items():
        child = is_child_table(table)
        fields = enrich_fields_for_sql_context(table, raw_fields)

        parts.append(f"TABLE: {table}")
        parts.append(f"TYPE: {'Child Table' if child else 'Main Table'}")

        if child:
            parts.append("JOIN RULES:")
            parts.append("- parent = parent document name")
            parts.append("- parenttype = parent DocType")
            parts.append("- parentfield = child table fieldname")

        parts.append("FIELDS:")
        for field in fields:
            parts.append(f"- {field}")

        parts.append("")

    return "\n".join(parts)


def publish_pipeline_update(request_id, stage, message, data=None, done=False, error=False):
    if not request_id:
        return
    payload = {
        "request_id": request_id,
        "stage": stage,
        "message": message,
        "data": data or {},
        "done": done,
        "error": error,
        "timestamp": frappe.utils.now_datetime().isoformat(),
    }
    frappe.publish_realtime(
        event=f"debug_{request_id}",
        message=payload,
        user=frappe.session.user,
    )


def _safe_join(base: Path, rel: str) -> Path:
    """
    Prevent path traversal. Only allow reading inside base directory.
    """
    p = (base / rel).resolve()
    if base != p and base not in p.parents:
        frappe.throw(_("Unsafe path: {0}\n"
                       "Check Quick Start Guide Here 👇:\n {1}").format(rel,CHANGAI_GUIDE_LINK))
    return p


def read_asset(file_name: str, base: str = "assets") -> Any:
    """
    base:
      - "assets"  -> changai/changai/api/v2/assets
      - "prompts" -> changai/changai/prompts
    """
    file_name = (file_name or "").strip()
    if not file_name:
        frappe.throw(_("file_name is required\n"
                       "Check Quick Start Guide Here 👇:\n {0}").format(CHANGAI_GUIDE_LINK))

    ext = Path(file_name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        frappe.throw(_("Unsupported file type: {0}\n"
                       "Check Quick Start Guide Here 👇:\n {1}").format(ext, CHANGAI_GUIDE_LINK))

    if base == "assets":
        root = _ASSETS_DIR
    elif base == "prompts":
        root = _PROMPTS_DIR
    else:
        root = None
    if root is None:
        frappe.throw(_("Invalid base: {0}\n"
                       "Check Quick Start Guide Here 👇:\n {1}").format(base, CHANGAI_GUIDE_LINK))
    # nosemgrep: frappe-semgrep-rules.rules.security.frappe-security-file-traversal
    path = _safe_join(root, file_name)

    if not path.is_file():
        frappe.throw(_("File not found: {0}\n"
                       "Check Quick Start Guide Here 👇:\n {1}").format(str(path), CHANGAI_GUIDE_LINK))

    content = path.read_text(encoding="utf-8", errors="replace")

    if ext == ".json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            frappe.throw(_("Invalid JSON in {0}: {1}"
                           "Check Quick Start Guide Here 👇:\n {2}").format(str(path), str(e), CHANGAI_GUIDE_LINK))
    if ext == ".yaml" or ext == ".yml":
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            frappe.throw(_("Invalid YAML in {0}: {1}"
                           "Check Quick Start Guide Here 👇:\n {2}").format(str(path), str(e), CHANGAI_GUIDE_LINK))
    return content

_VS_TABLE = None
_VS_MASTER = None
_EMBEDDER_INSTANCE = None
_FULL_FIELDS_VS = None
STATUS_200 = 200
_SUB_VS_CACHE = {}
APPLICATION_JSON = "application/json"
CHANGAI_GUIDE_LINK="https://app.erpgulf.com/en/articles/chang-ai-quick-start-guide"
EMBEDDING_ENGINE_NONE_MESSG = f"""
Embedding engine is None. Model not loaded.
Check Quick Start Guide Here 👇:
{CHANGAI_GUIDE_LINK}"""
MODEL_ID = "gemini-2.5-flash-lite"
RETRY_LIMIT = 2
REPORT_INTENT_MAP  = read_asset("report_intent_map.json",base="assets")
bk = read_asset("business_keywords_v1.json", base="assets")
BUSINESS_KEYWORDS = bk.get("business_keywords", bk)

mapping_data = read_asset("metaschema_clean_v2.json", base="assets")
CONVERSATION_TEMPLATE = read_asset("conversation_template_v2.j2", base="assets")
SQL_SYS_PROMPT = read_asset("sql_system_prompt.txt", base="prompts")
SQL_PROMPT = read_asset("sql_user_prompt.txt", base="prompts")
FORMAT_PROMPT = read_asset("user_friendly_prompt.txt", base="prompts")
NON_ERP_PROMPT = read_asset("non_erp_prompt.txt", base="prompts")
SUPPORT_PROMPT = read_asset("support.txt", base="prompts")
SUPPORT_USER_PROMPT = read_asset("support_user_prompt.txt", base="prompts")
SUPPORT_SYS_PROMPT = read_asset("support_sys_prompt.txt", base="prompts")

FILTER_TABLES = read_asset("filter_tables.txt", base="prompts")
filter_fields = read_asset("filter_fields.txt", base="prompts")

@frappe.whitelist(allow_guest=False)
def download_model():
    frappe.enqueue(
        "changai.changai.api.v2.text2sql_pipeline_v2.download_model_from_ui",  # dot-path to the function
        queue="long",           # use "long" queue for heavy tasks
        timeout=3600,           # 1 hour timeout (in seconds)
        is_async=True,          # run in background (default True)
        job_name="download_model",  # optional: helps track/deduplicate jobs
    )
    return {
        "ok":True,"message":"Model Downloading.."
    }

def _get_model_path():
    site_path = frappe.get_site_path("private", "files", "changai_model")
    return site_path


@frappe.whitelist(allow_guest=False)
def download_model_from_ui():
    global _EMBEDDER_INSTANCE

    model_path = _get_model_path()

    try:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        os.makedirs(model_path, exist_ok=True)

        snapshot_download(
            repo_id="hyrinmansoor/changAI-nomic-embed-text-v1.5-finetuned",
            local_dir=model_path,
            ignore_patterns=[
        "*.pt",
        "*.pth",
        "*.bin",
        "trainer_*",
        "optimizer*"
    ]
        )
        _EMBEDDER_INSTANCE = None
        return {"status": "success", "message": "Embedding model downloaded successfully."}

    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Embedding Model Download Failed")
        frappe.throw(_("Model download failed: {0}\n Check Quick Start Guide Here 👇:\n{1} <br>" 
        "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Download Embedding Model</a></b>.<br>"
        "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>."
).format(str(e),CHANGAI_GUIDE_LINK,settingsUrl,ERPGULF_LINK))


_FIELD_DOCS_CACHE = None
_FIELD_EMBS_CACHE = None
_TABLE_TO_IDX_CACHE = None


def load_field_matrix():
    global _FIELD_DOCS_CACHE, _FIELD_EMBS_CACHE, _TABLE_TO_IDX_CACHE

    if _FIELD_DOCS_CACHE is not None:
        return _FIELD_DOCS_CACHE, _FIELD_EMBS_CACHE, _TABLE_TO_IDX_CACHE

    app_root = Path(frappe.get_app_path("changai")).resolve()
    schema_rel = "changai/api/v2/fvs_stores/erpnext/emb_dir"
    # nosemgrep: frappe-semgrep-rules.rules.security.frappe-security-file-traversal
    schema_path = _safe_join(app_root, schema_rel)

    embs_path = schema_path / "field_embs.npy"
    docs_path = schema_path / "field_docs.pkl"
    table_idx_path = schema_path / "table_to_idx.pkl"

    if not embs_path.exists():
        frappe.throw(f"Missing field_embs.npy. Rebuild schema FVS first: {embs_path}")

    # nosemgrep: frappe-semgrep-rules.rules.security.frappe-security-file-traversal
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)

    # nosemgrep: frappe-semgrep-rules.rules.security.frappe-security-file-traversal
    with open(table_idx_path, "rb") as f:
        table_to_idx = pickle.load(f)

    embs = np.load(embs_path, mmap_mode="r")

    _FIELD_DOCS_CACHE = docs
    _FIELD_EMBS_CACHE = embs
    _TABLE_TO_IDX_CACHE = table_to_idx

    return docs, embs, table_to_idx

def _get_cached_embedding_test(q: str) -> tuple:
    t0=time.time()
    emb = get_embedding_engine()
    emb_load_time = time.time() - t0
    t1 = time.time()
    vec = emb.embed_query(q)
    embed_query_time = time.time() - t1
    return emb_load_time,embed_query_time # tuple for hashability


def get_embedding_engine():
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is not None:
        return _EMBEDDER_INSTANCE
    
    model_path = _get_model_path()  # check path first, always
    
    if not os.path.exists(model_path):
        _EMBEDDER_INSTANCE = None  # reset if model missing
        frappe.throw(
            _(
                "Go to <b>ChangAI Settings</b> and click"
                "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Download Embedding Model</a></b>.<br><br>"
                "Check this Quick Start Guide for more detail: "
                "<a href='{0}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a>"
                "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>."

            ).format(CHANGAI_GUIDE_LINK,settingsUrl,ERPGULF_LINK),
            title=_("Embedding Model Required")
        )
    
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu","trust_remote_code": True,},
            encode_kwargs={
        "normalize_embeddings": True,
    },
        )
    
    return _EMBEDDER_INSTANCE

def _build_frontend_settings_config() -> Dict[str, Any]:
    settings = frappe.get_single(CHANGAI_SETTINGS)

    aws_access_key_id = (getattr(settings, "aws_access_key_id", None) or "").strip()
    aws_secret_access_key = (getattr(settings, "aws_secret_access_key", None) or "").strip()
    aws_region = (
        getattr(settings, "aws_region", None)
        or getattr(settings, "aws_default_region", None)
        or "us-east-1"
    )

    return {
        "RETAIN_MEM": settings.retain_memory,
        "LLM_VERSION_ID": settings.llm_version_id,
        "EMBED_VERSION_ID": settings.embedder_version_id,
        "REMOTE": bool(settings.remote),
        "deploy_url": settings.deploy_url,
        "entity_retriever": settings.entity_retriever,
        "support_api_url": settings.support_url,
        "get_ticket_details_url": settings.get_ticket_details_url,
        "llm": settings.llm,
        "location": settings.gemini_location,
        "retriever_structure": settings.retriever_structure,
        "gemini_project_id": settings.gemini_project_id,
        "gemini_json_content": settings.gemini_json_content,
        "enable_voice_chat": bool(settings.enable_voice_chat),
        "aws_region": aws_region,
        "polly_voice_id": "Zayd",
        "polly_enabled": bool(settings.enable_voice_chat and aws_access_key_id and aws_secret_access_key),
        "enable_changai": bool(settings.enable_changai)
    }


@frappe.whitelist(allow_guest=False)
def get_settings() -> Dict[str, Any]:
    settings = frappe.get_single(CHANGAI_SETTINGS)
    config = {
        "RETAIN_MEM": settings.retain_memory,
        "LLM_VERSION_ID": settings.llm_version_id,
        "EMBED_VERSION_ID": settings.embedder_version_id,
        "API_TOKEN": settings.api_token,
        "REMOTE": bool(settings.remote),
        "deploy_url": settings.deploy_url,
        "entity_retriever": settings.entity_retriever,
        "support_api_url": settings.support_url,
        "get_ticket_details_url": settings.get_ticket_details_url,
        "llm": settings.llm,
        "location": settings.gemini_location,
        "retriever_structure": settings.retriever_structure,
        "gemini_project_id": settings.gemini_project_id,
        "gemini_json_content": settings.gemini_json_content,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
        "enable_voice_chat": settings.enable_voice_chat,
    }
    return config


@frappe.whitelist(allow_guest=False)
def get_frontend_settings() -> Dict[str, Any]:
    return _build_frontend_settings_config()

class ChangAIConfig:
    @classmethod
    def get(cls):
        if not hasattr(frappe.local, "_changai_config"):
            frappe.clear_document_cache(CHANGAI_SETTINGS)
            frappe.local._changai_config = get_settings()
        return frappe.local._changai_config
_POLLY_CLIENT = None

def get_polly_client(config):
    global _POLLY_CLIENT

    if _POLLY_CLIENT is None:
        _POLLY_CLIENT = boto3.client(
            "polly",
            aws_access_key_id=(config.get("aws_access_key_id") or "").strip(),
            aws_secret_access_key=(config.get("aws_secret_access_key") or "").strip(),
            region_name=(config.get("aws_region") or "us-east-1"),
        )
    return _POLLY_CLIENT

def build_ssml(text: str) -> str:
    parts = []
    current = []
    current_lang = None

    for token in text.split():
        lang = "ar-AE" if re.search(r'[\u0600-\u06FF]', token) else "en-US"

        if current_lang is None:
            current_lang = lang

        if lang != current_lang:
            parts.append(
                f'<lang xml:lang="{current_lang}">{" ".join(current)}</lang>'
            )
            current = [token]
            current_lang = lang
        else:
            current.append(token)

    if current:
        parts.append(
            f'<lang xml:lang="{current_lang}">{" ".join(current)}</lang>'
        )

    return "<speak>" + " ".join(parts) + "</speak>"
@frappe.whitelist(allow_guest=False)
def synthesize_tts(text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
    config = ChangAIConfig.get()
    if not bool(config.get("enable_voice_chat")):
        return {"ok": False, "error": "Voice chat is disabled in settings.", "provider": "browser"}
    aws_access_key_id = (config.get("aws_access_key_id") or "").strip()
    aws_secret_access_key = (config.get("aws_secret_access_key") or "").strip()
    if not aws_access_key_id or not aws_secret_access_key:
        return {"ok": False, "error": "AWS Polly credentials are missing.", "provider": "browser"}
    cleaned_text = re.sub(r"<[^>]*>", " ", text or "")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    if not cleaned_text:
        return {"ok": False, "error": "Text is empty.", "provider": "browser"}

    if len(cleaned_text) > 2500:
        cleaned_text = cleaned_text[:2500]

    try:
        polly_client = get_polly_client(config)
        voice = (voice_id or config.get("polly_voice_id") or "Zayd").strip() or "Zayd"
        ssml_text = build_ssml(cleaned_text)
        response = polly_client.synthesize_speech(
    Text=ssml_text,
    OutputFormat="mp3",
    VoiceId="Zayd",
    Engine="neural",
    TextType="ssml",
)
        stream = response.get("AudioStream")
        if stream is None:
            return {"ok": False, "error": "Polly did not return audio stream.", "provider": "browser"}

        audio_bytes = stream.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "ok": True,
            "provider": "polly",
            "mime_type": "audio/mpeg",
            "audio_base64": audio_base64,
            "voice_id": voice,
        }
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "ChangAI Polly TTS Error")
        return {"ok": False, "error": str(e), "provider": "browser"}


@frappe.whitelist(allow_guest=True)  # nosemgrep: security.guest-whitelisted-method - intentional, validates credentials via OAuth client lookup and Frappe password grant before returning a token
def generate_token_secure(api_key: str, api_secret: str, app_key: str):
    try:
        try:
            app_key = base64.b64decode(app_key).decode("utf-8")
        except Exception:
            return Response(
                json.dumps(
                    {"message": "Security Parameters are not valid", "user_count": 0}
                ),
                status=401,
                mimetype=APPLICATION_JSON,
            )

        doc = frappe.db.get_value(
            "OAuth Client",
            {"app_name": app_key},
            ["name", "client_id", "client_secret", "user"],
            as_dict=True
        )
        if not doc:
            frappe.local.response["http_status_code"] = 401
            return {"ok": False, "error": "OAuth client not found / invalid app_key"}
        if doc.client_id is None:
            return Response(
                json.dumps(
                    {"message": "Security Parameters are not valid", "user_count": 0}
                ),
                status=401,
                mimetype=APPLICATION_JSON,
            )
        url = (
            frappe.local.conf.host_name
            + "/api/method/frappe.integrations.oauth2.get_token"
        )
        payload = {
            "username": api_key,
            "password": api_secret,
            "grant_type": "password",
            "client_id": doc.client_id,
            "client_secret": doc.client_secret,
        }
        response = requests.request("POST", url, data=payload)
        if response.status_code == STATUS_200:
            result_data = json.loads(response.text)
            return Response(
                json.dumps({"data": result_data}),
                status=STATUS_200,
                mimetype=APPLICATION_JSON,
            )
        else:
            frappe.local.response.http_status_code = 401
            return json.loads(response.text)
    except Exception as e:
        return Response(
            json.dumps({"message":str(e), "user_count": 0}),
            status=500,
            mimetype=APPLICATION_JSON,
        )


# Api for  checking user name  using token
@frappe.whitelist(allow_guest=False)
def whoami() -> Dict[str, Any]:
    """This function returns the current session user"""
    try:
        response_content = {
            "user": frappe.session.user,
        }
        frappe.local.response = {
            "data": response_content,
            "http_status_code": STATUS_200,
        }
        return Response(
            json.dumps({"data": response_content}),
            status=STATUS_200,
            mimetype=APPLICATION_JSON,
        )
    except ValueError as ve:
        frappe.throw(_("{0}\n Check Quick Start Guide Here 👇:\n {1}").format(str(ve),CHANGAI_GUIDE_LINK))
                     


def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract all table names from a SQL query."""
    if not sql:
        return []
    matches = re.findall(r'`(tab[^`]+)`', sql, re.IGNORECASE)
    seen = set()
    tables = []
    for t in matches:
        if t not in seen:
            seen.add(t)
            tables.append(t)
    return tables


def call_model(prompt: str, task: str = "llm",sys_prompt: str = "") -> Any:
    config = ChangAIConfig.get()
    if config["REMOTE"] and config["llm"] == "QWEN3":
        return remote_llm_request_deploy_test(prompt=prompt, task=task)
    else:
        if config["llm"] == "Gemini":
            return call_gemini(prompt,sys_prompt)


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 120):
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        ct = (res.headers.get("Content-Type") or "").lower()
        try:
            body = res.json() if APPLICATION_JSON in ct else {"raw_text": res.text}
        except Exception:
            body = {"raw_text": res.text}
        if res.status_code not in (STATUS_200, 201, 202):
            return {"ok": False, "status_code": res.status_code, "body": body}
        return {"ok": True, "status_code": res.status_code, "body": body}
    except requests.exceptions.Timeout:
        return {"ok": False, "status_code": None, "body": {"error": "timeout"}}
    except Exception as e:
        return {"ok": False, "status_code": None, "body": {"error": str(e)}}


def local_llm_request(prompt: str) -> str:
    config = ChangAIConfig.get()
    url = f"{config['URL'].rstrip('/')}/api/generate"
    payload = {"model": config["LOCAL_LLM"], "prompt": prompt, "stream": False}
    resp = _post_json(url, headers={}, payload=payload, timeout=120)
    if not resp.get("ok"):
        return f"Error: local LLM call failed ({resp.get('status_code')}): {resp.get('body')}"
    text = (resp.get("body") or {}).get("response")
    return (text or "").strip() or "Error: Empty response from local LLM."


def _get_gemini_vertex_config(config):
    project_id = (config.get("gemini_project_id") or "").strip()
    credentials_json = (config.get("gemini_json_content") or "").strip()
    location = (config.get("gemini_location") or "").strip()
    return project_id, credentials_json, location


def _throw_missing_vertex_field(project_id: str, location: str, credentials_json: str) -> None:
    if not project_id:
        frappe.throw(
            _("Gemini Project ID is missing.<br><br>Please <b> <a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Go to Settings Page</a> </b> and enter your <b>Gemini Project ID</b>.<br>"
            "Check Quick Start Guide 👇:<br><a href='{0}' target='_blank'>Click here</a><br>"
            "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>.").format(CHANGAI_GUIDE_LINK,settingsUrl,ERPGULF_LINK),
            title=_("Missing Gemini Project ID"),
        )
    if not location:
        frappe.throw(
            _("Gemini Location is missing.<br><br>Please <b><a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Go to Settings Page</a></b> and enter your <b>Gemini Location</b>.<br>"
              "Check Quick Start Guide 👇:<br><a href='{0}' target='_blank'>Click here</a><br>"
              "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>.").format(CHANGAI_GUIDE_LINK,settingsUrl,ERPGULF_LINK),
            title=_("Missing Gemini Location"),
        )
    if not credentials_json:
        frappe.throw(
            _("Service Account Credentials are missing.<br><br>Please <b><a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Go to Settings Page</a></b> and enter your <b>Service Account Credential</b>.<br>"
            "Check Quick Start Guide 👇:<br><a href='{0}' target='_blank'>Click here</a>"
            "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>."
).format(CHANGAI_GUIDE_LINK,settingsUrl,ERPGULF_LINK),
            title=_("Missing Service Account Credentials"),
        )


def _build_vertex_gemini_client(project_id: str, location: str, credentials_json: str):
    _throw_missing_vertex_field(project_id, location, credentials_json)

    service_account_info = json.loads(credentials_json)
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=creds,
    )


def _get_api_key_client(config):
    try:
        api_key = config.get("gemini_api_key")
    except Exception:
        api_key = None

    if not api_key:
        frappe.throw(
            _(
                "Gemini API key is not configured.<br><br>"
                "You have two options to authenticate with Gemini:<br><br>"
                "<b>Option 1 (Free / API Key):</b><br>"
                "Go to <b>ChangAI Settings</b> and enter your <b>Gemini API Key</b>.<br>"
                "Get your free API key from "
                "<a href='https://aistudio.google.com/app/apikey' target='_blank'>Google AI Studio</a>.<br><br>"
                "<b>Option 2 (Vertex AI / Service Account):</b><br>"
                "Fill in <b>Gemini Project ID</b>, <b>Gemini Location</b>, "
                "and <b>Service Account Credentials</b> in <b><a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Got to Settings Page</a></b>.<br>"
                "ChangAI Quick Start Guide 👇:<br>"
                "<a href='{0}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
                "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>."

            ).format(CHANGAI_GUIDE_LINK,settingsUrl,ERPGULF_LINK),
            title=_("Gemini Authentication Not Configured"),
        )

    return genai.Client(api_key=api_key)


def _build_gemini_client(config):
    project_id, credentials_json, location = _get_gemini_vertex_config(config)

    if project_id or credentials_json or location:
        return _build_vertex_gemini_client(project_id, location, credentials_json)

    return _get_api_key_client(config)


def _build_gemini_contents(prompt: str):
    return [
        {
            "role": "user",
            "parts": [{"text": str(prompt)}],
        }
    ]


def _clean_gemini_response_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return text


def _handle_gemini_api_exception(e: Exception) -> None:
    if isinstance(e, google_exceptions.ResourceExhausted):
        frappe.throw(
            _("Gemini API quota exceeded.<br><br>Please wait and try again or upgrade your plan.<br>Check Quick Start Guide 👇:<br>"
            "<a href='{0}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
            "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>."
).format(CHANGAI_GUIDE_LINK,ERPGULF_LINK),

            title=_("Gemini Quota Exceeded"),
        )
    if isinstance(e, google_exceptions.Unauthenticated):
        frappe.throw(
            _("Gemini API key is invalid.<br><br>Please go to <b>ChangAI Settings</b> and enter a valid <b>Gemini API Key</b>.<br>"
            "Check ChangAI Quick Start Guide 👇:<br><a href='{0}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
            "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>."
).format(CHANGAI_GUIDE_LINK,ERPGULF_LINK),
            title=_("Invalid Gemini API Key"),
        )
    if isinstance(e, google_exceptions.PermissionDenied):
        frappe.throw(
            _("Gemini API permission denied.<br><br>Please check your API key permissions.<br>"
            "Check ChangAI Quick Start Guide 👇:<br><a href='{0}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
            "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>."
).format(CHANGAI_GUIDE_LINK,ERPGULF_LINK),
            title=_("Gemini Permission Denied"),
        )
    if isinstance(e, google_exceptions.InvalidArgument):
        frappe.throw(
            _("Invalid request to Gemini API: {0}<br>"
            "Check ChangAI Quick Start Guide 👇:<br>"
            "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
            "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>.").format(str(e),CHANGAI_GUIDE_LINK,ERPGULF_LINK),
            title=_("Gemini Invalid Request"),
        )

    frappe.log_error(frappe.get_traceback(), "Gemini API Unexpected Error")
    frappe.throw(
        _("Gemini API error: {0}<br>"
        "Check ChangAI Quick Start Guide 👇:<br>"
        "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
        "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>ERPGulf.com</a></b>.").format(str(e),CHANGAI_GUIDE_LINK,ERPGULF_LINK),
        title=_("Gemini API Error"),
    )


def gemini_client():
    global _GEMINI_CLIENT,_GEMINI_CONFIG
    if _GEMINI_CLIENT is None:
        config = frappe.get_single(CHANGAI_SETTINGS)
        _GEMINI_CONFIG =  config
        _GEMINI_CLIENT = _build_gemini_client(config)
    return _GEMINI_CLIENT

def call_gemini(prompt: str,sys_prompt: str) -> Union[str, Dict[str, Any]]:
    try:
        # frappe.clear_document_cache(CHANGAI_SETTINGS)
        client = gemini_client()

        gemini_config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
        )
        response = client.models.generate_content(
            model=MODEL_ID,
            config=gemini_config,
            contents=_build_gemini_contents(prompt),
        )
        return _clean_gemini_response_text(response.text)

    except frappe.exceptions.ValidationError:
        raise
    except Exception as e:
        _handle_gemini_api_exception(e)


def _build_input_payload(task: str, prompt: str, question: Optional[str],
                          db_result_json: Optional[str], user_message: Optional[str]) -> Dict[str, Any]:
    if task == "format_db":
        return {"task": "format_db", "question": question or "", "db_result_json": db_result_json or "{}"}
    if task == "helpdesk_task":
        return {"task": "helpdesk_task", "user_message": user_message or prompt or ""}
    return {"task": "llm", "user_input": prompt}


def _poll_until_done(get_url: str, headers: Dict) -> Any:
    terminal = {"succeeded", "failed", "canceled"}
    deadline = time.time() + 300
    last = None
    while time.time() < deadline:
        try:
            poll = requests.get(get_url, headers=headers, timeout=120).json()
        except Exception as e:
            poll = {"raw_text": str(e)}
        last = poll
        status = poll.get("status")
        if status in terminal:
            if status == "succeeded":
                return poll.get("output")
            return {"Error": f"Model ended with status {status}", "details": poll}
        time.sleep(2)
    return {"Error": "Polling timed out", "details": last}


def remote_llm_request_deploy_test(
    prompt: str = "",
    task: str = "llm",
    question: Optional[str] = None,
    db_result_json: Optional[str] = None,
    user_message: Optional[str] = None,
) -> Any:
    config = ChangAIConfig.get()
    headers = {
        "Content-Type": APPLICATION_JSON,
        "Prefer": "wait",
        "Authorization": f"Bearer {config['API_TOKEN']}",
    }
    input_payload = _build_input_payload(task, prompt, question, db_result_json, user_message)
    create = _post_json(config["deploy_url"], headers=headers, payload={"input": input_payload}, timeout=120)

    if not create.get("ok"):
        return {"Error": "Create prediction failed", "status_code": create.get("status_code"), "details": create.get("body")}

    get_url = ((create.get("body") or {}).get("urls") or {}).get("get")
    if not get_url:
        return {"Error": "Missing get URL from deploy response", "details": create.get("body")}

    return _poll_until_done(get_url, headers)


def remote_embedder_request(formatted_q: str) -> Union[List[Any], str]:
    config = ChangAIConfig.get()
    payload = {"version": config["EMBED_VERSION_ID"], "input": {"user_input": formatted_q}}
    headers = {
        "Content-Type": APPLICATION_JSON,
        "Prefer": "wait",
        "Authorization": f"Bearer {config['API_TOKEN']}",
    }
    response = _post_json(config["URL"], headers, payload)
    try:
        if response:
            return response["body"]["output"]
    except Exception as e:
        return "Error: " + str(e)


def _safe_strip(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v).strip()


# Shared State
class SQLState(TypedDict, total=False):
    filters:str
    entity_type_list:List[str]
    create_entity:bool
    open_report:bool
    report_name:str
    entity_name:str
    doc:str
    reports_filter_before_call:list
    entity_type:str
    final_prompt:str
    request_id: str
    sendNonErptoAI:bool
    session_id: str
    question: str
    contains_values: bool
    entity_words:List[str]
    formatted_q: str
    hits: List[Any]
    context: str
    sql: str
    orm: str
    validation: Dict[str, Any]
    error: Optional[str]
    tries: int
    query_type: str
    sql_prompt: str
    formatting_prompt: str
    non_erp_res: str
    entity_cards: List[str]
    entity_raw: Any
    retrieval_mode: str
    top_tables: List[str]
    selected_tables: List[str]
    top_fields: Dict[str, Any]
    selected_fields: str
    message:str
    stop_followup:bool

def route_action(state: SQLState) -> str:
    if state.get("stop_followup"):
        return "STOP_FOLLOW"
    if state.get("create_entity"):
        return "CREATE_ENTITY"

    if state.get("open_report"):
        return "OPEN_REPORT"

    return "CONTINUE"


def fill_sql_prompt(question: str, context: str) -> str:
    return SQL_PROMPT.format(question=question, context=context)


@lru_cache(maxsize=None)
def _word_is_erp(word: str) -> bool:
    if len(word) <= 3:
        return False
    if word in _KEYWORDS_SET:
        return True
    for kw in _KEYWORDS_SET:
        if word in kw or kw in word:
            return True
    if len(word) >= 4:
        match = process.extractOne(
            word, _KEYWORDS_LIST, scorer=fuzz.ratio, score_cutoff=70
        )
        if match:
            return True
    return False


STOP_WORDS = {
    # English greetings / casual
    "hi", "hello", "hey", "thanks", "thank", "please", "pls",
    "ok", "okay", "yes", "no", "bye", "goodbye","have","has","had","do","does","did",

    # English question/helper words
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "can", "could", "would", "should", "do", "does", "did", "is", "are",
    "was", "were", "be", "been", "being",

    # English filler/common words
    "the", "a", "an", "to", "for", "from", "of", "in", "on", "at", "by",
    "with", "without", "and", "or", "but", "if", "then", "than", "as",
    "this", "that", "these", "those", "it", "its", "there", "here",

    # English user command words
    "show", "list", "give", "get", "find", "display", "tell", "me",
    "need", "want", "make", "create", "check", "see", "view",

    # English time/common filters
    "today", "yesterday", "tomorrow", "now", "current", "latest",
    "last", "next", "this", "week", "month", "year", "daily", "weekly",
    "monthly", "yearly",

    # Arabic greetings / casual
    "مرحبا", "مرحبًا", "اهلا", "أهلا", "أهلًا", "السلام", "شكرا", "شكرًا",
    "نعم", "لا", "طيب", "تمام", "مع السلامة",

    # Arabic question/helper words
    "ما", "ماذا", "من", "متى", "أين", "اين", "كيف", "لماذا", "هل", "كم",
    "أي", "اي", "الذي", "التي", "الذين",

    # Arabic filler/common words
    "في", "من", "إلى", "الى", "على", "عن", "مع", "بدون", "و", "أو", "او",
    "لكن", "إذا", "اذا", "ثم", "هذا", "هذه", "هؤلاء", "ذلك", "تلك", "هنا",

    # Arabic user command words
    "اعرض", "عرض", "اظهر", "أظهر", "هات", "اعطني", "أعطني", "اريد", "أريد",
    "احتاج", "ابحث", "تحقق", "شوف",

    # Arabic time/common filters
    "اليوم", "أمس", "امس", "غدا", "غدًا", "الآن", "الان", "الحالي",
    "الأخير", "الاخير", "هذا", "هذه", "الأسبوع", "الاسبوع", "الشهر",
    "السنة", "العام", "يومي", "أسبوعي", "شهري", "سنوي",
}


def tokenize_mixed(text):
    return re.findall(r'[\u0600-\u06FF]+|[a-zA-Z0-9]+', text.lower())


def is_erp_query(master_match:bool, q: str, words_list: list,cut_off_perc:int) -> bool:
    if master_match:
        match = process.extract(
            q.strip().lower(),
            [v.strip().lower() for v in words_list],
            scorer=fuzz.WRatio,
            limit=5        )
        # matched_value = match[
        return {
            "matched_value": match,
        }

    words = tokenize_mixed(q)

    for word in words:

        if words_list != THREAD_WORDS:
            if word in STOP_WORDS:
                continue
            if len(word) <= 2:
                continue

        match = process.extractOne(
            word,
            words_list,
            scorer=fuzz.ratio,
            score_cutoff=cut_off_perc
        )

        if match:
            return True

    return False


def guardrail_router(state: SQLState) -> SQLState:
    request_id = state.get("request_id")
    chat_id = state.get("session_id")
    raw_q = state.get("question") or ""
    try:
        is_erp= is_erp_query(False,raw_q,BUSINESS_KEYWORDS,80)
        if is_erp:
            query_type = "ERP"
        elif is_thread_erp(raw_q, chat_id):
            query_type = "ERP"
        else:
            query_type = "NON_ERP"
    except Exception as e:
        query_type = "NON_ERP"
        frappe.log_error(frappe.get_traceback(), "Guardrail Router Error")
        return {**state, "query_type": query_type, "error": f"Error in guardrail router: {str(e)}"}
    
    state["query_type"] = query_type
    publish_pipeline_update(
            request_id,
            "question_classify_done",
            "Query classified as " + query_type,
            data={"query_type": query_type}
        )
    return state


def send_non_erp_request(state: SQLState) -> SQLState:
    qstn =state.get("question")
    if not qstn:
        return {**state, "non_erp_res": "", "error": "No question provided"}
    # prompt = NON_ERP_PROMPT.format(question=qstn)
    try:
        # response = handle_non_erp_query(qstn)
        response = non_erp_response(qstn)
        # response = call_model(prompt, "llm")
        if not response or not response.get("data"):
            return {**state,"non_erp_res": "", "error": str(response)}
        return {**state,"non_erp_res": response["data"], "error": None}
    except frappe.exceptions.ValidationError:
        raise
    except Exception as e:
        return {**state, "non_erp_res": "", "error": f"NON-ERP call failed: {e}"}


def _parse_rewrite_response(raw: Any, user_qstn: str) -> Tuple[str, bool]:
    standalone = ""
    contains_values = False
    obj = None
    doc = None
    entity_words = []
    stop_followup =False
    entity_name = None
    report_name = None
    open_report = False
    create_entity = False
    report_intent = None
    message=None

    if isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        try:
            obj = json.loads(raw.strip())
        except Exception:
            standalone = raw.strip()
    else:
        standalone = str(raw).strip()

    if isinstance(obj, dict):
        standalone = (obj.get("standalone_question") or "").strip() or standalone
        contains_values = bool(obj.get("contains_values"))
        entity_words = obj.get("entity_words") or [] if contains_values else []
        create_entity = bool(obj.get("create_entity"))
        create_entity = obj.get("create_entity") if create_entity else False
        if create_entity :
            doc = obj.get("doctype")
            entity_name = obj.get("entity_name")
        open_report = bool(obj.get("open_report"))
        if open_report:
            report_name = obj.get("report_name")
            report_intent = obj.get("report_intent")
        stop_followup = bool(obj.get("stop_followup"))
        if stop_followup:
            message = obj.get("message")

    elif isinstance(obj, list) and not standalone:
        standalone = json.dumps(obj)

    return standalone or user_qstn.strip(), contains_values, entity_words,create_entity, doc,entity_name,report_name,open_report,report_intent,stop_followup,message

SQL_REWRITE_SYS_PROMPT = read_asset("sql_rewrite_sys_prompt.txt", base="prompts")
SQL_REWRITE_USER_PROMPT = read_asset("sql_rewrite_user_prompt.txt", base="prompts")
def rewrite_question(state: SQLState) -> SQLState:
    report_intent = None
    request_id = state.get("request_id")
    user_qstn = state.get("question") or ""
    session_id = state.get("session_id")
    entity_words = []
    sys_prompt = SQL_REWRITE_SYS_PROMPT
    prompt = inject_prompt(user_qstn, session_id)
    report_name_new=None
    try:
        raw = call_model(prompt, "llm",sys_prompt)
        standalone, contains_values,entity_words,create_entity, doc,entity_name,report_name,open_report,report_intent,stop_followup,message = _parse_rewrite_response(raw, user_qstn)
        if report_intent:
            report_name_new = match_report_intent(report_intent)


        publish_pipeline_update(
            request_id,
            "question_rewrite_done",
            "Question rewritten",
            data={"formatted_q": standalone}
        )

        return {
            **state,
            "report_name":report_name_new or report_name or "",
            "report_intent": report_intent,
            "open_report":open_report,
            "create_entity":create_entity,
            "entity_name":entity_name if create_entity else None,
            "doc":doc if create_entity else None,
            "formatted_q": standalone,
            "contains_values": contains_values,
            "entity_words": entity_words,
            "formatting_prompt": prompt,
            "error": None,
            "message":message if stop_followup else None,
            "stop_followup": stop_followup
        }
    
    except frappe.exceptions.ValidationError:
        raise

    except Exception as e:
        publish_pipeline_update(
            request_id,
            "failed",
            str(e),
            error=True,
            done=True
        )
        return {**state, "error": str(e)}


def get_vs(istable: bool):
    global _VS_TABLE, _VS_REPORT

    emb = get_embedding_engine()
    if emb is None:
        frappe.throw(_(EMBEDDING_ENGINE_NONE_MESSG))

    app_path = frappe.get_app_path("changai")

    if istable:
        if _VS_TABLE is None:
            table_vs_path = os.path.join(
                app_path, "changai", "api", "v2",
                "fvs_stores", "erpnext", "table_fvs"
            )

            if not os.path.exists(table_vs_path):
                frappe.throw(_("FAISS table store not found at {0}").format(table_vs_path))

            _VS_TABLE = FAISS.load_local(
                table_vs_path,
                emb,
                allow_dangerous_deserialization=True
            )

        return _VS_TABLE

    else:
        if _VS_REPORT is None:
            report_vs_path = os.path.join(
                app_path, "changai", "api", "v2",
                "fvs_stores", "erpnext", "report_fvs"
            )

            if not os.path.exists(report_vs_path):
                frappe.throw(_("FAISS report store not found at {0}").format(report_vs_path))

            _VS_REPORT = FAISS.load_local(
                report_vs_path,
                emb,
                allow_dangerous_deserialization=True
            )

        return _VS_REPORT



# def call_fvs_table_search(q: str) -> List[str]:
#     hits = get_table_vs().similarity_search(q, k=20)
#     out, seen = [], set()
#     for h in hits:
#         t = h.metadata.get("table")
#         if t and t not in seen:
#             seen.add(t)
#             out.append(t)
#     return out

def check_memory_status() -> dict:
    return {
        "pid": os.getpid(),
        "module": __name__,
        "file": __file__,
        "globals": {
            "embedding_model": {
                "loaded": _EMBEDDER_INSTANCE is not None,
                "id": id(_EMBEDDER_INSTANCE),
            },
            "table_vs": {
                "loaded": _VS_TABLE is not None,
                "id": id(_VS_TABLE),
            },
            "full_fields_vs": {
                "loaded": _FULL_FIELDS_VS is not None,
                "id": id(_FULL_FIELDS_VS),
            },
            "field_docs": {
                "loaded": _FIELD_DOCS_CACHE is not None,
                "id": id(_FIELD_DOCS_CACHE),
            },
            "field_embs": {
                "loaded": _FIELD_EMBS_CACHE is not None,
                "id": id(_FIELD_EMBS_CACHE),
            },
            "table_to_idx": {
                "loaded": _TABLE_TO_IDX_CACHE is not None,
                "id": id(_TABLE_TO_IDX_CACHE),
            },
            "master_vs": {
                "loaded": _VS_MASTER is not None,
                "id": id(_VS_MASTER),
            },
            "gemini_client": {
                "loaded": _GEMINI_CLIENT is not None,
                "id": id(_GEMINI_CLIENT),
            },
            # "symspell": {
            #     "loaded": sym_spell is not None,
            #     "id": id(sym_spell),
            # },
            "keywords": {
                "loaded": _KEYWORDS_SET is not None,
                "id": id(_KEYWORDS_SET),
            },
        }
    }

@lru_cache(maxsize=512)
def _get_cached_embedding(q: str, request_id: str) -> tuple:
    publish_pipeline_update(
            request_id,
            "embedding_start",
            "embedding started"
    )
    emb = get_embedding_engine()
    publish_pipeline_update(
            request_id,
            "embedding_end",
            "get_embedding_engine ended"
    )
    vec = emb.embed_query(q)
    publish_pipeline_update(
            request_id,
            "embedding_query_done",
            "embedding query done"
    )
    return tuple(vec)  # tuple for hashability


def create_entity(state:SQLState):
    request_id = state.get("request_id")
    res = call_fvs_table_search(True, state.get("formatted_q"), state.get("request_id"))
    prompt = f"""You are selecting the correct ERPNext DocType for entity creation.
User query:
{state.get("formatted_q")}
Candidate DocTypes:
{res[:10]}
Rules:
- Choose only one DocType from the candidate list.
- Use only the given candidate DocTypes.
- Do not add the "tab" prefix.
- Return only valid JSON.
- Do not return explanations or extra text.
- If no suitable DocType is found, return {{"doctype": ""}}.
Output format:
{{"doctype": "chosen_doctype_name"}}

    """
    publish_pipeline_update(
    request_id,
    "Detecting doctype for creation",
    "Detecting doctype for creation",
    done=True
)
    res = call_gemini(prompt,"")
    try:
        if isinstance(res, str):
            res = res.replace("```json", "").replace("```", "").strip()
            result = json.loads(res)
            doctype = result.get("doctype", "")
        return {**state,"doc": doctype}
    except json.JSONDecodeError as e:
        return {**state,"error":str(e)}




def call_fvs_table_search(get_table: bool, q: str, request_id: str) -> List[str]:
    # get cached embedding
    publish_pipeline_update(
            request_id,
            "Inside the Table Search Function",
            _("Inside the Table Search Function")
        )
    q_vec = np.array(_get_cached_embedding(q,request_id), dtype="float32")
    
    # use FAISS index directly instead of similarity_search
    publish_pipeline_update(
            request_id,
            "q_vec_ready",
            _("q_vec_ready")
        )
    vs = get_vs(get_table)
    publish_pipeline_update(
            request_id,
            "vs_ready",
            _("vs_ready")
        )
    scores, indices = vs.index.search(q_vec.reshape(1, -1), k=20)
    publish_pipeline_update(
            request_id,
            "index_search_done",
            _("index_search_done")
        )
    
    out, seen = [], set()
    for idx in indices[0]:
        if idx == -1:
            continue
        doc_id = vs.index_to_docstore_id[idx]
        doc = vs.docstore.search(doc_id)
        t = doc.metadata.get("table") if get_table else doc.metadata.get("report_name")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out



def _parse_json_list(raw: str) -> List[Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


from langchain_community.vectorstores import FAISS
import faiss

def build_hnsw_index(embeddings):
    dim = len(embeddings[0])
    
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 = neighbors (tune this)
    index.hnsw.efConstruction = 200       # build quality
    index.hnsw.efSearch = 50              # search accuracy/speed tradeoff
    
    return index


def call_retrieve_multi_line(user_question: str, request_id: str) -> Dict[str, Any]:
    try:
        top_tables = call_fvs_table_search(True, user_question, request_id)
        publish_pipeline_update(
            request_id,
            "table_retrieval_done",
            _("Tables retrieved")
        )
        fields_candidates= call_fvs_field_search_global_k(
            user_question,
            selected_tables=top_tables,
            k_total=40,
            request_id=request_id
        )
        publish_pipeline_update(
            request_id,
            "field_retrieval_done",
            "Fields selected"
        )
        return {
            "selected_fields": fields_candidates,
            "selected_tables": top_tables,
            "top_tables": top_tables,
            "top_fields": fields_candidates,
        }
    except frappe.exceptions.ValidationError:
        raise
    except Exception as e:
        return {"selected_fields": {}, "selected_tables": [], "top_tables": [], "error": str(e)}

def call_fvs_field_search_global_k(
    user_question: str,
    selected_tables: List[str],
    k_total: int = 40,
    request_id: Optional[str] = None
) -> str:

    if not user_question or not selected_tables:
        return ""

    docs, embs, table_to_idx = load_field_matrix()

    # emb = get_embedding_engine()

    q_vec = np.array(
        _get_cached_embedding(user_question, request_id),
        dtype="float32"
    )

    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-12)

    # collect indices
    all_idxs = []
    for t in selected_tables:
        all_idxs.extend(table_to_idx.get(t, []))

    if not all_idxs:
        return ""

    sub_embs = embs[all_idxs]
    scores = sub_embs @ q_vec

    top_global = np.argsort(-scores)[:k_total]

    grouped = {}
    seen = set()

    for i in top_global:
        doc_i = all_idxs[int(i)]
        d = docs[doc_i]

        meta = getattr(d, "metadata", {}) or {}
        is_table = meta.get("is_table")
        table = meta.get("table")
        field = meta.get("field") or meta.get("name") 

        if not table or not field:
            continue

        key = (table, field)
        if key in seen:
            continue
        seen.add(key)

        name = field

        # join hint
        if meta.get("join_hint"):
            linked_table = meta["join_hint"].get("table")
            if linked_table:
                name += f" -> {linked_table}"

        # options
        if meta.get("options"):
            opts = meta["options"]
            if isinstance(opts, list):
                name += " {" + ", ".join(str(o) for o in opts[:5]) + "}"
        grouped.setdefault(table, {
            "is_table": is_table,
            "fields": []
        })

    grouped[table]["fields"].append(name)
    
    res = format_schema_context(grouped)
    # 🔥 final compact string
    return res


# Node 1: Retrive with Fiass Vector Store.
def schema_retriever(state: SQLState) -> SQLState:
    config = ChangAIConfig.get()
    try:
        if config["REMOTE"]:
            hits = remote_embedder_request(state.get("formatted_q", "") or state.get("question", ""))
            return {**state, "hits": hits}
        else:
            out = call_retrieve_multi_line(state.get("formatted_q") or state.get("question") or "",state.get("request_id"),)
            return {
                **state,
                "retrieval_mode": "multi",
                "top_tables": out.get("top_tables", []),
                "top_fields": out.get("top_fields", {}),
                "selected_fields": out.get("selected_fields", ""),
                "selected_tables": out.get("selected_tables", []),
            }
    except frappe.exceptions.ValidationError:
        raise
    except Exception as e:
        return {**state, "error": f"Schema retrieval failed: {e}"}


# # Node 2: Build schema context from hits - for SQL Prompt
def hits_to_prompt_context(state:SQLState) -> SQLState:
    ctx=hits_to_schema_context(state["hits"],title="SCHEMA CONTEXT",max_fields_per_table=25)
    entity_context=state.get("entity_cards", [])
    full_context = ctx

    if entity_context:
        full_context += "\n\nENTITY_CARDS:\n"
        full_context += "\n".join(entity_context)

    return {
        **state,
        "context": full_context
    }


# # Node 3:Generate the SQL Prompt and call LLM(Ollama Http)
def generate_sql(state:SQLState) -> SQLState:
    # if state.get("context") == "" or state.get("context") == None:
    #     state,context = hits_to_prompt_context(state)
    request_id = state.get("request_id")
    fields = _safe_strip(state.get("selected_fields") or "")
    entity_cards = state.get("entity_cards") or []
    entity_block = ""
    config = ChangAIConfig.get()
    formatted_q = state.get("formatted_q")
    if not formatted_q:
        return {**state, "sql": "", "orm": "", "error": "No question to generate SQL for", "sql_prompt": ""}
    if entity_cards:
        entity_block = "\n\nENTITY_CARDS:\n" + "\n".join(str(c) for c in entity_cards)
    if config["retriever_structure"]=="multi line":
        context = fields + (entity_block or "")
        prompt = fill_sql_prompt(formatted_q, context)
        state["final_prompt"] = prompt
    else:
        prompt=fill_sql_prompt(formatted_q,state["context"])
    try:
        response=call_model(prompt,"llm",SQL_SYS_PROMPT)
        if not response:
            return {**state, "error": "Empty response from LLM", "sql_prompt": prompt}
        if isinstance(response, str):
            response = json.loads(response)
        sql = response.get("sql", "")
        orm = response.get("orm", "")
        publish_pipeline_update(
            request_id,
            "sql_generated",
            "SQL generated"
        )
        return {**state,"sql_prompt":prompt,"sql":sql,"orm":orm,"error":None}
    except frappe.exceptions.ValidationError:
        raise
    except Exception as e:
        return {**state,"error": f"LLM call failed: {e}","sql_prompt":prompt}


# # Node 4:Validate the SQL Generate with meta schema mapping using SQLGlot
def validate_sql(state: SQLState) -> SQLState:
    sql = clean_sql(state.get("sql") or "")
    if not sql:
        return {
            **state,
            "validation": {
                "ok": False,
                "unknown_tables": [],
                "unknown_columns": [],
                "ambiguous_columns": [],
                "details": {
                    "parse_error": sql or "Empty SQL from LLM"
                },
            },
        }

    val = validate_sql_against_mapping(sql, mapping_data, dialect="mysql")
    return {**state, "validation": val}


def remote_entity_embedder(q: str) -> Union[list, str]:
    config = ChangAIConfig.get()
    payload = {"version": config["entity_retriever"], "input": {"query": q}}
    headers = {
        "Content-Type": APPLICATION_JSON,
        "Prefer": "wait",
        "Authorization": f"Bearer {config['API_TOKEN']}",
    }
    response = _post_json(config["URL"], headers, payload)
    return response


settingsUrl = frappe.utils.get_url(
    "/app/changai-settings/ChangAI%20Settings"
)


def get_master_vs():
    global _VS_MASTER
    try:
        if _VS_MASTER is None:
            emb = get_embedding_engine()
            if emb is None:
                frappe.throw(_(EMBEDDING_ENGINE_NONE_MESSG))

            master_vs_path = frappe.get_site_path(
                "private", "changai", "fvs_stores", "erpnext", "masterdata_fvs"
            )
            if not os.path.exists(master_vs_path):
                frappe.throw(_(
                    "FAISS MASTER store not found at {0}.<br><br>"
                    "Please open "
                    "<a href='{1}' target='_blank' rel='noopener noreferrer'>Go to Settings Page</a>"
                    "and click on the <b>Update Master Data</b> button in the Training tab.<br><br>"
                    "Check Quick Start Guide Here 👇<br>"
                    "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color:#1e90ff;'>Click here</a><br><br><br>"
                    "<a href='{3}' target='_blank' rel='noopener noreferrer' style='color:#1e90ff;'>ERPGulf.com</a>"

                ).format(
                    master_vs_path,
                    settingsUrl,
                    CHANGAI_GUIDE_LINK,
                    ERPGULF_LINK
                ))

            _VS_MASTER = FAISS.load_local(
                master_vs_path,
                emb,
                allow_dangerous_deserialization=True
            )
    except Exception as e:
        frappe.log_error(f"Error loading master vector store: {e}", "ChangAI Master VS Load Error")

    return _VS_MASTER

import re

def append_entity_field_to_schema(top_fields: str, table_name: str, field_name: str) -> str:
    """
    Append field_name to the FIELDS section of table_name if missing.
    Example: append customer_name into TABLE: tabCustomer block.
    """

    pattern = rf"(TABLE:\s*{re.escape(table_name)}\n.*?FIELDS:\n)(.*?)(?=\n\nTABLE:|\Z)"

    def replace_block(match):
        header = match.group(1)
        fields_block = match.group(2)

        # already exists
        if re.search(rf"^- {re.escape(field_name)}(\s|$)", fields_block, re.MULTILINE):
            return match.group(0)

        return header + fields_block.rstrip() + f"\n- {field_name}\n"

    return re.sub(pattern, replace_block, top_fields, count=1, flags=re.DOTALL)


def local_entity_embedder(q: str) -> List[Dict[str, Any]]:
    hits = get_master_vs().similarity_search(q, k=20)
    out, seen = [], set()
    for h in hits:
        entity_type = h.metadata.get("entity_type")   # example: tabCustomer
        entity_id = h.metadata.get("entity_id")    # example: customer_name
        entity_label = h.metadata.get("entity_label")
        # if entity_type in state["selected_tables"]:
        #     state["selected_fields"] = append_entity_field_to_schema(
        #         top_fields=state["selected_fields"],
        #         table_name=entity_type,
        #         field_name=entity_id
        #     )

        key = (entity_type, entity_label)
        if key not in seen:
            seen.add(key)
            out.append({"entity_type": entity_type, "entity_id": entity_id, "entity_label": entity_label})
    return out


def call_entity_retriever(isreport: bool, qstn: str, state: Dict) -> Dict[str, Any]:
    config = ChangAIConfig.get()
    if config["REMOTE"] and config["llm"] == "QWEN3":
        response = remote_entity_embedder(qstn)

        if not response.get("ok"):
            frappe.log_error(f"Entity retriever failed: {response.get('body')}", "ChangAI Entity Retriever")
            return {"raw": response, "cards": []}

        body = response.get("body") or {}
        output = body.get("output") or {}
        results = output.get("results") or []

        cards = [r.get("entity_label") for r in results if r.get("entity_label")]

        return {"raw": body, "cards": cards}
    else:
        from changai.changai.api.v2.schema_utils import phonetic_match
        entity_words = state.get("entity_words")
        cards = []
        debug=[]
        if entity_words is None:
            return {"cards":[]}
        for word in entity_words:
            result = phonetic_match(isreport, word)
            labels = result.get("entity_labels") or []
            debug.append({
                "word": word,
                "result": result,
                "labels": labels
    })   
            for label in labels:
                if isreport:
                    try:
                        table_field, _  = labels.split(":", 1)
                        table, field = table_field.split(".", 1)
                        doctype = table.removeprefix("tab")
                        state["doc"] =  doctype
                    except Exception:
                        pass
                if label and label not in cards:
                    cards.append(label)
        return {"cards": cards}


# # Node 5:Repair Loop :Simple prompt for one more try.
def repair_sqlquery(state: SQLState) -> SQLState:
    hints: List[str] = []
    tries = int(state.get("tries") or 0) + 1
    val = state.get("validation", {})
    unknown_tables = val.get("unknown_tables", [])
    unknown_cols = val.get("unknown_columns", [])
    ambiguous = val.get("ambiguous_columns", [])

    if unknown_tables:
        hints.append(f"Unknown tables:{unknown_tables}.Use only tables in context")
    if unknown_cols:
        hints.append(f"Unknown Columns:{unknown_cols}.Use only fields listed for each tables from the context")
    if ambiguous:
        hints.append(f"Ambiguous columns(qualify them):{ambiguous}")
    sql_prompt = state.get("sql_prompt")
    if not sql_prompt:
        return {**state, "tries": tries, "error": "No SQL prompt to repair from"}
    patched_prompt =sql_prompt + "\n\n#VALIDATION HINTS\n" + "\n".join(f"-{h}" for h in hints)

    try:
        response = call_model(patched_prompt,"llm",SQL_SYS_PROMPT)
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return {**state, "tries": tries, "error": f"{response[:200]}"}

        if not response or not isinstance(response, dict):
            return {**state, "tries": tries, "error": "Repair: empty or invalid response from LLM"}

        sql = response.get("sql", "")
        return {**state, "sql": sql, "tries": tries, "error": None}
    except frappe.exceptions.ValidationError:
        raise
    except Exception as e:
        return {**state, "tries": tries, "error": f"Repair call failed {e}"}


def detect_specific_entities(state: SQLState) -> SQLState:
    if not state.get("contains_values"):
        return {**state, "entity_cards": [], "entity_raw": None}
    
    q = (state.get("formatted_q") or "").strip()
    if not q:
        return {**state, "entity_cards": [], "entity_raw": None}

    try:
        res = check_file_updates("master_data.yaml")

        if not res.get("data"):
            frappe.throw(_(
                "Master Data does not exist. Because of this, results may not be accurate. "
                "For better accuracy, please open "
                "<a href='{0}' target='_blank' rel='noopener noreferrer' style='color:#1e90ff;'>Go to Settings Page</a> "
                "and click on the <b>Update Master Data</b> button in the Training tab.<br><br>"
                "Check Quick Start Guide Here 👇:<br>"
                "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
                "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color:#1e90ff;'>ERPGulf.com</a>"
            ).format(settingsUrl, CHANGAI_GUIDE_LINK, ERPGULF_LINK))

        if res.get("is_stale") and res.get("days", 0) > 0:
            frappe.throw(_(
                "Your master data is {0} days old. "
                "Because of this, results may not be accurate. "
                "For better accuracy, please open "
                "<a href='{1}' target='_blank' rel='noopener noreferrer' style='color:#1e90ff;'>Go to Settings Page</a> "
                "and click on the <b>Update Master Data</b> button in the Training tab.<br><br>"
                "Check Quick Start Guide Here 👇:<br>"
                "<a href='{2}' target='_blank' rel='noopener noreferrer' style='color: #1e90ff;'>Click here</a><br>"
                "<a href='{3}' target='_blank' rel='noopener noreferrer' style='color:#1e90ff;'>ERPGulf.com</a>"
            ).format(res.get("days"), settingsUrl, CHANGAI_GUIDE_LINK, ERPGULF_LINK))

        out = call_entity_retriever(False, q, state)
        return {
            **state,
            "entity_cards": out.get("cards") or [],
            # "entity_raw": out.get("raw"),
        }
    except frappe.exceptions.ValidationError:
        raise
    except Exception as e:
        frappe.log_error(f"Entity retriever failed: {e}", "ChangAI Entity Gate")
        return {**state, "entity_cards": [], "entity_raw": {"error": str(e)}}


def route_after_entities(state: SQLState) -> str:
    config = ChangAIConfig.get()
    return "DIRECT" if config.get("retriever_structure") == "multi line" else "CONTEXT"


def route_guardrail(state: SQLState) -> str:
    return "ERP" if state.get("query_type") == "ERP" else "NON_ERP"

def clean_sql(s: Any) -> str:
    if isinstance(s, dict):
        s = s.get("output") or s.get("sql") or s.get("text") or json.dumps(s, ensure_ascii=False, default=str)
    elif isinstance(s, list):
        s = "\n".join(str(x) for x in s)
    else:
        s = "" if s is None else str(s)

    s = s.strip()

    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            header = s[:first_newline].strip().lower()
            if header in {"```", "```sql"}:
                s = s[first_newline + 1 :].lstrip()

    stripped = s.rstrip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()
    s = stripped

    if s[:3].lower() == "sql" and (len(s) == 3 or s[3].isspace()):
        s = s[3:].lstrip()

    return s.strip()


# # Router to decide next stage:
def router(state:SQLState) -> str:
    if state.get("error"):
        return "end"
    val=state.get("validation",{})
    if val.get("ok"):
        return "end"
    tries=int(state.get("tries") or 0)
    if tries < RETRY_LIMIT:
        return "repair"
    return "end"


def _parse_sql_ast(sql_text: str, dialect: str):
    try:
        return sqlglot.parse_one(sql_text, read=dialect), None
    except Exception as e:
        return None, str(e)


def _extract_tables(ast) -> Tuple[List[str], Dict[str, str]]:
    base_tables = []
    alias_to_table = {}
    for t in ast.find_all(exp.Table):
        if not t.name:
            continue
        base_tables.append(t.name)
        a = t.args.get("alias")
        if a and a.name:
            alias_to_table[a.name] = t.name
    return list(dict.fromkeys(base_tables)), alias_to_table


def _extract_derived_aliases(ast) -> Set[str]:
    derived = set()
    for sq in ast.find_all(exp.Subquery):
        a = sq.args.get("alias")
        if a and a.name:
            derived.add(a.name)
    for cte in ast.find_all(exp.CTE):
        a = cte.args.get("alias")
        if a and a.name:
            derived.add(a.name)
    return derived


def _extract_select_aliases(ast) -> Set[str]:
    aliases = set()
    for sel in ast.find_all(exp.Select):
        for proj in sel.expressions:
            if isinstance(proj, exp.Alias) and proj.alias:
                aliases.add(proj.alias)
    return aliases


def _validate_qualified_col(col_name: str, qual: str, mapping: Dict,
                              alias_to_table: Dict, derived_aliases: Set) -> Optional[Tuple]:
    if col_name == "*" or qual in derived_aliases:
        return None
    if qual in mapping:
        if col_name not in mapping[qual]:
            return (f"{qual}.{col_name}", qual)
        return None
    if qual in alias_to_table:
        real = alias_to_table[qual]
        if real in mapping and col_name not in mapping[real]:
            return (f"{qual}.{col_name}", real)
        return None
    return (f"{qual}.{col_name}", None)


def _validate_unqualified_col(col_name: str, base_tables_set: Set,
                               mapping: Dict, select_aliases: Set,
                               unknown_cols: List, ambiguous: Set):
    if col_name in select_aliases:
        return
    candidates = [t for t in base_tables_set if col_name in mapping.get(t, [])]
    if len(candidates) == 0:
        unknown_cols.append((col_name, None))
    elif len(candidates) > 1:
        ambiguous.add(col_name)


def _validate_columns(ast, mapping: Dict, alias_to_table: Dict,
                       derived_aliases: Set, select_aliases: Set,
                       base_tables_set: Set) -> Tuple[List, Set]:
    unknown_cols: List[Tuple[str, str]] = []
    ambiguous: Set[str] = set()
    for col in ast.find_all(exp.Column):
        if not col.name:
            continue
        if col.table:
            result = _validate_qualified_col(
                col.name, str(col.table), mapping, alias_to_table, derived_aliases
            )
            if result:
                unknown_cols.append(result)
        else:
            _validate_unqualified_col(
                col.name, base_tables_set, mapping, select_aliases, unknown_cols, ambiguous
            )
    return unknown_cols, ambiguous


def validate_sql_against_mapping(
    sql_text: str,
    mapping: Dict[str, List[str]],
    dialect: str = "mysql"
) -> Dict[str, Any]:
    result = {
        "ok": True,
        "unknown_tables": [],
        "unknown_columns": [],
        "ambiguous_columns": [],
        "details": {"from_tables": [], "alias_to_table": {}, "derived_aliases": [], "select_aliases": []},
    }

    ast, parse_error = _parse_sql_ast(sql_text, dialect)
    if parse_error:
        result["ok"] = False
        result["details"]["parse_error"] = parse_error
        return result

    base_tables, alias_to_table = _extract_tables(ast)
    derived_aliases = _extract_derived_aliases(ast)
    select_aliases = _extract_select_aliases(ast)

    result["details"]["from_tables"] = base_tables
    result["details"]["alias_to_table"] = alias_to_table
    result["details"]["derived_aliases"] = sorted(derived_aliases)
    result["details"]["select_aliases"] = sorted(select_aliases)

    unknown_tables = [t for t in base_tables if t not in mapping]
    if unknown_tables:
        result["ok"] = False
        result["unknown_tables"] = unknown_tables

    unknown_cols, ambiguous = _validate_columns(
        ast, mapping, alias_to_table, derived_aliases, select_aliases, set(base_tables)
    )
    if unknown_cols or ambiguous:
        result["ok"] = False
        result["unknown_columns"] = unknown_cols
        result["ambiguous_columns"] = sorted(ambiguous)

    return result

def routeNonErpToAI(state: SQLState):
    question= state["question"]
    sys_prompt = """You are ChangAI, an intelligent assistant powered by ERPGulf. 
The user has asked a general question that is not related to ERP. 
Answer the question clearly and helpfully.
Always mention that you are ChangAI by ERPGulf when introducing yourself."""
    if frappe.utils.cint(state.get("sendNonErptoAI", 0)) == 1 or state.get("sendNonErptoAI") == "true":
        try:
            res = call_gemini(question,sys_prompt)
            return {**state, "non_erp_res": res}
        # except ValidationError as ve:
        #     return {**state,"error":str(ve)}
        except frappe.exceptions.ValidationError:
            raise
        except Exception as e:
            return {**state, "non_erp_res": "Model Calling Failed .Please try Again","error":str(e)}


    else:
        res= send_non_erp_request(state)
        return res



def prepare_report_action(state: SQLState) -> SQLState:
    report_name = None
    q = state.get("formatted_q") or state.get("question") or ""
    request_id = state.get("request_id")
    report_name = state.get("report_name")

    cards = call_entity_retriever(True, q, state) if state.get("contains_values") else {"cards": []}

    # reports_listed = call_fvs_table_search(False, q, request_id)
    doctype = state.get("doc") if state.get("doc") else None
    result = []
    # for report in reports_listed[:30]:
    #     report_name = report.get("name") if isinstance(report, dict) else report
    #     result.append({
    #         "report": report_name,
    #         "filters": get_report_filter_fields(report_name)
    #     })
    result.append({"report":report_name,"filters": get_report_filter_fields(report_name)})
    prompt = f"""
Given the user query:
{q}
Detected DocType: use this value only for filtering by party type.
{doctype}
Available Report and Filter Fields:
{result}
after seeing the reports look wisely at the question and find the intent that speaks of report name take that intent and match with listed reports 
and find the best matching report name.
Carefully identify the report intent from the user's question and match only that intent against the available report names. Do not use entity names, entity types, party names, customer names, supplier names, or filter values to choose the report, even if similar words appear in the listed report names. Entity information must be used only for filters, not for report_name selection.
even if the right report name is not ranked top find it from the top 10 and use it .
Entity Cards:
{cards}
Report selection MUST be based on the user's requested report type first.
Entity type (Customer, Supplier, Employee, etc.) is only used for filters and MUST NOT determine the report name.
Examples:
User: "show account statement for nova supplier"
Report: General Ledger
Filters:
{{
  "party_type": "Supplier",
  "party": "nova supplier"
}}
TASK
Select exactly ONE report from the Available Reports and Filter Fields.
Step 1 - Understand User Intent
- Analyze the user's intent and meaning carefully.
- Match reports by meaning and business purpose, not only exact words.
- Consider synonyms and common ERP terminology.
Examples:
- "account statement", "customer statement", "supplier statement", "ledger statement" → usually General Ledger when available.
- "profit report" → Gross Profit.
- "trial balance" → Trial Balance.
- "balance sheet" → Balance Sheet.
Step 2 - Select the Best Report
- report_name MUST be selected only from the provided report list.
- Never invent a report name.
- Never return an empty report name.
- Choose the report whose purpose best matches the user's request.
- If multiple reports are similar, select the one whose available filters best satisfy the user's request.
- Report selection is more important than filter selection.
- Do not choose a report simply because a filter matches.
Step 3 - Apply Filters
- Use ONLY filters that exist in the selected report.
- Never invent filter names.
- Never invent filter values.
- Use Entity Card values whenever applicable.
- Apply only relevant filters supported by the selected report.
Party Filtering Rules
If:
- detected DocType is Customer, Supplier, Employee, Shareholder, Student, Member, or any party-type entity
AND
- selected report contains both party_type and party filters
THEN:
- party_type = detected DocType
- party = entity value from Entity Card
Example:
DocType = Customer
Entity Card:
customer_name: مؤسسة الإبداع التجارية
Filters:
{{
  "party_type": "Customer",
  "party": "مؤسسة الإبداع التجارية"
}}
Important:
- Never infer party_type from the entity value.
- Always use the detected DocType.
- Prefer party_type + party over name-based filters whenever both are available.
Date Rules
Whenever a report requires date filtering:
- Convert relative dates into explicit dates.
- Use DD-MM-YYYY format only.
- Never use natural language dates.
Date Format Rules:
- Always output dates in YYYY-MM-DD format.
- Never use DD-MM-YYYY, MM-DD-YYYY, or natural language dates.
- Convert all relative dates to YYYY-MM-DD before returning filters.
Validation Before Answering
Verify:
1. report_name exists in Available Reports.
2. Every filter exists in the selected report.
3. Every filter value comes from the user query, Entity Cards, or a valid date conversion.
4. No invented report names.
5. No invented filters.
6. No empty report_name.
7. Report intent matches the user's request.
Return ONLY valid JSON:
{{
  "report_name": "<selected_report>",
  "filters": {{
    "<filter_name>": "<filter_value>"
  }}
}}
**Always use party_type when it is avaible for filtering if filtering by party name like customer, supplier,item etc
STRICT OUTPUT
- Return only JSON.
- No markdown.
- No explanations.
- No comments.
- Response must start with {{ and end with }}.
"""

    response = call_model(prompt, "llm", "")

    raw_response = str(response or "").strip()
    raw_response = raw_response.replace("```json", "").replace("```", "").strip()

    frappe.logger().info(f"prepare_report_action raw response: {repr(raw_response)}")

    if not raw_response:
        response = {"report_name": "", "filters": {}}
    else:
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError:
            frappe.log_error(
                f"Invalid JSON from prepare_report_action:\n{raw_response}",
                "prepare_report_action JSON Error"
            )
            response = {"report_name": "", "filters": {}}

    if not isinstance(response, dict):
        response = {"report_name": "", "filters": {}}
    publish_pipeline_update(
        request_id,
        "Detected Report name & filters",
        "Detected Report name & filters",
        data={"report_name": response.get("report_name"),"filters":response.get("filters")},
        done=True
    )
    return {
        **state,
        "report_name": response.get("report_name") or "",
        "filters": response.get("filters") or {},
        "reports_filter_before_call":result[:30],
        "entity_cards": cards.get("cards") or [],
        "entity_raw": cards,
        "error": None,
    }


# def route_action(state: SQLState):
#     if state.get("create_entity") is True:
#         qstn = state.get("user_question")
#         response = call_fvs_table_search(False, qstn)
#         return response
#         return "CREATE_ENTITY"
#     if state.get("open_report") is True:
#         return "OPEN_REPORT"
#     else:
#         return "CONTINUE"

# Building the Workflow Graph
workflow=StateGraph(SQLState)
workflow.add_node("rewrite_question",rewrite_question)
workflow.add_node("guardrail_router",guardrail_router)
workflow.add_node("retrieve",schema_retriever)
workflow.add_node("detect_entities", detect_specific_entities)
workflow.add_node("build_context",hits_to_prompt_context)
workflow.add_node("generate_sql",generate_sql)
workflow.add_node("validate_sql",validate_sql)
workflow.add_node("repair_sql",repair_sqlquery)
workflow.add_node("send_non_erp_request",send_non_erp_request)
workflow.add_node("routeNonErpToAI",routeNonErpToAI)
workflow.add_node("prepare_report_action", prepare_report_action)
workflow.add_node("create_entity", create_entity)
# workflow.add_node("stop_follow_msg_api", stop_follow_msg_api)


workflow.set_entry_point("guardrail_router")
workflow.add_conditional_edges("guardrail_router",route_guardrail,{"ERP":"rewrite_question","NON_ERP":"routeNonErpToAI"})
# workflow.add_edge("guardrail_router", "rewrite_question")
workflow.add_edge("routeNonErpToAI", END)
workflow.add_conditional_edges(
    "rewrite_question",
    route_action,
    {
        "CREATE_ENTITY": "create_entity",
        "OPEN_REPORT": "prepare_report_action",
        "STOP_FOLLOW":END,
        "CONTINUE": "retrieve",
    }
)
workflow.add_edge("prepare_report_action", END)
workflow.add_edge("create_entity", END)
# workflow.add_edge("stop_follow_msg_api", END)
workflow.add_edge("retrieve","detect_entities")
workflow.add_conditional_edges("detect_entities", route_after_entities, {"CONTEXT":"build_context","DIRECT":"generate_sql"})
workflow.add_edge("build_context", "generate_sql")
workflow.add_edge("generate_sql",END)
# workflow.add_conditional_edges("validate_sql",router,{"repair":"repair_sql","end":END})
# workflow.add_edge("repair_sql","validate_sql")
checkpointer=MemorySaver()
app=workflow.compile(checkpointer=checkpointer)

def _build_match_conditions(doctypes: List[str]) -> str:
    conditions = []
    for t in doctypes:
        doctype = t[3:] if t.startswith("tab") else t
        cond = build_match_conditions(doctype)
        if cond:
            conditions.append(cond)
    return " AND ".join(conditions) if conditions else ""


def _append_conditions(sql: str, combined: str) -> str:
    if "where" in sql.lower():
        return sql + f" AND {combined}"
    return sql + f" WHERE {combined}"


def execute_query(sql: str, doctypes: List[str]) -> Any:
    try:
        if not sql:
            return []
        if not str(sql).lower().strip().startswith("select"):
            frappe.throw(_("Only SELECT queries are allowed."
                           "Check Quick Start Guide Here 👇:\n {0}").format(CHANGAI_GUIDE_LINK))
        sql = sql.rstrip().rstrip(';')
        combined = _build_match_conditions(doctypes)
        if combined:
            sql = _append_conditions(sql, combined)
        return frappe.db.sql(sql, as_dict=True)
    except PermissionError:
        return {
            "error": _("You do not have permission to access this data. Check the Quick Start Guide here 👇: {0}").format(
                f'<a href="{CHANGAI_GUIDE_LINK}" target="_blank">Click here</a><br><br><a href="{ERPGULF_LINK}" target="_blank">ERPGulf.com</a>'
            )        }
    except Exception as e:
        return {"error": f"SQL Execution Failed: {e}\n Check Quick Start Guide Here 👇:\n {CHANGAI_GUIDE_LINK}"}


@frappe.whitelist(allow_guest=False)
def support_bot(message: str) -> Dict[str, Any]:
    user_email = frappe.session.user
    full_name = frappe.get_value("User", frappe.session.user, "full_name")
    prompt = SUPPORT_USER_PROMPT.format(user_message=message)
    raw = call_gemini(prompt, SUPPORT_SYS_PROMPT)
    output = json.loads(raw)
    task_flag = (output.get("task_flag") or "UNKNOWN").strip()
    ticket_id = output.get("ticket_id")

    if isinstance(ticket_id, str) and ticket_id.isdigit():
        ticket_id = int(ticket_id)
    if not isinstance(ticket_id, int):
        ticket_id = None

    if task_flag == "CREATE_TICKET":
        try:
            response = create_helpdesk_ticket(message, full_name, user_email)
            return json.loads(response.get_data(as_text=True))  # ✅ unwrap Response → dict
        except Exception as e:
            return {"error": str(e)}

    if task_flag == "TICKET_DETAILS":
        if not ticket_id:
            return {"kind": "TICKET_DETAILS", "error": "Ticket id missing. Please say like: ticket 29"}
        try:
            response = get_user_tickets(ticket_id)
            return json.loads(response.get_data(as_text=True))  # ✅ unwrap Response → dict
        except Exception as e:
            return {"error": str(e)}

    if task_flag == "GET_USER_TICKETS":
        response = get_user_tickets()
        return json.loads(response.get_data(as_text=True))      # ✅ unwrap Response → dict

    return {"kind": "UNKNOWN", "message": "Please describe the issue or provide a ticket number."}

def save_logs(
    user_question: Optional[str] = None,
    formatted_q: Optional[str] = None,
    context: Optional[str] = None,
    sql: Optional[str] = None,
    val: Any = None,
    result: Any = None,
    tries: Optional[int] = None,
    err: Any = None,
    formatted_result: Any = None,
    tables:Any=None,
    fields:Any=None,
    entity_debug:Any=None,
    type_=None
) -> str:
    def to_json_if_needed(v: Any) -> Any:
        if isinstance(v, (dict, list)):
            return json.dumps(v, default=str, ensure_ascii=False)
        return v
    
    MAX_LOG_LEN = 140
    doc = frappe.new_doc("ChangAI Logs")
    doc.user_question = user_question
    safe_question=(formatted_q[:137] + "..." if len(formatted_q) > MAX_LOG_LEN else formatted_q)
    doc.rewritten_question = safe_question
    doc.schema_retrieved = to_json_if_needed(context)
    doc.sql_generated = to_json_if_needed(sql)
    doc.validation = to_json_if_needed(val)
    doc.tries = tries
    doc.error = to_json_if_needed(err)
    doc.result = to_json_if_needed(result)
    doc.formatted_result = to_json_if_needed(formatted_result)
    doc.tables = to_json_if_needed(tables)
    doc.fields = to_json_if_needed(fields)
    doc.entity = to_json_if_needed(entity_debug)
    doc.type = type_
    doc.insert(ignore_permissions=True)
    return doc.name


def format_data_conversationally(user_data: Any) -> str:
    # Safe: CONVERSATION_TEMPLATE is a hardcoded internal template string.
    # User SQL result is passed only as data context, not as template source.
    # nosemgrep: frappe-semgrep-rules.rules.security.frappe-ssti
    return render_template(
        CONVERSATION_TEMPLATE,  # nosemgrep: frappe-semgrep-rules.rules.security.frappe-ssti

        {"data": user_data}
    )


def format_data(qstn: str, sql_data: Any) -> Dict[str, str]:
    if isinstance(sql_data, (dict, list)):
        db_result_json = json.dumps(sql_data, ensure_ascii=False, default=str)
    else:
        db_result_json = str(sql_data) if sql_data is not None else "{}"

    sys_prompt = """
You are ChangAI, a warm and intelligent business assistant.
Your job is to turn raw database results into clear, friendly, human-readable answers.
CONTENT RULES:
- Use BOTH the user question and the DB result JSON to form the answer.
- Use ONLY values present in the JSON. NEVER invent numbers or fields.
- If result is empty, respond warmly and suggest refining the search.
- Do NOT mention SQL, tables, fields, JSON, reasoning, or steps.

TONE & STYLE:
- Warm, conversational, and helpful — like a knowledgeable friend, not a report.
- If the question is in Arabic, reply in natural Arabic — not translated English.
- Never respond with a cold, empty, or robotic answer.

FORMATTING:
- Start with ONE relevant emoji matching the topic (📦💰🧾👥📊📅🔍💤📉)
- For 3+ items, use a bullet list: • Item — value
- If list exceeds shown items, state exactly how many remain.
- Keep answers brief (1–6 lines). Lead with the direct answer, then light context.

CLOSING:
- End with ONE short, relevant follow-up question to keep the conversation going.
- Make it feel natural, not robotic.
Never list names or items in a comma-separated line. Ever.
OUTPUT:
- Markdown ALLOWED: **bold**, • bullets, emojis
- i dont want too much gap between the texts also gaps are not allowed between items listed.
- No JSON. No code blocks. No labels. No explanations.
- Output ONLY the final user-facing answer. Nothing else.
- if the user question is in english reply in english only very important.
if the user question is in arabic respond in arabic only. and if the question is in english respond answer also english
"""
    user_prompt=f"""
            QUESTION:
            {qstn}

            DATABASE_RESULT_JSON:
            {db_result_json}
    """
    output = call_model(user_prompt,"llm",sys_prompt)
    answer = str(output)
    return {"answer": answer}

def _collect_docs(hits: Union[List[Any], Dict, str]) -> List[Tuple[str, Dict]]:
    def _to_txt_md(doc: Any) -> Tuple[str, Dict]:
        if isinstance(doc, dict):
            return doc.get("text", "") or "", doc.get("metadata", {}) or {}
        if isinstance(doc, str):
            return doc, {}
        return "", {}

    if isinstance(hits, dict) and "message" in hits and isinstance(hits["message"], list):
        hits = hits["message"]

    if isinstance(hits, (dict, str)) or hasattr(hits, "page_content"):
        return [_to_txt_md(hits)]
    return [_to_txt_md(d) for d in (hits or [])]


def _parse_tag(txt: str, tag: str) -> str:
    m = re.search(rf"\[{re.escape(tag)}\]\s*(.+?)(?:\s*\||\s*$)", txt or "")
    return m.group(1).strip() if m else ""


def _infer_type(txt: str) -> str:
    if not (txt or "").startswith("["):
        return ""
    order = [
        ("TABLE", "table"), ("FIELD", "field"), ("JOIN", "join"),
        ("METRIC", "metric"), ("ENUM", "enum"), ("PERIOD", "period"),
        ("CURRENCY", "currency"), ("ENTITY", "entity")
    ]
    for tg, tp in order:
        if txt.startswith(f"[{tg}]"):
            return tp
    return ""


class _SchemaAccumulator:
    def __init__(self):
        self.tables: List[str] = []
        self.fields_by_table: Dict[str, List[str]] = OrderedDict()
        self.joins: List[str] = []
        self.metrics: List[Tuple[str, str, str]] = []
        self.periods: List[str] = []
        self.currencies: List[str] = []
        self.enums: OrderedDict = OrderedDict()
        self.entities: List[Tuple[str, Dict]] = []

    def add_table(self, t: str):
        if t and t not in self.tables:
            self.tables.append(t)
            if t not in self.fields_by_table:
                self.fields_by_table[t] = []

    def add_field(self, tbl: str, fld: str):
        if tbl and fld:
            self.add_table(tbl)
            fq = f"{tbl}.{fld}"
            if fq not in self.fields_by_table[tbl]:
                self.fields_by_table[tbl].append(fq)

    def add_join(self, on: str):
        if on and on not in self.joins:
            self.joins.append(on)

    def add_metric(self, mname: str, mexpr: str, mtbl: str):
        if mtbl:
            self.add_table(mtbl)
        if mname:
            tup = (mname, mexpr or "", mtbl or "")
            if tup not in self.metrics:
                self.metrics.append(tup)

    def add_period(self, pname: str):
        if pname and pname not in self.periods:
            self.periods.append(pname)

    def add_currency(self, code: str):
        if code and code not in self.currencies:
            self.currencies.append(code)

    def add_enum(self, tbl: str, fld: str, vals: Any):
        if tbl:
            self.add_table(tbl)
        if tbl and fld:
            key = f"{tbl}.{fld}"
            if isinstance(vals, (list, tuple)):
                vals = ", ".join([str(v) for v in vals])
            if key not in self.enums:
                self.enums[key] = vals or ""
            self.add_field(tbl, fld)

    def add_entity(self, ent_name: str, filt: Dict):
        self.entities.append((ent_name, filt or {}))

    def sort(self):
        self.tables.sort()
        for t in self.fields_by_table:
            if t not in self.tables:
                self.tables.append(t)
        for t in self.fields_by_table:
            self.fields_by_table[t] = sorted(
                self.fields_by_table[t], key=lambda s: s.split(".", 1)[1]
            )
        self.joins.sort()
        self.metrics.sort(key=lambda x: x[0])
        self.periods.sort()
        self.currencies.sort()
        self.enums = OrderedDict(sorted(self.enums.items(), key=lambda kv: kv[0]))


def _process_enum(txt: str, md: Dict, acc: _SchemaAccumulator):
    tbl = md.get("table") or _parse_tag(txt, "TABLE")
    fld = md.get("field")
    if not fld:
        ef = _parse_tag(txt, "ENUM")
        if "." in ef:
            tbl = tbl or ef.split(".", 1)[0].strip()
            fld = ef.split(".", 1)[1].strip()
    vals = md.get("values")
    if vals is None:
        vals = _parse_tag(txt, "VALUES")
    acc.add_enum(tbl, fld, vals)


def _process_entity(txt: str, md: Dict, acc: _SchemaAccumulator):
    ent_name = md.get("entity") or _parse_tag(txt, "ENTITY") or "Entity"
    filt = md.get("filters")
    if filt is None:
        filt_txt = _parse_tag(txt, "FILTERS")
        filt = {}
        if filt_txt:
            for part in [p.strip() for p in filt_txt.split(";") if p.strip()]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    filt[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]
    acc.add_entity(ent_name, filt)


def _get_table_name(txt: str, md: Dict) -> str:
    return md.get("table") or _parse_tag(txt, "TABLE")


def _get_field_name(txt: str, md: Dict) -> str:
    return md.get("field") or _parse_tag(txt, "FIELD").split(" (", 1)[0]


def _process_table_doc(txt: str, md: Dict, acc: _SchemaAccumulator) -> None:
    acc.add_table(_get_table_name(txt, md))


def _process_field_doc(txt: str, md: Dict, acc: _SchemaAccumulator) -> None:
    acc.add_field(_get_table_name(txt, md), _get_field_name(txt, md))


def _process_join_doc(txt: str, md: Dict, acc: _SchemaAccumulator) -> None:
    acc.add_join(md.get("on") or _parse_tag(txt, "ON"))


def _process_metric_doc(txt: str, md: Dict, acc: _SchemaAccumulator) -> None:
    acc.add_metric(
        md.get("name") or _parse_tag(txt, "METRIC"),
        md.get("expression") or _parse_tag(txt, "EXPR"),
        _get_table_name(txt, md),
    )


def _process_period_doc(txt: str, md: Dict, acc: _SchemaAccumulator) -> None:
    acc.add_period(md.get("name") or _parse_tag(txt, "PERIOD"))


def _process_currency_doc(txt: str, md: Dict, acc: _SchemaAccumulator) -> None:
    acc.add_currency(md.get("code") or _parse_tag(txt, "CURRENCY"))


_DOC_PROCESSORS = {
    "table": _process_table_doc,
    "field": _process_field_doc,
    "join": _process_join_doc,
    "metric": _process_metric_doc,
    "period": _process_period_doc,
    "currency": _process_currency_doc,
    "enum": _process_enum,
    "entity": _process_entity,
}


def _process_doc(txt: str, md: Dict, acc: _SchemaAccumulator) -> None:
    dtype = md.get("type") or _infer_type(txt)
    processor = _DOC_PROCESSORS.get(dtype)
    if processor:
        processor(txt, md, acc)


def _append_table_lines(
    lines: List[str],
    acc: _SchemaAccumulator,
    max_fields_per_table: int,
) -> None:
    for tbl in acc.tables:
        lines.append(f"Table: {tbl}")
        lines.append("Fields:")
        fields = acc.fields_by_table.get(tbl, [])

        if not fields:
            lines.append("  -")
            lines.append("")
            continue

        lines.extend(f"  - {field}" for field in fields[:max_fields_per_table])

        extra_count = len(fields) - max_fields_per_table
        if extra_count > 0:
            lines.append(f"  # +{extra_count} more")

        lines.append("")


def _append_simple_section(lines: List[str], title: str, items: List[str]) -> None:
    if not items:
        return

    lines.append(f"{title}:")
    lines.extend(f"  - {item}" for item in items)
    lines.append("")


def _append_metric_lines(lines: List[str], acc: _SchemaAccumulator) -> None:
    if not acc.metrics:
        return

    lines.append("Metrics:")
    for metric_name, metric_expr, metric_table in acc.metrics:
        suffix = f"  # table: {metric_table}" if metric_table else ""
        line = f"  - {metric_name}: {metric_expr}{suffix}" if metric_expr else f"  - {metric_name}{suffix}"
        lines.append(line)
    lines.append("")


def _append_enum_lines(lines: List[str], acc: _SchemaAccumulator) -> None:
    if not acc.enums:
        return

    lines.append("Enums:")
    for key, values in acc.enums.items():
        lines.append(f"  - {key}: {values}" if values else f"  - {key}")
    lines.append("")


def _format_filter_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _append_entity_lines(
    lines: List[str],
    acc: _SchemaAccumulator,
    show_entity_filters_yaml: bool,
) -> None:
    if not acc.entities:
        return

    lines.append("Entities:")
    for entity, filters in acc.entities:
        if show_entity_filters_yaml and isinstance(filters, dict) and filters:
            lines.append(f"  - Entity: {entity}")
            lines.append("    Filters:")
            for key, value in filters.items():
                lines.append(f"      {key}: {_format_filter_value(value)}")
            continue

        lines.append(f"  - Entity: {entity}, Filters: {filters if filters else '{}'}")


def _trim_trailing_blank_lines(lines: List[str]) -> None:
    while lines and not lines[-1].strip():
        lines.pop()


def _build_context_lines(
    acc: _SchemaAccumulator,
    title: str,
    max_fields_per_table: int,
    show_entity_filters_yaml: bool,
) -> List[str]:
    lines: List[str] = [title]

    _append_table_lines(lines, acc, max_fields_per_table)

    if acc.joins:
        lines.append("Join:")
        lines.extend(f"  {join}" for join in acc.joins)
        lines.append("")

    _append_metric_lines(lines, acc)
    _append_simple_section(lines, "Periods", acc.periods)
    _append_simple_section(lines, "Currencies", acc.currencies)
    _append_enum_lines(lines, acc)
    _append_entity_lines(lines, acc, show_entity_filters_yaml)

    _trim_trailing_blank_lines(lines)
    return lines


def hits_to_schema_context(
    hits: Union[List[Any], Dict, str],
    title: str = "SCHEMA CONTEXT",
    max_fields_per_table: int = 20,
    sort_sections: bool = True,
    show_entity_filters_yaml: bool = True
) -> str:
    acc = _SchemaAccumulator()
    for txt, md in _collect_docs(hits):
        _process_doc(txt, md, acc)
    if sort_sections:
        acc.sort()
    lines = _build_context_lines(acc, title, max_fields_per_table, show_entity_filters_yaml)
    return "\n".join(lines)


def debug_entity_retriever(q: str,state: SQLState):
    resp = remote_entity_embedder(q)   # this returns {"ok":..., "body":...}
    return {
        "query": q,
        "raw_response": resp,
        "parsed_entity_cards": call_entity_retriever(False, q,state)
    }


def _invoke_pipeline(user_question: str, chat_id: str, request_id: str,sendNonErptoAI: bool = False):
    initial_state: SQLState = {
        "question": user_question or "",
        "session_id": chat_id,
        "request_id": request_id,
        "sendNonErptoAI":sendNonErptoAI
    }
    config = {
        "configurable": {"thread_id": chat_id},
        "run_name": "changai_text2sql_graph",
        "run_type": "graph",
        "tags": ["changai", "rag", "sql"],
        "metadata": {"tenant": "demo"},
    }
    try:
        return app.invoke(initial_state, config=config), None
    except frappe.exceptions.ValidationError as e:
        # clean_msg = re.sub(r'<[^>]+>', '', str(e))
        return None, {"Bot": str(e), "error": str(e)}
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "ChangAI Pipeline Invoke Error")
        return None, {"Bot": "⚠️ An unexpected error occurred. Please try again.", "error": str(e)}


def _handle_non_erp_(non_erp_res,formatted_q,err,user_question: str,request_id:str, chat_id: str) -> Dict:

    if not non_erp_res:
        if err:
            frappe.log_error(err, "ChangAI NON_ERP Error")
            save_logs(user_question=user_question, formatted_q="Not formatted as its NONERP",err=err, result=non_erp_res,type_="NonERP")

        return {
            "Question": user_question,
            "Formatted-Question": formatted_q,
            "Bot": err if err else "⚠️ Could not get a response. Please try again.",
        }
    
    if not err and non_erp_res:
        try:
            publish_pipeline_update(request_id, "mapped to non-erp", "Mapped as NON-ERP")
            save_turn_2(session_id=chat_id, user_text=formatted_q or user_question, bot_text=non_erp_res,type_="non_erp")
            save_logs(user_question=user_question, formatted_q="Not formatted as its NONERP",err="None", result=non_erp_res,type_="NonERP")
        except Exception as e:
            frappe.log_error(f"Failed to save NON_ERP logs: {e}", "ChangAI Logs")
            save_logs(user_question=user_question, formatted_q="Not formatted as its NONERP",err=str(e), result=non_erp_res,type_="NonERP")


    return {"Question": user_question, "Formatted-Question": formatted_q, "Bot": non_erp_res}


def _get_sql_error_message(err: Any, val: Dict) -> str:
    # if err:
    #     frappe.log_error(err, "ChangAI SQL Pipeline Error")
    #     return "⚠️ The model encountered an error generating your query. Please try the same Question again."

    error_text = (val.get("error") or "").strip()

    if not error_text:
        return "⚠️ Could nprocess your request. Please try rephrasing."

    if "Empty SQL from LLM" in error_text:
        return "⚠️ The model could not generate a SQL query for your question. Please try rephrasing."

    if "does not exist in schema" in error_text:
        return f"⚠️ The model generated an invalid table reference. {error_text}"

    if "could not be resolved" in error_text:
        return f"⚠️ The model generated an invalid field reference. {error_text}"

    if "parse" in error_text.lower() or "syntax" in error_text.lower() or "expected" in error_text.lower():
        return "⚠️ The model generated invalid SQL syntax. Please try rephrasing."

    return f"⚠️ The model generated an invalid query. {error_text}"

def _handle_sql_result(
    memory_status: Dict,
    request_id: str,
    sql_prompt: Optional[str],
    final: Optional[Dict],
    sql: str,
    orm: Optional[str],
    formatted_q: Optional[str],
    fields: Optional[str],
    selected_tables: Optional[List],
    val: Optional[Dict],
    entity_debug: Optional[Dict],
    user_question: str,
    chat_id: str
) -> Dict:
    final = final or {}
    val = val or {}
    entity_debug = entity_debug or {}
    selected_tables = selected_tables or []
    fields = fields or ""
    formatted_q = formatted_q or user_question

    try:
        request_id = request_id or final.get("request_id")
        org_sql = final.get("sql") or sql
        extracted_tables = extract_tables_from_sql(sql)
        try:
            sql_result = execute_query(sql, extracted_tables)
        except Exception as e:
            # err = str(e)
            final["error"] = str(e)

        publish_pipeline_update(request_id, "sql_executed", "Query executed")

        entity_words = final.get("entity_words") or []

    except Exception as e:
        final["error"] = str(e)
        return {"ok": False, "error": f"SQL Execution Failed: {e}"}

    context = (final.get("context") or final.get("selected_fields") or fields or "")[:800]
    contains_values = final.get("contains_values") or entity_debug.get("contains_values") or ""
    err = final.get("error")

    formatted_result = format_data(user_question, sql_result)

    publish_pipeline_update(
        request_id,
        "format_data_completed",
        "Completed Formatting Result",
        done=True
    )

    if not err:
        try:
            save_turn_2(
                session_id=chat_id,
                user_text=formatted_q,
                bot_text=formatted_result,
                type_="erp"
            )
        except Exception as e:
            save_logs(
                user_question=user_question,
                formatted_q=formatted_q,
                context=context,
                sql=sql,
                val=val,
                err=str(e),
                result=sql_result,
                formatted_result=formatted_result,
                tables=selected_tables,
                fields=fields,
                entity_debug=entity_debug,
                type_="ERP"
            )
            return {"ok": False, "error": str(e)}
    save_logs(
                user_question=user_question,
                formatted_q=formatted_q,
                context=context,
                sql=sql,
                val=val,
                err=err if err else "",
                result=sql_result,
                formatted_result=formatted_result,
                tables=selected_tables,
                fields=fields,
                entity_debug=entity_debug,
                type_="ERP"
            )
    return {
        "Model returned SQL": org_sql,
        "context": context,
        "entity_words": entity_words,
        "Question": user_question,
        "Formated Question": formatted_q,
        "Cleaned SQL": sql,
        "ORM": orm,
        "Tables": selected_tables,
        "Fields": fields,
        "Entity Values present ?": contains_values,
        "Validation": val,
        "Error": err,
        "result": sql_result,
        "EntityDebug": entity_debug if entity_debug.get("contains_values") else None,
        "Bot": formatted_result,
    }

def retry_sql(sql, error, formatted_q, sql_prompt):
    retry_prompt = SQL_SYS_PROMPT + """

═══ RETRY MODE — STRICT FIX REQUIRED ═══
STEP 1: Read the failed SQL and error message.
STEP 2: Find the broken field/table.
STEP 3: Check SCHEMA CONTEXT — does it exist?
        YES → fix the syntax.
        NO  → remove it, find correct field from SCHEMA CONTEXT.
STEP 4: Verify every remaining field exists in SCHEMA CONTEXT.
STEP 5: Output fixed SQL. NEVER output the same broken SQL again."""

    user_prompt = sql_prompt + f"""

Failed SQL: {sql}
Error: {error}
User Question: {formatted_q}

DO NOT repeat the same SQL.
DO NOT use the field mentioned in the error.
Find the correct field from SCHEMA CONTEXT and fix it."""

    try:
        rewritten = call_gemini(user_prompt, sys_prompt=retry_prompt)
        rewritten_json = json.loads(rewritten)
        retried_sql = clean_sql(rewritten_json.get("sql") or "")
        retried_orm = clean_sql(rewritten_json.get("orm") or "")
    except Exception:
        return "", "", {"ok": False, "error": "Retry failed to parse response"}

    if not retried_sql:
        return "", "", {"ok": False, "error": "Retry returned empty SQL"}

    val_res = validate_sql_schema(retried_sql)
    return retried_sql, retried_orm, val_res


def get_last_thread_message(chat_id: str):
    data = frappe.get_all(
        "ChangAI Chat History",
        filters={"session_id": chat_id},
        fields=["content"],
        order_by="creation asc"
    )

    for row in reversed(data):
        try:
            msg = json.loads(row["content"])
            # human_msg = msg[-2]["human"]
            msg_type = msg[-2]["type"]
            return msg_type

        except Exception:
            pass

    return ""


THREAD_WORDS = [
    # English confirmation
    "yes", "yep", "yeah", "yup", "yes please","list",
    "of course", "sure", "surely", "absolutely",
    "definitely", "certainly", "indeed", "correct", "ofcourse",
    "right", "exactly", "precisely",
    "ok", "okay", "fine", "alright", "go ahead",
    "do it", "show me", "please", "go on",
    "continue", "proceed", "why not",
    "aye", "affirmative", "true", "agreed",
    "hmm", "hm", "umm", "uh", "ah","give",
    "interesting", "i see", "got it", "ok got it",
    "and", "so", "then", "also", "but",
    "what", "how", "when", "who", "where", "why",
    "more", "less", "again", "another", "other",
    "next", "previous", "back", "forward",
    "noted", "understood", "makes sense",
    "okay okay", "fine fine", "sure sure","show",
    # Arabic confirmation
    "نعم", "أجل", "بالتأكيد", "طبعاً", "حسناً",
    "موافق", "صحيح", "بالضبط", "تماماً", "إي",
    "ماشي", "تمام", "أوكي", "يلا", "استمر",
    "كمّل", "واضح", "فاهم", "مفهوم", "اوك",
    # Arabic neutral / continuation
    "و", "ثم", "لكن", "أيضاً", "كذلك",
    "ماذا", "كيف", "متى", "من", "أين", "لماذا",
    "أكثر", "أقل", "مرة أخرى", "التالي", "السابق",
    "حسناً حسناً", "تمام تمام", "مزيد", "غيره",
    # Arabic rejection
    "لا", "لأ", "لا شكراً", "إلغاء", "توقف",
    "اتركه", "مش محتاج", "مو صح", "خطأ",
]

def is_thread_erp(q:str,chat_id:str):
    msg_type = get_last_thread_message(chat_id)
    if msg_type == "erp" and is_erp_query(False,q, THREAD_WORDS,85):
        return True
    else:
        return False

from rapidfuzz import fuzz
@frappe.whitelist()
def find_similar_log_question(new_question:str, threshold: int = 90):
    logs = frappe.get_all(
        "ChangAI Logs",
        fields=["name", "user_question", "sql_generated","rewritten_question","fields","tables","error","entity","result"],
        limit_page_length=500
    )

    best_match = None
    best_score = 0
    for log in logs:
        score = fuzz.token_set_ratio(new_question, log.rewritten_question)

        if score > best_score:
            best_score = score
            best_match = log

    if best_score >= threshold:
        return {
            "matched": True,
            "score": best_score,
            "log_name": best_match.name,
            "question": best_match.user_question,
            "sql": best_match.sql_generated,
            "rewritten_question":best_match.rewritten_question,
            "fields":best_match.fields,
            "tables":best_match.tables,
            "entity_debug":best_match.entity,
            "result":best_match.result,
            "error":best_match.error
        }

    return {
        "matched": False,
        "score": best_score
    }

def _error_response(memory_status, user_question, formatted_q, context,
                    selected_tables, fields, sql, validation,
                    entity_debug, tries, error, err):
    return {
        "Memory Status": memory_status,
        "Question": user_question,
        "Formatted_Question": formatted_q,
        "Context": (context or "")[:800],
        "Tables": selected_tables,
        "Fields": fields,
        "SQL": sql,
        "Validation": validation,
        "EntityDebug": entity_debug,
        "Tries": tries,
        "Error": error,
        "Result": [],
        "Bot": _get_sql_error_message(error, validation),
    }


_WARMUP_COUNT=0
def load_on_startup():
    global _WARMUP_COUNT,_EMBEDDER_INSTANCE, _VS_TABLE, _FULL_FIELDS_VS, _VS_MASTER, _FIELD_DOCS_CACHE, sym_spell, _GEMINI_CLIENT
    _WARMUP_COUNT+=1
    frappe.log_error(
        title=f"ChangAI Warmup called | PID {os.getpid()} | Count {_WARMUP_COUNT}",
        message="load_on_startup triggered"
    )

    # If all are already loaded, skip
    if all([
        _EMBEDDER_INSTANCE is not None,
        _VS_TABLE is not None,
        _FULL_FIELDS_VS is not None,
        _VS_MASTER is not None,
        _FIELD_DOCS_CACHE is not None,
        sym_spell is not None,
        _GEMINI_CLIENT is not None,
    ]):
        frappe.log_error(
            title=f"ChangAI Warmup skipped | PID {os.getpid()}",
            message="Already loaded in this worker"
        )
        return 
    message=f"PID={os.getpid()} | module={__name__} | file={__file__} | loaded={_EMBEDDER_INSTANCE is not None} | id={id(_EMBEDDER_INSTANCE)}"

    try:
        load_non_erp_data()
        get_embedding_engine()
        get_vs()
        load_field_matrix()
        gemini_client()
        get_master_vs()
        _init_keywords()
        config = ChangAIConfig.get()
        get_polly_client(config)
        frappe.log_error(
        title="ChangAI Warmup Completed",
        message=frappe.get_traceback()  # full stack trace
    )
    except Exception as e:
        frappe.log_error(
        title="ChangAI Warmup Failed",
        message=frappe.get_traceback()  # full stack trace
    )
    return message


def _init_keywords():
    global _KEYWORDS_SET, _KEYWORDS_LIST
    if not _KEYWORDS_SET:
        _KEYWORDS_SET = set(kw.lower() for kw in BUSINESS_KEYWORDS)
        _KEYWORDS_LIST = list(_KEYWORDS_SET)
        
        # ✅ pre-warm cache — run every keyword through _word_is_erp at startup
        for kw in _KEYWORDS_LIST:
            _word_is_erp(kw)  # result gets cached — first real request is instant
            


def get_embedding_engine_test():
    global _EMBEDDER_INSTANCE
    import time, os

    before_id = id(_EMBEDDER_INSTANCE)
    before_loaded = _EMBEDDER_INSTANCE is not None

    if _EMBEDDER_INSTANCE is not None:
        return {
            "before_loaded": before_loaded,
            "before_id": before_id,
            "pid": os.getpid(),
            "result": "returned_cached"
        }

    t3 = time.time()
    model_path = _get_model_path()

    _EMBEDDER_INSTANCE = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    return {
        "before_loaded": before_loaded,
        "before_id": before_id,
        "after_loaded": _EMBEDDER_INSTANCE is not None,
        "after_id": id(_EMBEDDER_INSTANCE),
        "pid": os.getpid(),
        "load_time": time.time() - t3,
        "result": "loaded_now"
    }


_NON_ERP_DATA = None
_NON_ERP_QUESTIONS = None
_NON_ERP_RESPONSE_MAP = None


def load_non_erp_data():
    global _NON_ERP_DATA, _NON_ERP_QUESTIONS, _NON_ERP_RESPONSE_MAP

    if _NON_ERP_DATA is not None:
        return _NON_ERP_QUESTIONS, _NON_ERP_RESPONSE_MAP
    try:
        _NON_ERP_DATA = read_asset("non_erp_combined.processed.json")
    except Exception as e:
        frappe.log_error(f"Failed to load NON-ERP data: {e}", "ChangAI NON-ERP Data Load Error")
        _NON_ERP_DATA = []

    _NON_ERP_QUESTIONS = []
    _NON_ERP_RESPONSE_MAP = {}

    for item in _NON_ERP_DATA:
        q = item.get("user_input")
        if not q:
            continue

        _NON_ERP_QUESTIONS.append(q)
        _NON_ERP_RESPONSE_MAP[q] = item.get("response")

    return _NON_ERP_QUESTIONS, _NON_ERP_RESPONSE_MAP


def non_erp_response(non_erp_q: str) -> Optional[str]:
    questions, response_map = load_non_erp_data()
    result = process.extractOne(
        non_erp_q,
        questions,
        scorer=fuzz.WRatio,
        score_cutoff=65
    )
    if not result:
        return {"data":"Hey Iam ChangAI from ERPGulf,iam here to help you with your queries..."}
    matched_q = result[0]
    return {"data": response_map.get(matched_q, "Hey Iam ChangAI from ERPGulf,iam here to help you with your queries...")}


@frappe.whitelist(allow_guest=False)
def run_text2sql_pipeline(user_question: str, chat_id: str, request_id: str, sendNonErptoAI: bool = False) -> Dict:
    memory_status = check_memory_status()
    logs = find_similar_log_question(user_question)
    if logs.get("matched") and logs["error"] == "" and logs.get("type") != "NonERP":
        publish_pipeline_update(request_id, "cache_hit", "Using cached result")
        # return True,logs
        # if logs.get("rewritten_question") == "Not formatted as its NONERP":
            # return _handle_non_erp_(logs.get("result"),logs.get("rewritten_question"),logs.get("error"), user_question, chat_id)
        formatted_q = logs.get("rewritten_question")
        sql = logs.get("sql")
        tables = json.loads(logs.get("tables") or "[]")
        fields =logs.get("fields") or ""
        entity_debug = json.loads(logs.get("entity_debug") or "{}")
        return _handle_sql_result(
    memory_status,
    request_id,
    None,
    {},
    sql,
    None,
    formatted_q,
    fields,
    tables,
    {"ok": True, "from_cache": True},
    entity_debug,
    user_question,
    chat_id
)
    else:
        final, err_response = _invoke_pipeline(user_question, chat_id, request_id, sendNonErptoAI)
        if err_response:
            return err_response

        entity_debug = {
            "contains_values": final.get("contains_values"),
            "entity_cards": final.get("entity_cards") or [],
        }

        if (final.get("query_type") or "NON_ERP") == "NON_ERP":     
            non_erp_res = _safe_strip(final.get("non_erp_res", ""))
            formatted_q = _safe_strip(final.get("formatted_q", ""))
            err = final.get("error")
            return _handle_non_erp_(non_erp_res,formatted_q,err, user_question,request_id, chat_id)

        sql = clean_sql(final.get("sql")) or ""
        orm = clean_sql(final.get("orm") or "")
        formatted_q = _safe_strip(final.get("formatted_q") or "")
        if final.get("create_entity") is True:
            return {
                "create_entity":True,
                "doc": final.get("doc"),
                "entity_name": final.get("entity_name")
            }
        if final.get("open_report") is True:
            return {
                "open_report": True,
                "report_name": final.get("report_name"),
                "filters": final["filters"],
                "reports_filter_before_call": final.get("reports_filter_before_call"),
                "entity_raw": final.get("entity_raw"),
                "question_rewritten": formatted_q
            }
        if final.get("stop_followup"):
            save_turn_2(session_id=chat_id, user_text=user_question, bot_text=final.get("message"),type_="non_erp")
            save_logs(
                user_question=user_question,
                formatted_q=None,
                context=None,
                sql=None,
                val=None,
                err=err if err else "",
                result=None,
                formatted_result=final.get("message"),
                tables=None,
                fields=None,
                entity_debug=None,
                type="NonERP"
            )
            publish_pipeline_update(
    request_id,
    "Stop follow-up detected",
    "Stop follow-up detected",
    data={"message": final.get("message")},
    done=True
)
            return {
        "Model returned SQL": None,
        "context": None,
        "entity_words": None,
        "Question": user_question,
        "Formated Question": formatted_q,
        "Cleaned SQL": None,
        "stop_followup": True,
        "ORM": None,
        "Tables": None,
        "Fields": None,
        "Entity Values present ?": None,
        "Validation": None,
        "Error": err,
        "result": fNone,
        "EntityDebug": None,
        "Bot": final.get("message"),
    }
        selected_tables = final.get("selected_tables") or []
        fields = _safe_strip(final.get("selected_fields") or "")
        sql_prompt = _safe_strip(final.get("sql_prompt") or "")
        final_prompt = final.get("final_prompt") or ""
        entity_words=final.get("entity_words") or []
        try:
            context = final.get("context")
        except Exception as e:
            frappe.log_error(e, "Error occurred while fetching final values")
        err = final.get("error")

        # # # guard empty sql
        # # # if not sql:
        # # #     return _error_response(memory_status, user_question, formatted_q, context,
        # # #                            selected_tables, fields, sql, 
        # # #                            {"ok": False, "error": "SQL is empty"},
        # # #                            entity_debug, 0, "SQL not valid or missing", err)
        # retried_sql1, retried_orm1, retry1_val_res = retry_sql(retried_sql, retry_val_res.get("error"), formatted_q, sql_prompt)
        # if retry1_val_res.get("ok"):
        #     return _handle_sql_result(memory_status,request_id, sql_prompt, final, retried_sql1, retried_orm1,
        #                               formatted_q, fields, selected_tables, retry1_val_res,
        #                               entity_debug, user_question, chat_id)
        res = validate_sql_schema(sql)
        publish_pipeline_update(request_id, "sql_validated", _("SQL validation Completed"))

        # valid on first try
        if res.get("ok") and sql.upper().startswith("SELECT"):
            return _handle_sql_result(memory_status,request_id, sql_prompt, final, sql, orm,
                                    formatted_q, fields, selected_tables, res,
                                    entity_debug, user_question, chat_id)

        # retry 2
        retried_sql2, retried_orm2, retry2_val_res = retry_sql(sql, res.get("error"), formatted_q, sql_prompt)
        if retry2_val_res.get("ok"):
            return _handle_sql_result(memory_status,request_id, sql_prompt, final, retried_sql2, retried_orm2,
                                    formatted_q, fields, selected_tables, retry2_val_res,
                                    entity_debug, user_question, chat_id)

        # retry 3
        retried_sql3, retried_orm3, retry3_val_res = retry_sql(retried_sql2, retry2_val_res.get("error"), formatted_q, sql_prompt)
        if retry3_val_res.get("ok"):
            return _handle_sql_result(memory_status,request_id, sql_prompt, final, retried_sql3, retried_orm3,
                                    formatted_q, fields, selected_tables, retry3_val_res,
                                    entity_debug, user_question, chat_id)

        # all retries failed
        final_error = retry2_val_res.get("error")  or retry3_val_res.get("error") or res.get("error") or "SQL not valid or missing"
        return _error_response(memory_status, user_question, formatted_q, context,
                            selected_tables, fields, retried_sql2 or sql,
                            retry2_val_res, entity_debug, 2, final_error, err)

from frappe.desk.query_report import get_script
def get_report_filter_fields(report_name: str):
    try:
        script = get_script(report_name).get("script") or ""
    except Exception:
        return []

    fieldnames = re.findall(
        r'fieldname\s*:\s*["\']([^"\']+)["\']',
        script
    )

    return "|".join(dict.fromkeys(fieldnames))


def match_report_intent(report_intent: str):
    choices = list(REPORT_INTENT_MAP.keys())
    match = process.extractOne(
        report_intent.lower(),
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=75
    )
    if not match:
        return ""

    matched_intent = match[0]
    return REPORT_INTENT_MAP[matched_intent]


