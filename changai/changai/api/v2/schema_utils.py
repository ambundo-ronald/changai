import sqlglot
from sqlglot import exp
from sqlglot import optimizer
from sqlglot.schema import MappingSchema
import frappe
from sqlglot.errors import ParseError, OptimizeError
from sqlglot.optimizer.qualify import qualify
import json
from typing import Any, Dict, List, Tuple, Union, Optional, Set
import yaml
from frappe.utils import getdate
from pathlib import Path

def _safe_join(base: Path, rel: str) -> Path:
    """
    Prevent path traversal. Only allow reading inside base directory.
    """
    p = (base / rel).resolve()
    if base != p and base not in p.parents:
        frappe.throw(_("Unsafe path: {0}").format(rel))
    return p

_ALLOWED_EXT = {".json", ".yaml",".j2", ".yml", ".txt", ".md"}
_ASSETS_DIR = Path(frappe.get_app_path("changai", "changai", "api", "v2", "assets")).resolve()
_PROMPTS_DIR = Path(frappe.get_app_path("changai", "changai", "prompts")).resolve()
RAG_FOLDER = "Home/RAG Sources"
JSON_EXT = ".json"
YAML_EXT = ".yaml"
def _get_file_doc_by_name(file_name: str, folder: str = RAG_FOLDER) -> Optional["frappe.model.document.Document"]:
    file_id = frappe.db.get_value("File", {"file_name": file_name, "folder": folder}, "name")
    if not file_id:
        return None
    return frappe.get_doc("File", file_id)


def _read_filedoctype(file_name: str, folder: str = RAG_FOLDER):
    doc = _get_file_doc_by_name(file_name, folder)
    if not doc:
        if file_name.endswith(JSON_EXT):
            return []
        if file_name.endswith((YAML_EXT, ".yml")):
            return {}
        return ""
    raw = doc.get_content() or ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if file_name.endswith(JSON_EXT):
        return json.loads(raw or "[]")
    if file_name.endswith((YAML_EXT, ".yml")):
        obj = yaml.safe_load(raw) or {}
        return obj if isinstance(obj, dict) else {}
    return raw


def read_asset(file_name: str, base: str = "assets") -> Any:
    """
    base:
      - "assets"  -> changai/changai/api/v2/assets
      - "prompts" -> changai/changai/prompts
    """
    file_name = (file_name or "").strip()
    if not file_name:
        frappe.throw(_("file_name is required"))

    ext = Path(file_name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        frappe.throw(_("Unsupported file type: {0}").format(ext))

    if base == "assets":
        root = _ASSETS_DIR
    elif base == "prompts":
        root = _PROMPTS_DIR
    else:
        root = None
    if root is None:
        frappe.throw(_("Invalid base: {0}").format(base))

    path = _safe_join(root, file_name)

    if not path.is_file():
        frappe.throw(_("File not found: {0}").format(str(path)))

    content = path.read_text(encoding="utf-8", errors="replace")

    if ext == ".json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            frappe.throw(_("Invalid JSON in {0}: {1}").format(str(path), str(e)))
    if ext == ".yaml" or ext == ".yml":
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            frappe.throw(_("Invalid YAML in {0}: {1}").format(str(path), str(e)))
    return content

def _load_mapping_data() -> dict:
    return read_asset("metaschema_clean_v2.json")

@frappe.whitelist()
def validate_sql_schema(sql: str, dialect: str = "mysql") -> dict:
    try:
        mapping_data, schema = get_mapping_schema(dialect)

        ast = sqlglot.parse_one(sql, read=dialect)
        used_tables = {table.name for table in ast.find_all(exp.Table)}
        small_mapping = {
            table: mapping_data[table]
            for table in used_tables
            if table in mapping_data
        }

        for table in ast.find_all(exp.Table):
            if table.name and table.name not in mapping_data:
                return {
                    "ok": False,
                    "error": f"Table '{table.name}' does not exist in schema"
                }

        qualified = optimizer.qualify.qualify(
            ast,
            schema=small_mapping,
            dialect=dialect,
            identify=False,
        )

        return {
            "ok": True,
            "qualified_sql": qualified.sql()
        }

    except sqlglot.errors.OptimizeError as e:
        return {"ok": False, "error": str(e)}
    except sqlglot.errors.ParseError as e:
        return {"ok": False, "error": str(e)}

from frappe.utils import add_to_date, today, date_diff, days_diff
MASTER_DOCTYPES = [
    "Customer",
    "Supplier",
    "Item",
    "Warehouse",
    "Company",
    "Account"
]

@frappe.whitelist(allow_guest=False)
def check_file_updates(file_name=None):
    settings = frappe.get_single("ChangAI Settings")

    if file_name == "master_data.yaml":
        last_sync = settings.last_masterdata_sync
    elif file_name == "schema.yaml":
        last_sync = settings.last_schema_sync
    else:
        frappe.throw("Invalid file_name")

    if not last_sync:
        return {
            "update_status": False,
            "data": True,
            "days": 0,
            "last_sync": None
        }

    if file_name == "schema.yaml":
        changed = frappe.db.exists(
            "DocType",
            {
                "modified": [">", last_sync]
            }
        )

    elif file_name == "master_data.yaml":
        changed = False
        for doc in MASTER_DOCTYPES:
            if frappe.db.exists(doc, {"modified": [">", last_sync]}):
                changed = True
                break

    days = days_diff(today(), getdate(last_sync))

    return {
        "update_status": not bool(changed),
        "data": True,
        "days": days,
        "last_sync": last_sync
    }


@frappe.whitelist()
def reload_mapping_schema_cache():
    global _MAPPING_DATA, _MAPPING_SCHEMA
    _MAPPING_DATA = None
    _MAPPING_SCHEMA = None
    get_mapping_schema()
    return {"ok": True}


_MAPPING_DATA = None
_MAPPING_SCHEMA = None


def get_mapping_schema(dialect="mysql"):
    global _MAPPING_DATA, _MAPPING_SCHEMA

    if _MAPPING_DATA is None:
        mapping_data = _load_mapping_data()
        _MAPPING_DATA = {
            table: columns
            for table, columns in mapping_data.items()
            if table and table.strip() and columns
        }

    if _MAPPING_SCHEMA is None:
        _MAPPING_SCHEMA = MappingSchema(_MAPPING_DATA, dialect=dialect)

    return _MAPPING_DATA, _MAPPING_SCHEMA

@frappe.whitelist()
def convert_yaml_schema_to_sqlglot_meta() -> dict:
    try:
        FRAPPE_GENERIC_FIELDS = {
            "name": "TEXT",
            "owner": "TEXT",
            "creation": "TEXT",
            "modified": "TEXT",
            "modified_by": "TEXT",
            "docstatus": "INT",
            "idx": "INT",
            "parent": "TEXT",
            "parentfield": "TEXT",
            "parenttype": "TEXT",
        }
        data = _read_filedoctype("schema.yaml")
        meta = {}
        for table_entry in data.get("tables", []):
            table_name = table_entry.get("table")
            fields = table_entry.get("fields", [])
            if table_name and fields:
                meta[table_name] = {
                    field["name"]: "TEXT"
                    for field in fields
                    if field.get("name")
                }
                meta[table_name].update(FRAPPE_GENERIC_FIELDS)

        output_path = _ASSETS_DIR / "metaschema_clean_v2.json"
        output_path.write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8"
        )
        reload_mapping_schema_cache()

        return {
            "ok": True,
            "message": "Successfully updated MetaSchema for Validation"
        }
    except Exception as e:
        return {
            "ok": False,
            "message": str(e)
        }
    
from frappe import _
@frappe.whitelist(allow_guest=False)
def test():
        res=check_file_updates("master_data.yaml")
        if not res.get("update_status"):
            frappe.throw(_("Please update master data for entity recognition to work. Click on Update Master Data button in Training tab in ChangAI Settings.<br>Check Quick Start Guide Here 👇"))
