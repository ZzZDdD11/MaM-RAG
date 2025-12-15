# tools/ingest_texts.py
import argparse
import asyncio
import inspect
import json
import os
from typing import List, Optional, Iterable, Dict, Any

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
try:
    # Some LightRAG versions expose EmbeddingFunc; some don't
    from lightrag.utils import EmbeddingFunc  # type: ignore
except Exception:
    EmbeddingFunc = None  # type: ignore
from json import JSONDecodeError


def _sanitize_doc_status(working_dir: str):
    """If kv_store_doc_status.json is corrupted, back it up and reset to {}."""
    path = os.path.join(working_dir, "kv_store_doc_status.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        # backup and reset
        try:
            bak = path + ".bak"
            try:
                os.replace(path, bak)
            except Exception:
                pass
            with open(path, "w", encoding="utf-8") as w:
                w.write("{}")
            print(f"[SANITIZE] reset corrupted {os.path.basename(path)} (backup saved as {os.path.basename(bak)})")
        except Exception:
            print(f"[SANITIZE] failed to reset {os.path.basename(path)}: {e}")


def _norm_str(s: str) -> str:
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    return " ".join(s.split()).strip()


def _get_path(obj: Any, dotted_key: str) -> Optional[Any]:
    cur = obj
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first_text(obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
    # Try given keys in order; support dotted paths (a.b.c)
    for k in keys:
        val = _get_path(obj, k)
        if isinstance(val, str) and val.strip():
            return _norm_str(val)
        if isinstance(val, list):
            parts = [x for x in val if isinstance(x, str) and x.strip()]
            if parts:
                joined = _norm_str(" ".join(parts))[:4000]
                if joined:
                    return joined
    return None


def _concat_fields(obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
    if not keys:
        return None
    parts: List[str] = []
    for k in keys:
        val = _get_path(obj, k)
        if isinstance(val, str) and val.strip():
            parts.append(_norm_str(val))
    if parts:
        return ("\n".join(parts))[:4000]
    return None


def yield_texts(path: str, max_docs: int, keys: List[str], concat_keys: List[str]) -> Iterable[str]:
    cnt = 0
    skipped = 0

    def _emit(txt: Optional[str]):
        nonlocal cnt, skipped
        if txt and isinstance(txt, str) and txt.strip():
            yield _norm_str(txt)[:8000]  # safety cap
            cnt += 1
        else:
            skipped += 1

    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if cnt >= max_docs:
                    break
                line = line.strip()
                if not line:
                    skipped += 1
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                txt = _first_text(obj, keys)
                if not txt:
                    txt = _concat_fields(obj, concat_keys)
                yield from _emit(txt)
    else:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                print(f"[ERROR] Failed to load JSON: {path}")
                return
        if isinstance(data, list):
            for obj in data:
                if cnt >= max_docs:
                    break
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                txt = _first_text(obj, keys)
                if not txt:
                    txt = _concat_fields(obj, concat_keys)
                yield from _emit(txt)
        elif isinstance(data, dict):
            txt = _first_text(data, keys) or _concat_fields(data, concat_keys)
            yield from _emit(txt)
        else:
            print(f"[WARN] Unsupported JSON structure at root: {type(data)}")

    if skipped:
        print(f"[WARN] skipped {skipped} records without matched text fields")


def init_lightrag(working_dir: str, llm_name: str = "qwen2.5:7b") -> LightRAG:
    lr_kwargs = {"working_dir": working_dir}
    sig = inspect.signature(LightRAG)

    # Avoid OpenAI; use local Ollama
    if "llm_model_func" in sig.parameters:
        lr_kwargs["llm_model_func"] = ollama_model_complete
    if "llm_model_name" in sig.parameters:
        lr_kwargs["llm_model_name"] = llm_name

    # Prefer new-style embedding_function (统一为 bge-m3 / 1024 维)
    if "embedding_function" in sig.parameters:
        lr_kwargs["embedding_function"] = lambda texts: ollama_embed(
            texts, embed_model="bge-m3", host="http://localhost:11434"
        )
        if "embedding_dim" in sig.parameters:
            lr_kwargs["embedding_dim"] = 1024
    elif "embedding_func" in sig.parameters and EmbeddingFunc is not None:
        lr_kwargs["embedding_func"] = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="bge-m3", host="http://localhost:11434"
            ),
        )

    # Initialize
    try:
        rag = LightRAG(**lr_kwargs)
    except TypeError:
        rag = LightRAG(working_dir=working_dir)

    # storages must be initialized for pipeline ops in newer versions
    async def _init():
        await rag.initialize_storages()
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()

    try:
        asyncio.run(_init())
    except (RuntimeError, JSONDecodeError) as e:
        # Try sanitize doc_status when JSONDecodeError happens inside init
        if isinstance(e, JSONDecodeError) or "JSONDecodeError" in str(e):
            _sanitize_doc_status(working_dir)
        # Retry with a fresh loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_init())
        finally:
            loop.close()

    return rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--working_dir", required=True, help="LightRAG working directory")
    ap.add_argument("--input", required=True, help="Path to .jsonl or .json")
    ap.add_argument("--max_docs", type=int, default=200, help="Max docs to ingest")
    ap.add_argument("--keys", default="text,content,summary", help="Comma-separated candidate fields, supports dotted paths (e.g., meta.abstract)")
    ap.add_argument("--concat_keys", default="", help="Comma-separated fields to concatenate if none of --keys hit")
    ap.add_argument("--llm_name", default="qwen2.5:7b", help="Ollama model name for LLM ops if needed")
    args = ap.parse_args()

    print(f"[INGEST] working_dir={args.working_dir}")
    rag = init_lightrag(args.working_dir, args.llm_name)

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    concat_keys = [k.strip() for k in args.concat_keys.split(",") if k.strip()]

    n = 0
    for txt in yield_texts(args.input, args.max_docs, keys, concat_keys):
        rag.insert(txt)
        n += 1
        if n % 50 == 0:
            print(f"[INGEST] inserted {n} docs")

    print(f"[DONE] Inserted {n} docs into {args.working_dir}")


if __name__ == "__main__":
    main()