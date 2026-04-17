#!/usr/bin/env python3
"""Lightweight local RAG starter for Grok workflows.

This script builds a simple lexical index over local files, retrieves top chunks,
and sends grounded context to an OpenAI-compatible chat completion endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def split_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def iter_files(paths: list[Path], allowed_exts: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in allowed_exts:
            out.append(p)
            continue
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in allowed_exts:
                    out.append(child)
    return out


def build_index(paths: list[Path], allowed_exts: set[str], chunk_size: int, chunk_overlap: int) -> list[dict]:
    entries = []
    for file_path in iter_files(paths, allowed_exts):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, chunk in enumerate(split_chunks(text, chunk_size, chunk_overlap)):
            terms = Counter(tokenize(chunk))
            if not terms:
                continue
            entries.append(
                {
                    "path": str(file_path),
                    "chunk_id": idx,
                    "text": chunk,
                    "terms": terms,
                }
            )
    return entries


def score(query_terms: Counter, entry: dict) -> float:
    s = 0.0
    for term, qn in query_terms.items():
        dn = entry["terms"].get(term, 0)
        if dn:
            s += min(float(qn), float(dn))
    return s


def retrieve(query: str, index: list[dict], top_k: int) -> list[tuple[float, dict]]:
    q_terms = Counter(tokenize(query))
    if not q_terms:
        return []

    scored = []
    for entry in index:
        s = score(q_terms, entry)
        if s > 0:
            scored.append((s, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def call_chat_api(
    api_url: str,
    api_key: str,
    model: str,
    query: str,
    retrieved: list[tuple[float, dict]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    context_blocks = []
    for score_value, item in retrieved:
        context_blocks.append(
            "[Source: {path} | Chunk: {chunk} | Score: {score:.2f}]\n{text}".format(
                path=item["path"],
                chunk=item["chunk_id"],
                score=score_value,
                text=item["text"],
            )
        )
    context = "\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded assistant. Use supplied context first. "
                "If context is missing key facts, say that clearly. "
                "Include source path citations like [path/to/file]."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n{query}\n\n"
                "Context:\n{context}\n\n"
                "Answer in concise bullets with citations."
            ).format(query=query, context=context),
        },
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"API request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected API response format: {body}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RAG starter for Grok workflows")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--corpus", action="append", required=True, help="File or directory to index; repeat for multiple")
    parser.add_argument(
        "--ext",
        default=".md,.txt,.rst,.json,.yaml,.yml,.py,.js,.ts,.tsx,.jsx,.java,.go,.rs,.c,.cpp,.h,.hpp,.html,.css,.csv",
        help="Comma-separated file extensions to include",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")
    parser.add_argument("--api-url", default="https://api.x.ai/v1/chat/completions", help="OpenAI-compatible chat endpoint")
    parser.add_argument("--model", default="grok-2-latest", help="Model name")
    parser.add_argument("--api-key", default=os.environ.get("XAI_API_KEY"), help="API key (or set XAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max output tokens")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunk metadata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("Missing API key: pass --api-key or set XAI_API_KEY", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be > 0", file=sys.stderr)
        return 2

    allowed_exts = set()
    for ext in args.ext.split(","):
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        allowed_exts.add(ext.lower())

    paths = [Path(p).expanduser() for p in args.corpus]
    index = build_index(paths, allowed_exts, args.chunk_size, args.chunk_overlap)
    if not index:
        print("No indexable documents found in corpus paths", file=sys.stderr)
        return 1

    hits = retrieve(args.query, index, args.top_k)
    if not hits:
        print("No relevant chunks found for the query", file=sys.stderr)
        return 1

    if args.show_context:
        print("Retrieved chunks:")
        for score_value, item in hits:
            print(f"- {item['path']}#{item['chunk_id']} score={score_value:.2f}")
        print()

    try:
        answer = call_chat_api(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            query=args.query,
            retrieved=hits,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())