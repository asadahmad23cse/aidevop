"""
ollama_client.py — LLM API Client (Ollama + Optional Cloud API Key Fallback)
=============================================================================
Supports:
  1. Ollama (Local open-source, default for Exercise 1 & 3: Code Llama)
  2. Optional API Key Providers (OpenAI, Groq) if an API key is provided
  3. Grounded Fallback if neither is configured
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def check_ollama_status(host: str = OLLAMA_HOST) -> Dict[str, Any]:
    """Check if Ollama service is reachable, or if Cloud API key is available."""
    # 1. Check local Ollama
    try:
        url = f"{host.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            return {
                "available": True,
                "provider": "ollama",
                "host": host,
                "models": models,
                "default_model": DEFAULT_MODEL,
                "has_default_model": any(DEFAULT_MODEL in m for m in models)
            }
    except Exception:
        pass

    # 2. Check OpenAI API key fallback
    if OPENAI_API_KEY:
        return {
            "available": True,
            "provider": "openai",
            "host": "api.openai.com",
            "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
            "default_model": "gpt-4o-mini",
            "has_default_model": True
        }

    # 3. Check Groq API key fallback
    if GROQ_API_KEY:
        return {
            "available": True,
            "provider": "groq",
            "host": "api.groq.com",
            "models": ["llama-3.1-8b-instant"],
            "default_model": "llama-3.1-8b-instant",
            "has_default_model": True
        }

    return {
        "available": False,
        "provider": "none",
        "host": host,
        "models": [],
        "default_model": DEFAULT_MODEL,
        "has_default_model": False,
        "message": "Ollama not running locally. (To use cloud LLM, set $env:OPENAI_API_KEY or $env:GROQ_API_KEY)"
    }


def query_cloud_api(prompt: str, system_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Query OpenAI or Groq if keys are set."""
    t0 = time.time()
    if OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.2}
            resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "response": data["choices"][0]["message"]["content"].strip(),
                    "model": "gpt-4o-mini (Cloud API)",
                    "latency_s": round(time.time() - t0, 3)
                }
        except Exception:
            pass

    if GROQ_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {"model": "llama-3.1-8b-instant", "messages": messages, "temperature": 0.2}
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "response": data["choices"][0]["message"]["content"].strip(),
                    "model": "llama-3.1-8b-instant (Groq API)",
                    "latency_s": round(time.time() - t0, 3)
                }
        except Exception:
            pass

    return None


def query_ollama(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.2,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Send prompt to Ollama /api/generate endpoint (or Cloud API key if Ollama is not running).
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    if system_prompt:
        payload["system"] = system_prompt

    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "response": data.get("response", "").strip(),
                "model": model,
                "latency_s": round(elapsed, 3),
                "eval_count": data.get("eval_count", 0),
            }
    except Exception:
        pass

    # Fallback to Cloud API Key if configured
    cloud_res = query_cloud_api(prompt, system_prompt)
    if cloud_res:
        return cloud_res

    return {
        "success": False,
        "error": f"Ollama not reachable at {host}. Start Ollama with 'ollama serve' or set OPENAI_API_KEY.",
        "latency_s": round(time.time() - t0, 3),
        "model": model
    }


def generate_with_rag(
    query: str,
    context_chunks: list,
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST
) -> Dict[str, Any]:
    """
    Exercise 3: Construct RAG context prompt and query Ollama / Code Llama (or Cloud API).
    """
    context_text = "\n\n".join([
        f"[Context {i+1}] (Score: {chunk.get('score', 0):.4f})\n{chunk.get('text', '')}"
        for i, chunk in enumerate(context_chunks)
    ])

    system_prompt = (
        "You are an expert medical AI assistant. Answer the user's question accurately "
        "and concisely using ONLY the provided verified context. If the context does not contain "
        "the answer, state clearly that the verified knowledge base does not have this information."
    )

    prompt = (
        f"### Retrieved Context:\n{context_text}\n\n"
        f"### User Question:\n{query}\n\n"
        f"### Answer:"
    )

    result = query_ollama(prompt=prompt, system_prompt=system_prompt, model=model, host=host)
    result["augmented_prompt"] = prompt
    result["chunks_count"] = len(context_chunks)
    return result
