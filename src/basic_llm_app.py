"""
basic_llm_app.py — Exercise 1: Basic LLM Application
======================================================
Architecture:
  User → Application → API → Ollama → Code Llama → Response

Usage:
  # Interactive mode:
  python src/basic_llm_app.py

  # Direct prompt mode:
  python src/basic_llm_app.py --prompt "Explain the pathophysiology of Type 2 Diabetes."
  python src/basic_llm_app.py --model codellama --host http://localhost:11434
"""

import sys
import argparse
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ollama_client import query_ollama, check_ollama_status, DEFAULT_MODEL, OLLAMA_HOST


def print_banner():
    print("=" * 65)
    print("  EXERCISE 1: Basic LLM Application")
    print("  Flow: User -> Application -> REST API -> Ollama -> Code Llama")
    print("=" * 65)


def run_interactive(model: str, host: str):
    print_banner()
    status = check_ollama_status(host)
    if status["available"]:
        print(f"✅ Connected to Ollama at {host}")
        print(f"📦 Available models: {', '.join(status['models']) if status['models'] else 'None'}")
        print(f"🎯 Target model: {model}")
    else:
        print(f"⚠️  {status['message']}")
        print("💡 Note: Ensure 'ollama serve' is running and 'ollama pull codellama' is executed.")
    print("-" * 65)
    print("Type your question below (or 'exit' / 'quit' to end):\n")

    while True:
        try:
            query = input("User Question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting Exercise 1 application.")
                break

            print("\n[Application] Sending request to Ollama API...")
            res = query_ollama(prompt=query, model=model, host=host)
            
            if res["success"]:
                print(f"\n--- Code Llama Response (Latency: {res['latency_s']}s) ---")
                print(res["response"])
                print("-" * 65 + "\n")
            else:
                print(f"\n❌ Error from Ollama: {res['error']}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Exercise 1: Basic LLM Application with Ollama & Code Llama")
    parser.add_argument("--prompt", type=str, help="Single prompt to execute and exit")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", type=str, default=OLLAMA_HOST, help=f"Ollama API host URL (default: {OLLAMA_HOST})")
    args = parser.parse_args()

    if args.prompt:
        print_banner()
        print(f"Prompt: {args.prompt}")
        print(f"Model: {args.model} | Host: {args.host}")
        res = query_ollama(prompt=args.prompt, model=args.model, host=args.host)
        if res["success"]:
            print(f"\nResponse ({res['latency_s']}s):\n{res['response']}")
        else:
            print(f"\nError: {res['error']}")
    else:
        run_interactive(model=args.model, host=args.host)


if __name__ == "__main__":
    main()
