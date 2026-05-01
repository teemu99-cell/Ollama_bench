#!/usr/bin/env python3
"""
ollama_bench.py - Benchmark multiple Ollama models on a set of prompts.

Usage:
    python ollama_bench.py --models llama3.2 mistral:7b --prompts prompts.txt
    python ollama_bench.py --models llama3.2 --prompts prompts.txt --runs 3 --host http://localhost:11434
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def query_model(host: str, model: str, prompt: str) -> dict:
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()

    data = resp.json()
    return {
        "response": data.get("response", ""),
        "total_duration_s": data.get("total_duration", 0) / 1e9,
        "load_duration_s": data.get("load_duration", 0) / 1e9,
        "prompt_eval_duration_s": data.get("prompt_eval_duration", 0) / 1e9,
        "eval_duration_s": data.get("eval_duration", 0) / 1e9,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "response_tokens": data.get("eval_count", 0),
    }


def lexical_diversity(text: str) -> float:
    """Unique tokens / total tokens. Rough proxy for vocabulary variety, not quality."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def get_tokens_per_sec(result: dict) -> float:
    if result["eval_duration_s"] > 0:
        return result["response_tokens"] / result["eval_duration_s"]
    return 0.0


def get_ttft(result: dict) -> float:
    """Time to first token: load time + prompt eval time."""
    return result["load_duration_s"] + result["prompt_eval_duration_s"]


def run_benchmark(host: str, models: list, prompts: list, runs: int) -> list:
    results = []
    total = len(models) * len(prompts) * runs
    done = 0

    for model in models:
        for prompt_idx, prompt in enumerate(prompts):
            run_data = []

            for run in range(runs):
                done += 1
                console.print(
                    f"  [{done}/{total}] {model} | prompt {prompt_idx + 1} | run {run + 1}",
                    style="dim",
                )
                try:
                    r = query_model(host, model, prompt)
                    run_data.append(r)
                except requests.exceptions.ConnectionError:
                    console.print(
                        f"  [red]Connection refused. Is Ollama running at {host}?[/red]"
                    )
                    run_data.append(None)
                except requests.exceptions.HTTPError as e:
                    console.print(f"  [red]HTTP {e.response.status_code} for model '{model}'. Model loaded?[/red]")
                    run_data.append(None)
                except Exception as e:
                    console.print(f"  [red]Unexpected error: {e}[/red]")
                    run_data.append(None)

            valid = [r for r in run_data if r is not None]
            if not valid:
                continue

            n = len(valid)
            results.append({
                "model": model,
                "prompt_idx": prompt_idx + 1,
                "prompt_preview": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "runs_completed": n,
                "latency_s": sum(r["total_duration_s"] for r in valid) / n,
                "ttft_s": sum(get_ttft(r) for r in valid) / n,
                "tokens_per_sec": sum(get_tokens_per_sec(r) for r in valid) / n,
                "response_tokens": sum(r["response_tokens"] for r in valid) / n,
                "lex_diversity": sum(lexical_diversity(r["response"]) for r in valid) / n,
            })

    return results


def print_table(results: list):
    table = Table(
        title="Ollama Benchmark Results",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Prompt", justify="center")
    table.add_column("Latency (s)", justify="right")
    table.add_column("TTFT (s)", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Resp Tokens", justify="right")
    table.add_column("Lex Div", justify="right")

    for r in results:
        table.add_row(
            r["model"],
            f"#{r['prompt_idx']}",
            f"{r['latency_s']:.2f}",
            f"{r['ttft_s']:.2f}",
            f"{r['tokens_per_sec']:.1f}",
            f"{r['response_tokens']:.0f}",
            f"{r['lex_diversity']:.2f}",
        )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Lex Div: lexical diversity (unique/total tokens). "
        "Heuristic only -- not a quality judge. Use an LLM evaluator for that.[/dim]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple Ollama models on a prompt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python ollama_bench.py --models llama3.2 mistral:7b --prompts prompts.txt
  python ollama_bench.py --models llama3.2 --prompts prompts.txt --runs 3 --output results.json
        """,
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="One or more Ollama model names (e.g. llama3.2 mistral:7b phi3:mini)",
    )
    parser.add_argument(
        "--prompts", required=True, type=Path,
        help="Text file with one prompt per line. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Runs per model/prompt pair for averaging (default: 1)",
    )
    parser.add_argument(
        "--host", default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save results as JSON",
    )

    args = parser.parse_args()

    if not args.prompts.exists():
        console.print(f"[red]Prompt file not found: {args.prompts}[/red]")
        sys.exit(1)

    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if not prompts:
        console.print("[red]No prompts found. Check the file format.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Ollama Benchmarker[/bold]")
    console.print(f"  Models  : {', '.join(args.models)}")
    console.print(f"  Prompts : {len(prompts)} loaded from {args.prompts}")
    console.print(f"  Runs    : {args.runs} per combination")
    console.print(f"  Host    : {args.host}\n")

    results = run_benchmark(args.host, args.models, prompts, args.runs)

    if not results:
        console.print("[red]No results collected. Verify model names and Ollama connection.[/red]")
        sys.exit(1)

    print_table(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        console.print(f"\n  Saved to [green]{args.output}[/green]")


if __name__ == "__main__":
    main()
