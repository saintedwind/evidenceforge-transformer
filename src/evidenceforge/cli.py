import argparse
import json
import os
from pathlib import Path

from .benchmark import run_benchmark
from .dataset import download_scifact
from .ingest import load_beir_corpus


def main():
    parser = argparse.ArgumentParser(prog="evidenceforge")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download the pinned official SciFact files")
    fetch.add_argument("--data", type=Path, default=Path("data/scifact"))
    demo = commands.add_parser("demo", help="inspect a local SciFact corpus")
    demo.add_argument("--data", type=Path, default=Path("data/scifact"))
    serve = commands.add_parser("serve", help="serve the configured corpus")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1)
    benchmark = commands.add_parser("benchmark", help="score the official test judgments")
    benchmark.add_argument("--data", type=Path, default=Path("data/scifact"))
    benchmark.add_argument("--output", type=Path, default=Path("benchmark.json"))
    benchmark.add_argument("--dense", action="store_true")
    benchmark.add_argument("--download", action="store_true")
    benchmark.add_argument("--model-path", type=Path)
    benchmark.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    if args.command == "fetch":
        print(download_scifact(args.data))
    elif args.command == "benchmark":
        data = download_scifact(args.data) if args.download else args.data
        result = run_benchmark(data, args.output, args.dense, args.model_path, args.threads)
        print(json.dumps(result, indent=2))
    elif args.command == "demo":
        corpus = args.data / "corpus.jsonl"
        docs = load_beir_corpus(corpus)
        print(f"Loaded {len(docs)} documents from {corpus}.")
        print("Set EVIDENCEFORGE_CORPUS to that file and run evidenceforge serve.")
    else:
        import uvicorn

        if args.workers < 1 or (args.workers > 1 and not os.getenv("EVIDENCEFORGE_STATE")):
            parser.error("multiple workers require a shared EVIDENCEFORGE_STATE SQLite file")
        uvicorn.run("evidenceforge.api:app", host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
