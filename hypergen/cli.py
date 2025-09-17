"""Command-line interface for HyperGen."""

import argparse
import json
from pathlib import Path
from typing import Optional

from hypergen import Dataset, Model, serve


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HyperGen - Unified diffusion model framework"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument(
        "model",
        type=str,
        help="Model ID or path to serve",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    serve_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent requests",
    )
    serve_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable automatic optimizations",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication",
    )
    
    generate_parser = subparsers.add_parser("generate", help="Generate images")
    generate_parser.add_argument(
        "model",
        type=str,
        help="Model ID or path",
    )
    generate_parser.add_argument(
        "prompt",
        type=str,
        help="Generation prompt",
    )
    generate_parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output file path",
    )
    generate_parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    generate_parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    generate_parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps",
    )
    generate_parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Guidance scale",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    generate_parser.add_argument(
        "--lora",
        type=str,
        help="LoRA adapter path",
    )
    
    train_parser = subparsers.add_parser("train", help="Train LoRA adapter")
    train_parser.add_argument(
        "model",
        type=str,
        help="Model ID or path",
    )
    train_parser.add_argument(
        "dataset",
        type=str,
        help="Dataset path",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="lora_output",
        help="Output directory",
    )
    train_parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model")
    benchmark_parser.add_argument(
        "model",
        type=str,
        help="Model ID or path",
    )
    benchmark_parser.add_argument(
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Test prompt",
    )
    benchmark_parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    benchmark_parser.add_argument(
        "--compile",
        type=str,
        choices=["none", "regional", "full"],
        default="regional",
        help="Compilation mode",
    )
    
    args = parser.parse_args()
    
    if args.command == "serve":
        serve(
            model=args.model,
            host=args.host,
            port=args.port,
            max_concurrent=args.max_concurrent,
            enable_optimizations=not args.no_optimize,
            api_key=args.api_key,
        )
        
    elif args.command == "generate":
        model = Model.load(
            args.model,
            compile_mode="regional",
        )
        
        image = model.run(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            lora=args.lora,
        )
        
        image.save(args.output)
        print(f"Generated image saved to {args.output}")
        
    elif args.command == "train":
        model = Model.load(args.model)
        dataset = Dataset.load(args.dataset)
        
        lora = model.train_lora(
            dataset=dataset,
            rank=args.rank,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output,
        )
        
        print(f"LoRA training complete. Saved to {lora.path}")
        
    elif args.command == "benchmark":
        print(f"Benchmarking {args.model}")
        print(f"Compilation mode: {args.compile}")
        print(f"Number of runs: {args.runs}")
        print("-" * 50)
        
        model = Model.load(
            args.model,
            compile_mode=args.compile,
        )
        
        results = model.benchmark(
            prompt=args.prompt,
            num_runs=args.runs,
        )
        
        print(f"Mean time: {results['mean_time']:.2f}s")
        print(f"Min time: {results['min_time']:.2f}s")
        print(f"Max time: {results['max_time']:.2f}s")
        print(f"Mean memory: {results['mean_memory_gb']:.2f} GB")
        print(f"Max memory: {results['max_memory_gb']:.2f} GB")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()