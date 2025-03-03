#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script: Demonstrates how to load models, optimize them and run benchmarks
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

import deeppowers as dp
from deeppowers import Model, Tokenizer, Pipeline

def print_dict(d: Dict[str, Any], indent: int = 0) -> None:
    """Print dictionary content, supports nested dictionaries"""
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Optimize models and run benchmarks")
    parser.add_argument("--model", type=str, default="deepseek-v3", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type (float32, float16, int8, int4)")
    parser.add_argument("--optimization", type=str, default="auto", 
                        choices=["auto", "fusion", "pruning", "quantization", "caching", "none"],
                        help="Optimization type")
    parser.add_argument("--level", type=str, default="o1", 
                        choices=["o1", "o2", "o3", "none"],
                        help="Optimization level")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--benchmark-text", type=str, default="This is a sample text for benchmarking. It should be long enough to test the model's performance.", 
                        help="Benchmark text")
    parser.add_argument("--benchmark-runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--quantize", action="store_true", help="Quantize model")
    parser.add_argument("--quantize-precision", type=str, default="int8", 
                        choices=["int8", "int4", "mixed"],
                        help="Quantization precision")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--prompt", type=str, default="Hello, please introduce yourself.", help="Prompt for text generation")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not dp.cuda_available():
        print("Warning: CUDA is not available, falling back to CPU")
        args.device = "cpu"
    
    # Print system information
    print("System information:")
    print(f"  DeepPowers version: {dp.version()}")
    print(f"  CUDA version: {dp.cuda_version()}")
    print(f"  CUDA available: {dp.cuda_available()}")
    print(f"  CUDA device count: {dp.cuda_device_count()}")
    print()
    
    # List available models
    print("Available models:")
    for model_name in dp.list_available_models():
        print(f"  - {model_name}")
    print()
    
    # Load model
    print(f"Loading model '{args.model}'...")
    start_time = time.time()
    model = dp.load_model(args.model, device=args.device, dtype=args.dtype)
    load_time = time.time() - start_time
    print(f"Model loaded, time taken: {load_time:.2f} seconds")
    print(f"   Model type: {model.model_type}")
    print(f"   Device: {model.device}")
    print(f"   Vocab size: {model.vocab_size}")
    print(f"   Max sequence length: {model.max_sequence_length}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer(args.model)
    
    # Set tokenizer
    model.set_tokenizer(tokenizer)
    
    # Optimize model
    if args.optimization != "none":
        print(f"Optimizing model (type: {args.optimization}, level: {args.level})...")
        start_time = time.time()
        results = dp.optimize_model(model, args.optimization, args.level, enable_profiling=True)
        optimize_time = time.time() - start_time
        print(f"Optimization completed, time taken: {optimize_time:.2f} seconds")
        print("Optimization results:")
        print_dict(results, 2)
        print()
    
    # Quantize model
    if args.quantize:
        print(f"Quantizing model (precision: {args.quantize_precision})...")
        start_time = time.time()
        results = dp.quantize_model(model, args.quantize_precision)
        quantize_time = time.time() - start_time
        print(f"Quantization completed, time taken: {quantize_time:.2f} seconds")
        print("Quantization results:")
        print_dict(results, 2)
        print()
    
    # Benchmark
    if args.benchmark:
        print("Running benchmark...")
        print(f"   Test text: '{args.benchmark_text}'")
        print(f"   Number of runs: {args.benchmark_runs}")
        print(f"   Number of warmup runs: {args.warmup_runs}")
        
        results = dp.benchmark_model(
            model, 
            input_text=args.benchmark_text, 
            num_runs=args.benchmark_runs,
            warmup_runs=args.warmup_runs
        )
        
        print("Benchmark results:")
        print_dict(results, 2)
        print()
    
    # Generate text
    if args.generate:
        print(f"Generating text (prompt: '{args.prompt}')...")
        
        # Create stream callback
        def stream_callback(result):
            # Print generated text
            print(result.texts[0], end="", flush=True)
            # Continue generating
            return True
        
        # Create generation config
        config = dp.GenerationConfig(
            max_length=args.max_length,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        # Stream generation
        print("Generation results:")
        model.generate_stream(args.prompt, stream_callback, config)
        print("\n")
    
    print("Completed!")

if __name__ == "__main__":
    main() 