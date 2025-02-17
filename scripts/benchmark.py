# Performance benchmark script for DeepPowers tokenizer
import sys
import time
import argparse
import numpy as np
from pathlib import Path
import deeppowers as dp

def run_benchmark(tokenizer_path, input_file, batch_size=32, num_iterations=100):
    # Load tokenizer
    tokenizer = dp.Tokenizer()
    tokenizer.load(tokenizer_path)
    
    # Load test data
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # Warm up
    print("Warming up...")
    for _ in range(10):
        batch = texts[:batch_size]
        tokenizer.encode_batch_parallel(batch)
    
    # Single text benchmarks
    print("\nSingle text benchmarks:")
    single_times = []
    for _ in range(num_iterations):
        text = texts[0]
        start = time.perf_counter()
        tokenizer.encode(text)
        end = time.perf_counter()
        single_times.append(end - start)
    
    print(f"Average time per text: {np.mean(single_times)*1000:.2f}ms")
    print(f"95th percentile: {np.percentile(single_times, 95)*1000:.2f}ms")
    
    # Batch processing benchmarks
    print("\nBatch processing benchmarks:")
    batch_times = []
    for _ in range(num_iterations):
        batch = texts[:batch_size]
        start = time.perf_counter()
        tokenizer.encode_batch_parallel(batch)
        end = time.perf_counter()
        batch_times.append(end - start)
    
    print(f"Average time per batch: {np.mean(batch_times)*1000:.2f}ms")
    print(f"Average time per text in batch: {np.mean(batch_times)*1000/batch_size:.2f}ms")
    print(f"95th percentile batch time: {np.percentile(batch_times, 95)*1000:.2f}ms")
    
    # Memory usage
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"\nMemory usage: {memory_info.rss / 1024 / 1024:.2f}MB")

def main():
    parser = argparse.ArgumentParser(description='Benchmark DeepPowers tokenizer')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer model')
    parser.add_argument('--input', required=True, help='Input text file for testing')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    
    args = parser.parse_args()
    
    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        sys.exit(1)
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    run_benchmark(args.tokenizer, args.input, args.batch_size, args.iterations)

if __name__ == '__main__':
    main() 