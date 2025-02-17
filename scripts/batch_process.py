# Batch processing example script for DeepPowers tokenizer
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import deeppowers as dp
from tqdm import tqdm

def process_file(tokenizer: dp.Tokenizer, 
                input_file: Path,
                output_file: Path,
                batch_size: int = 32,
                add_special_tokens: bool = True) -> Dict[str, Any]:
    """Process a single file and return statistics."""
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    total_tokens = 0
    total_texts = len(texts)
    max_sequence_length = 0
    token_frequency: Dict[int, int] = {}
    
    # Process in batches
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {input_file.name}"):
        batch = texts[i:i + batch_size]
        token_ids = tokenizer.encode_batch_parallel(batch, add_special_tokens)
        
        # Collect statistics
        for sequence in token_ids:
            total_tokens += len(sequence)
            max_sequence_length = max(max_sequence_length, len(sequence))
            for token_id in sequence:
                token_frequency[token_id] = token_frequency.get(token_id, 0) + 1
        
        results.extend(token_ids)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f)
    
    # Return statistics
    return {
        'total_texts': total_texts,
        'total_tokens': total_tokens,
        'average_length': total_tokens / total_texts,
        'max_length': max_sequence_length,
        'unique_tokens': len(token_frequency),
        'token_frequency': token_frequency
    }

def process_directory(tokenizer: dp.Tokenizer,
                     input_dir: Path,
                     output_dir: Path,
                     batch_size: int = 32,
                     file_pattern: str = "*.txt") -> None:
    """Process all matching files in a directory."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    all_stats = []
    for input_file in input_dir.glob(file_pattern):
        output_file = output_dir / f"{input_file.stem}.json"
        stats = process_file(tokenizer, input_file, output_file, batch_size)
        stats['file_name'] = input_file.name
        all_stats.append(stats)
    
    # Generate summary report
    total_texts = sum(s['total_texts'] for s in all_stats)
    total_tokens = sum(s['total_tokens'] for s in all_stats)
    max_length = max(s['max_length'] for s in all_stats)
    
    # Merge token frequencies
    merged_frequency: Dict[int, int] = {}
    for stats in all_stats:
        for token_id, freq in stats['token_frequency'].items():
            merged_frequency[token_id] = merged_frequency.get(token_id, 0) + freq
    
    # Save summary report
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(all_stats),
            'total_texts': total_texts,
            'total_tokens': total_tokens,
            'average_tokens_per_text': total_tokens / total_texts,
            'max_sequence_length': max_length,
            'unique_tokens': len(merged_frequency),
            'token_frequency': merged_frequency,
            'file_stats': all_stats
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Batch process texts with DeepPowers tokenizer')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer model')
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--pattern', default='*.txt', help='File pattern when input is directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return 1
    
    # Initialize tokenizer
    tokenizer = dp.Tokenizer()
    tokenizer.load(args.tokenizer)
    
    if input_path.is_file():
        # Process single file
        output_path.mkdir(parents=True, exist_ok=True)
        stats = process_file(tokenizer, input_path, 
                           output_path / f"{input_path.stem}.json",
                           args.batch_size)
        
        # Save statistics
        with open(output_path / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    else:
        # Process directory
        process_directory(tokenizer, input_path, output_path, 
                        args.batch_size, args.pattern)
    
    return 0

if __name__ == '__main__':
    main() 