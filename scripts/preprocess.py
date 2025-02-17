# Data preprocessing script for DeepPowers tokenizer
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import unicodedata
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def setup_logging(log_file: Path) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class TextCleaner:
    """Text cleaning and normalization class."""
    
    def __init__(self):
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
        )
        self.email_pattern = re.compile(
            r'[\w\.-]+@[\w\.-]+\.\w+'
        )
        self.number_pattern = re.compile(
            r'\d+(?:\.\d+)?'
        )
        self.whitespace_pattern = re.compile(
            r'\s+'
        )
        self.control_char_pattern = re.compile(
            r'[\x00-\x1f\x7f-\x9f]'
        )
    
    def clean_text(self, text: str, 
                  normalize_unicode: bool = True,
                  remove_urls: bool = True,
                  remove_emails: bool = True,
                  normalize_numbers: bool = True,
                  remove_control_chars: bool = True) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove control characters
        if remove_control_chars:
            text = self.control_char_pattern.sub(" ", text)
        
        # Normalize Unicode
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Replace URLs
        if remove_urls:
            text = self.url_pattern.sub("[URL]", text)
        
        # Replace email addresses
        if remove_emails:
            text = self.email_pattern.sub("[EMAIL]", text)
        
        # Normalize numbers
        if normalize_numbers:
            text = self.number_pattern.sub("[NUM]", text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(" ", text)
        
        return text.strip()
    
    def is_valid_text(self, text: str, 
                     min_length: int = 10,
                     max_length: int = 1000) -> bool:
        """Check if text is valid for training."""
        if not text or len(text) < min_length or len(text) > max_length:
            return False
        
        # Check text quality (can be customized)
        words = text.split()
        if len(words) < 3:  # At least 3 words
            return False
        
        return True

def process_file(file_path: Path,
                output_dir: Path,
                cleaner: TextCleaner,
                min_length: int = 10,
                max_length: int = 1000) -> Dict[str, Any]:
    """Process a single file and return statistics."""
    # Read input file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_texts = []
    stats = {
        'total_lines': len(lines),
        'valid_lines': 0,
        'total_chars': 0,
        'total_words': 0,
        'skipped_short': 0,
        'skipped_long': 0
    }
    
    for line in lines:
        # Clean text
        cleaned = cleaner.clean_text(line)
        
        # Validate text
        if cleaner.is_valid_text(cleaned, min_length, max_length):
            processed_texts.append(cleaned)
            stats['valid_lines'] += 1
            stats['total_chars'] += len(cleaned)
            stats['total_words'] += len(cleaned.split())
        else:
            if len(cleaned) < min_length:
                stats['skipped_short'] += 1
            if len(cleaned) > max_length:
                stats['skipped_long'] += 1
    
    # Save processed texts
    output_file = output_dir / f"{file_path.stem}_processed.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_texts))
    
    return stats

def process_directory(input_dir: Path,
                     output_dir: Path,
                     file_pattern: str = "*.txt",
                     min_length: int = 10,
                     max_length: int = 1000,
                     num_workers: int = 4) -> None:
    """Process all matching files in a directory using multiple processes."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / 'preprocessing.log')
    
    # Initialize text cleaner
    cleaner = TextCleaner()
    
    # Get all input files
    input_files = list(input_dir.glob(file_pattern))
    logging.info(f"Found {len(input_files)} files to process")
    
    # Process files in parallel
    process_func = partial(process_file, 
                          output_dir=output_dir,
                          cleaner=cleaner,
                          min_length=min_length,
                          max_length=max_length)
    
    all_stats = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for stats in tqdm(executor.map(process_func, input_files),
                         total=len(input_files),
                         desc="Processing files"):
            all_stats.append(stats)
    
    # Aggregate statistics
    total_stats = {
        'total_files': len(input_files),
        'total_lines': sum(s['total_lines'] for s in all_stats),
        'valid_lines': sum(s['valid_lines'] for s in all_stats),
        'total_chars': sum(s['total_chars'] for s in all_stats),
        'total_words': sum(s['total_words'] for s in all_stats),
        'skipped_short': sum(s['skipped_short'] for s in all_stats),
        'skipped_long': sum(s['skipped_long'] for s in all_stats)
    }
    
    # Log summary
    logging.info("Processing completed. Summary:")
    for key, value in total_stats.items():
        logging.info(f"{key}: {value}")
    
    # Save statistics
    with open(output_dir / 'preprocessing_stats.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_stats': total_stats,
            'file_stats': all_stats
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Preprocess texts for DeepPowers tokenizer')
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--pattern', default='*.txt', help='File pattern when input is directory')
    parser.add_argument('--min-length', type=int, default=10, help='Minimum text length')
    parser.add_argument('--max-length', type=int, default=1000, help='Maximum text length')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return 1
    
    if input_path.is_file():
        # Process single file
        output_path.mkdir(parents=True, exist_ok=True)
        setup_logging(output_path / 'preprocessing.log')
        stats = process_file(input_path, output_path,
                           TextCleaner(),
                           args.min_length,
                           args.max_length)
        
        # Save statistics
        with open(output_path / 'preprocessing_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    else:
        # Process directory
        process_directory(input_path, output_path,
                        args.pattern,
                        args.min_length,
                        args.max_length,
                        args.workers)
    
    return 0

if __name__ == '__main__':
    main() 