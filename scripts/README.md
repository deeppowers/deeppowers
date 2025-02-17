# DeepPowers Tokenizer Utility Scripts

This directory contains a collection of utility scripts for the DeepPowers tokenizer.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Script descriptions

### 1. Performance test script (benchmark.py)

Used for testing tokenizer performance, including single text and batch processing mode performance tests.

```bash
python benchmark.py --tokenizer /path/to/tokenizer \
                   --input /path/to/test/file \
                   --batch-size 32 \
                   --iterations 100
```

### 2. Memory monitoring script (memory_monitor.py)

Used to monitor the memory usage of the tokenizer, including statistics on the memory pool and string pool usage.

```bash
python memory_monitor.py --tokenizer /path/to/tokenizer \
                        --input /path/to/test/file \
                        --output /path/to/output \
                        --duration 60
```

### 3. Batch processing example script (batch_process.py)

Shows how to use the tokenizer for large-scale text processing.

```bash
python batch_process.py --tokenizer /path/to/tokenizer \
                       --input /path/to/input/dir \
                       --output /path/to/output/dir \
                       --batch-size 32 \
                       --pattern "*.txt"
```

### 4. Data preprocessing script (preprocess.py)

Used to clean and normalize training data.

```bash
python preprocess.py --input /path/to/input/dir \
                    --output /path/to/output/dir \
                    --pattern "*.txt" \
                    --min-length 10 \
                    --max-length 1000 \
                    --workers 4
```

## Features

1. Performance test
   - Supports single text and batch processing modes
   - Provides detailed performance statistics
   - Includes warm-up phase
   - Supports custom batch size and iteration count

2. Memory monitoring
   - Real-time monitoring of memory usage
   - Generates memory usage reports and charts
   - Analyzes memory pool and string pool efficiency
   - Supports long-term monitoring

3. Batch processing
   - Supports single file and directory processing
   - Provides detailed processing statistics
   - Generates JSON format results
   - Supports custom batch size

4. Data preprocessing
   - Unicode normalization
   - URL and email processing
   - Number normalization
   - Multi-process parallel processing
   - Detailed processing report

## Notes

1. All scripts support command line parameter configuration
2. When processing large files, it is recommended to adjust the batch size
3. Memory monitoring may slightly affect performance
4. Preprocessing supports custom rule extensions