#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例脚本：展示如何加载模型、优化模型并进行基准测试
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

import deeppowers as dp
from deeppowers import Model, Tokenizer, Pipeline

def print_dict(d: Dict[str, Any], indent: int = 0) -> None:
    """打印字典内容，支持嵌套字典"""
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="优化模型并进行基准测试")
    parser.add_argument("--model", type=str, default="deepseek-v3", help="模型名称或路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cpu 或 cuda)")
    parser.add_argument("--dtype", type=str, default="float16", help="数据类型 (float32, float16, int8, int4)")
    parser.add_argument("--optimization", type=str, default="auto", 
                        choices=["auto", "fusion", "pruning", "quantization", "caching", "none"],
                        help="优化类型")
    parser.add_argument("--level", type=str, default="o1", 
                        choices=["o1", "o2", "o3", "none"],
                        help="优化级别")
    parser.add_argument("--benchmark", action="store_true", help="进行基准测试")
    parser.add_argument("--benchmark-text", type=str, default="这是一个用于基准测试的示例文本。它应该足够长以测试模型的性能。", 
                        help="基准测试文本")
    parser.add_argument("--benchmark-runs", type=int, default=10, help="基准测试运行次数")
    parser.add_argument("--warmup-runs", type=int, default=3, help="预热运行次数")
    parser.add_argument("--quantize", action="store_true", help="量化模型")
    parser.add_argument("--quantize-precision", type=str, default="int8", 
                        choices=["int8", "int4", "mixed"],
                        help="量化精度")
    parser.add_argument("--generate", action="store_true", help="生成文本")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下自己。", help="生成文本的提示")
    parser.add_argument("--max-length", type=int, default=100, help="生成文本的最大长度")
    
    args = parser.parse_args()
    
    # 检查 CUDA 可用性
    if args.device == "cuda" and not dp.cuda_available():
        print("警告: CUDA 不可用，回退到 CPU")
        args.device = "cpu"
    
    # 打印系统信息
    print("系统信息:")
    print(f"  DeepPowers 版本: {dp.version()}")
    print(f"  CUDA 版本: {dp.cuda_version()}")
    print(f"  CUDA 可用: {dp.cuda_available()}")
    print(f"  CUDA 设备数量: {dp.cuda_device_count()}")
    print()
    
    # 列出可用模型
    print("可用模型:")
    for model_name in dp.list_available_models():
        print(f"  - {model_name}")
    print()
    
    # 加载模型
    print(f"加载模型 '{args.model}'...")
    start_time = time.time()
    model = dp.load_model(args.model, device=args.device, dtype=args.dtype)
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时 {load_time:.2f} 秒")
    print(f"  模型类型: {model.model_type}")
    print(f"  设备: {model.device}")
    print(f"  词汇表大小: {model.vocab_size}")
    print(f"  最大序列长度: {model.max_sequence_length}")
    print()
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = Tokenizer(args.model)
    
    # 设置分词器
    model.set_tokenizer(tokenizer)
    
    # 优化模型
    if args.optimization != "none":
        print(f"优化模型 (类型: {args.optimization}, 级别: {args.level})...")
        start_time = time.time()
        results = dp.optimize_model(model, args.optimization, args.level, enable_profiling=True)
        optimize_time = time.time() - start_time
        print(f"优化完成，耗时 {optimize_time:.2f} 秒")
        print("优化结果:")
        print_dict(results, 2)
        print()
    
    # 量化模型
    if args.quantize:
        print(f"量化模型 (精度: {args.quantize_precision})...")
        start_time = time.time()
        results = dp.quantize_model(model, args.quantize_precision)
        quantize_time = time.time() - start_time
        print(f"量化完成，耗时 {quantize_time:.2f} 秒")
        print("量化结果:")
        print_dict(results, 2)
        print()
    
    # 基准测试
    if args.benchmark:
        print("进行基准测试...")
        print(f"  测试文本: '{args.benchmark_text}'")
        print(f"  运行次数: {args.benchmark_runs}")
        print(f"  预热次数: {args.warmup_runs}")
        
        results = dp.benchmark_model(
            model, 
            input_text=args.benchmark_text, 
            num_runs=args.benchmark_runs,
            warmup_runs=args.warmup_runs
        )
        
        print("基准测试结果:")
        print_dict(results, 2)
        print()
    
    # 生成文本
    if args.generate:
        print(f"生成文本 (提示: '{args.prompt}')...")
        
        # 创建流式回调
        def stream_callback(result):
            # 打印生成的文本
            print(result.texts[0], end="", flush=True)
            # 继续生成
            return True
        
        # 创建生成配置
        config = dp.GenerationConfig(
            max_length=args.max_length,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        # 流式生成
        print("生成结果:")
        model.generate_stream(args.prompt, stream_callback, config)
        print("\n")
    
    print("完成!")

if __name__ == "__main__":
    main() 