# Memory usage monitoring script for DeepPowers tokenizer
import time
import psutil
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import deeppowers as dp

class MemoryMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.process = psutil.Process()
        self.timestamps = []
        self.memory_usage = []
        self.pool_usage = []
        self.string_pool_usage = []
        
    def start(self):
        self.timestamps = []
        self.memory_usage = []
        self.pool_usage = []
        self.string_pool_usage = []
        self.running = True
        
    def stop(self):
        self.running = False
        
    def record(self, tokenizer):
        memory_info = self.process.memory_info()
        current_time = time.time()
        
        self.timestamps.append(current_time - self.start_time)
        self.memory_usage.append(memory_info.rss / 1024 / 1024)  # MB
        
        # Get pool statistics
        pool_stats = tokenizer.get_memory_pool_stats()
        self.pool_usage.append(pool_stats['used_memory'] / 1024 / 1024)  # MB
        
        string_pool_stats = tokenizer.get_string_pool_stats()
        self.string_pool_usage.append(string_pool_stats['memory_usage'] / 1024 / 1024)  # MB
        
    def plot(self, output_path):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.timestamps, self.memory_usage, label='Total Memory')
        plt.plot(self.timestamps, self.pool_usage, label='Memory Pool')
        plt.plot(self.timestamps, self.string_pool_usage, label='String Pool')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        sizes = [
            sum(self.pool_usage) / len(self.pool_usage),
            sum(self.string_pool_usage) / len(self.string_pool_usage),
            sum(self.memory_usage) / len(self.memory_usage) - 
            sum(self.pool_usage) / len(self.pool_usage) - 
            sum(self.string_pool_usage) / len(self.string_pool_usage)
        ]
        labels = ['Memory Pool', 'String Pool', 'Other']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Average Memory Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def run_memory_test(tokenizer_path, input_file, output_dir, duration=60):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer and monitor
    tokenizer = dp.Tokenizer()
    tokenizer.load(tokenizer_path)
    monitor = MemoryMonitor()
    
    # Load test data
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    print(f"Starting memory monitoring for {duration} seconds...")
    monitor.start()
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Process some texts
            batch = texts[:32]
            tokenizer.encode_batch_parallel(batch)
            
            # Record memory usage
            monitor.record(tokenizer)
            time.sleep(monitor.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        monitor.stop()
    
    # Generate report
    print("\nGenerating memory usage report...")
    monitor.plot(output_dir / 'memory_usage.png')
    
    # Save statistics
    with open(output_dir / 'memory_stats.txt', 'w') as f:
        f.write("Memory Usage Statistics:\n")
        f.write("-----------------------\n")
        f.write(f"Average total memory: {sum(monitor.memory_usage)/len(monitor.memory_usage):.2f}MB\n")
        f.write(f"Peak total memory: {max(monitor.memory_usage):.2f}MB\n")
        f.write(f"Average memory pool usage: {sum(monitor.pool_usage)/len(monitor.pool_usage):.2f}MB\n")
        f.write(f"Peak memory pool usage: {max(monitor.pool_usage):.2f}MB\n")
        f.write(f"Average string pool usage: {sum(monitor.string_pool_usage)/len(monitor.string_pool_usage):.2f}MB\n")
        f.write(f"Peak string pool usage: {max(monitor.string_pool_usage):.2f}MB\n")

def main():
    parser = argparse.ArgumentParser(description='Monitor DeepPowers tokenizer memory usage')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer model')
    parser.add_argument('--input', required=True, help='Input text file for testing')
    parser.add_argument('--output', required=True, help='Output directory for reports')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        return 1
        
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    run_memory_test(args.tokenizer, args.input, args.output, args.duration)
    return 0

if __name__ == '__main__':
    main() 