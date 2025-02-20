import deeppowers as dp

def main():
    print(f"DeepPowers version: {dp.__version__}")
    print(f"CUDA available: {dp.cuda_available()}")

if __name__ == "__main__":
    main()
