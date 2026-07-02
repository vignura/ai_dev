import argparse
import time
import openvino_genai as ov_genai

def main():
    parser = argparse.ArgumentParser(description="Run OpenVINO LLM locally with live performance profiling.")
    parser.add_argument("model_dir", type=str, help="Path to the local model folder")
    parser.add_argument("--device", type=str, default="NPU", choices=["NPU", "GPU", "CPU"], help="Target device")
    parser.add_argument("--tokens", type=int, default=4096, help="Max new tokens to generate")
    
    args = parser.parse_args()

    print(f"Loading OpenVINO GenAI Engine...")
    try:
        # Dynamically allocate the compilation cache depending on target hardware
        if args.device.upper() == "GPU":
            print("🤖 GPU Mode detected: Disabling compilation disk cache to ensure clean weight parsing...")
            pipe = ov_genai.LLMPipeline(args.model_dir, args.device)
        else:
            # Keep cache active for NPU / CPU to avoid long initial boot times
            pipe = ov_genai.LLMPipeline(args.model_dir, args.device, CACHE_DIR="./ov_cache")
            
    except Exception as e:
        print(f"\n--- CRITICAL ERROR: Hardware compilation failed ---\n{e}")
        return

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.tokens
    config.do_sample = True
    config.temperature = 0.6
    config.top_p = 0.9

    def streamer(subword: str) -> ov_genai.StreamingStatus:
        print(subword, end="", flush=True)
        return ov_genai.StreamingStatus.RUNNING

    chat_history = ov_genai.ChatHistory()

    print(f"\n--- Local {args.device.upper()} Chatbot Live with Telemetry (Type 'exit' to quit) ---")

    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError:
            break
            
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            continue
            
        chat_history.append({"role": "user", "content": user_input})
        print("AI: ", end="", flush=True)
        
        # Start wall clock tracking
        start_wall_time = time.perf_counter()
        
        # KEY CHANGE: The pipeline returns a DecodedResults object containing .perf_metrics
        decoded_results = pipe.generate(chat_history, config, streamer)
        print()
        
        end_wall_time = time.perf_counter()
        
        # Extract the metrics recorded by the hardware plugin wrapper
        metrics = decoded_results.perf_metrics
        wall_duration = end_wall_time - start_wall_time
        
        # Extract parameters explicitly computed by OpenVINO
        tokens_generated = metrics.get_num_generated_tokens()
        throughput = metrics.get_throughput().mean           # Tokens per second
        ttft = metrics.get_ttft().mean                       # Time to First Token (ms)
        tpot = metrics.get_tpot().mean                       # Time Per Output Token (ms)

        # Append assistant history
        chat_history.append({"role": "assistant", "content": decoded_results.texts})
        
        # Print out the hardware profiling dashboard
        print("\n" + "="*50)
        print(f"📊 HARDWARE PERFORMANCE METRICS ({args.device.upper()})")
        print("="*50)
        print(f"⏱️ Time to First Token (TTFT) : {ttft:.2f} ms")
        print(f"⚡ Generation Speed           : {throughput:.2f} tokens/sec")
        print(f"🧵 Latency Per Token (TPOT)   : {tpot:.2f} ms/token")
        print(f"📦 Total Tokens Generated     : {tokens_generated} tokens")
        print(f"⏳ Total Execution Duration   : {wall_duration:.2f} seconds")
        print("-"*50)
        print("💡 POWER EFFICIENCY ANALYSIS:")
        
        # Provide contextual analysis for Lunar Lake architectures
        if args.device.upper() == "NPU":
            print("  • The NPU processes token matrices using low-leakage island SRAM cache.")
            print(f"  • Estimated energy profile: ~1.5W to 4W sustained SoC draw.")
            print(f"  • Computed efficiency score: ~{(throughput / 2.5):.2f} tokens per watt-second.")
        elif args.device.upper() == "GPU":
            print("  • The Arc GPU utilizes high-width vector execution units.")
            print(f"  • Estimated energy profile: ~12W to 25W burst package draw.")
            print(f"  • Computed efficiency score: ~{(throughput / 15.0):.2f} tokens per watt-second.")
        print("="*50)

if __name__ == "__main__":
    main()