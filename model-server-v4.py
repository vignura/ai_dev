from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import openvino_genai as ov_genai
import uvicorn
import json
import time
import uuid
import queue
import threading
import asyncio
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run model server with custom model name and path')
parser.add_argument('--model-name', type=str, default='qwen3-coder-30b', help='Name of the model')
parser.add_argument('--model-path', type=str, default='./qwen3-coder-30b-a3b-int4', help='Path to the model directory')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to log incoming requests')
args = parser.parse_args()

DEBUG_MODE = args.debug

MODEL_NAME = args.model_name
MODEL_PATH = args.model_path

# Define context window sizes for known models
MODEL_CONTEXT_WINDOWS = {
    "qwen3-coder-30b": 131072,
    "qwen2.5-coder": 32768,
    "qwen2-coder": 32768,
    # Add other models as needed
}

# Default context window size
DEFAULT_CONTEXT_WINDOW = 32768

# Get context window size for the current model
CONTEXT_WINDOW = MODEL_CONTEXT_WINDOWS.get(MODEL_NAME, DEFAULT_CONTEXT_WINDOW)

app = FastAPI()

print("Loading model...")
# Ensure this path points to your OpenVINO model directory
pipe = ov_genai.LLMPipeline(MODEL_PATH, "GPU")
print("Model loaded.")

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
                "context_window": CONTEXT_WINDOW,
                "max_tokens": 2048,
                "cost_per_token": {
                    "prompt": 0.000001,
                    "completion": 0.000002
                }
            }
        ]
    }

def build_prompt(messages):
    """
    Convert OpenAI chat messages into a single prompt.
    """
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"

    prompt += "<|assistant|>\n"
    return prompt

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        data = await request.json()
        
        if DEBUG_MODE:
            print("Debug: Incoming request body:")
            print(json.dumps(data, indent=2))
            
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        
        prompt = build_prompt(messages)
        
        # print("=" * 80)
        # print(prompt)
        # print("=" * 80)
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        if stream:
            async def event_stream():
                # 1. Send the initial role setup chunk
                yield (
                    "data: " + json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                    }) + "\n\n"
                )

                # 2. Set up a queue to hold tokens as they are generated
                token_queue = queue.Queue()

                # 3 & 4. Setup timers, counts, and the background thread
                def run_generation():
                    start_time = time.perf_counter()
                    first_token_time = [None]
                    token_count = [0]
                    
                    def ov_streamer(subword: str):
                        # Capture Time To First Token (TTFT)
                        if first_token_time[0] is None:
                            first_token_time[0] = time.perf_counter()
                            
                        token_count[0] += 1
                        token_queue.put(subword)
                        return False  # Continue generation

                    try:
                        pipe.generate(prompt, streamer=ov_streamer)
                    except Exception as e:
                        print(f"Generation error: {e}")
                    finally:
                        end_time = time.perf_counter()
                        token_queue.put(None)  # Signal completion to the async loop
                        
                        # --- Calculate and Print Stats ---
                        ttft = (first_token_time[0] - start_time) if first_token_time[0] else 0
                        total_time = end_time - start_time
                        gen_time = total_time - ttft
                        speed = token_count[0] / gen_time if gen_time > 0 else 0
                        
                        # Try to get exact prompt tokens, fallback to estimation if API differs
                        try:
                            # OV GenAI usually allows encoding to get token count
                            prompt_tokens = pipe.get_tokenizer().encode(prompt).get_shape()[1]
                        except:
                            prompt_tokens = int(len(prompt) / 3.5) # Safe heuristic
                            
                        print("\n" + "=" * 58)
                        print(f"Model              : {MODEL_NAME}")
                        print(f"Prompt Tokens      : {prompt_tokens}")
                        print(f"Completion Tokens  : {token_count[0]}")
                        print("")
                        print(f"TTFT               : {ttft:.2f} s")
                        print(f"Generation         : {gen_time:.2f} s")
                        print(f"Total Request      : {total_time:.2f} s")
                        print("")
                        print(f"Generation Speed   : {speed:.1f} tok/s")
                        print("=" * 58 + "\n")

                thread = threading.Thread(target=run_generation)
                thread.start()

                # 5. Read from the queue and send to client instantly
                while True:
                    try:
                        # Try to grab a token without blocking the async loop
                        token = token_queue.get_nowait()
                    except queue.Empty:
                        # If queue is empty, wait a tiny fraction of a second and check again
                        await asyncio.sleep(0.01)
                        continue
                        
                    # If we hit the None signal, break the loop
                    if token is None:
                        break
                        
                    # Yield the individual token to the client
                    yield (
                        "data: " + json.dumps({
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": MODEL_NAME,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                        }) + "\n\n"
                    )

                # 6. Send the final stop sequence
                yield (
                    "data: " + json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }) + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream"
            )
            
        else:
            # Non-streaming fallback (Used by Aider for background tasks)
            start_time = time.perf_counter()
            
            result = pipe.generate(prompt)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            if not isinstance(result, str):
                result = str(result)
                
            # Calculate stats for non-streaming
            try:
                prompt_tokens = pipe.get_tokenizer().encode(prompt).get_shape()[1]
                completion_tokens = pipe.get_tokenizer().encode(result).get_shape()[1]
            except:
                # Safe fallback if tokenizer fails
                prompt_tokens = int(len(prompt) / 3.5)
                completion_tokens = int(len(result) / 3.5)
                
            speed = completion_tokens / total_time if total_time > 0 else 0
            
            # Print the stats to your server console
            print("\n" + "=" * 58)
            print(f"Model              : {MODEL_NAME} (Background/No-Stream)")
            print(f"Prompt Tokens      : {prompt_tokens}")
            print(f"Completion Tokens  : {completion_tokens}")
            print("")
            print(f"Total Request      : {total_time:.2f} s")
            print("")
            print(f"Generation Speed   : {speed:.1f} tok/s")
            print("=" * 58 + "\n")
                
            return JSONResponse({
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            })
            
    except Exception as e:
        print(f"Error during completion: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
