from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import openvino_genai as ov_genai
import uvicorn
import json
import time
import uuid
import asyncio
import argparse
import requests
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Any
import logging
import time
from functools import wraps

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run model server with custom model name and path')
parser.add_argument('--model-name', type=str, default='qwen3-coder-30b', help='Name of the model')
parser.add_argument('--model-path', type=str, default='../models/qwen3-coder-30b-a3b-int4', help='Path to the model directory')
parser.add_argument('--device', type=str, default='GPU', help='Device to run model on (GPU, CPU, NPU)')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to log incoming requests')
args = parser.parse_args()

DEBUG_MODE = args.debug

MODEL_NAME = args.model_name
MODEL_PATH = args.model_path
DEVICE = args.device

# Define context window sizes for known models
MODEL_CONTEXT_WINDOWS = {
    "qwen3-coder-30b": 131072,
    "qwen2.5-coder": 32768,
    "qwen2-coder": 32768,
}

# Default context window size
DEFAULT_CONTEXT_WINDOW = 32768

# Define web search tool schema
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for relevant information to answer queries. Always use this tool before answering factual questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                }
            },
            "required": ["query"]
        }
    }
}

# Add the second tool schema
READ_URL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_url",
        "description": "Extract the full markdown content of a specific webpage or documentation URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"}
            },
            "required": ["url"]
        }
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Circuit breaker constants
MAX_RETRIES = 3
RETRY_DELAY = 1
BACKOFF_MULTIPLIER = 2
TIMEOUT = 30

# Map tools to their triggering tags
TOOL_TAGS = {
    "web_search": "<|web_search|>",
    "read_url": "<|read_url|>"
}

# Get context window size for the current model
CONTEXT_WINDOW = MODEL_CONTEXT_WINDOWS.get(MODEL_NAME, DEFAULT_CONTEXT_WINDOW)

app = FastAPI()

print(f"Loading model on {DEVICE}...")
pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE, CACHE_DIR="../models/cache")
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
    Convert OpenAI chat messages into a single prompt with tool injection.
    """
    prompt = ""
    
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    
    # Add system messages
    for msg in system_messages:
        prompt += f"<|system|>\n{msg.get('content', '')}\n"
    
    # Inject an aggressive, binding rule for the tool
    prompt += f"<|system|>\nYou have access to the following tools:\n"
    prompt += f"{json.dumps(WEB_SEARCH_TOOL_SCHEMA)}\n"
    prompt += f"{json.dumps(READ_URL_TOOL_SCHEMA)}\n"
    prompt += """CRITICAL RULES FOR TOOL USAGE:
                1. You MUST use `web_search` for any factual queries, version checks, or current events. Do not answer from memory.
                2. If the `web_search` snippets do not contain enough deep technical detail, you MUST use `read_url` on the most relevant official documentation link provided in the search results.
                3. Output exactly <|web_search|> {"query": "search term here"}` if you want to use web_search, and <|read_url|> {"url": "https://example.com"}` if you want to read a URL. Do not add any extra text or commentary in the tool call.
                THE "KNOW-YOUR-LIMITS" RULE: If you cannot find the exact technical details in the search results or the URL content, admit: 
                "I cannot find the specific details for [X] in the current technical documentation," and provide a summary of what you *did* find.
                """
    prompt += "Do not mention the tool schema to the user - keep it internal.\n"
    
    # Add messages maintaining sequence order
    for msg in messages:
        if msg.get("role") == "user":
            prompt += f"<|user|>\n{msg.get('content', '')}\n"
        elif msg.get("role") == "assistant":
            prompt += f"<|assistant|>\n{msg.get('content', '')}\n"

    # Add final assistant trigger if the last message was from the user
    if messages and messages[-1].get("role") != "assistant":
        prompt += "<|assistant|>\n"
        
    return prompt

executor = ThreadPoolExecutor(max_workers=4)

async def execute_tool_call(tool_call: dict) -> dict:
    tool_name = tool_call.get("function", {}).get("name")
    tool_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
    
    # Validate tool arguments against schema
    if tool_name == "web_search":
        required_params = ["query"]
        if not all(param in tool_args for param in required_params):
            return {"error": f"Missing required parameters for web_search: {required_params}"}
        
        query = tool_args.get("query")
        if DEBUG_MODE: print(f"\n[INTERNAL] 🔎 Searching web for: '{query}'")
        
        # Add circuit breaker pattern for web search
        for attempt in range(MAX_RETRIES):
            try:
                # Performs a quick web search and returns the top 3 results
                results = DDGS().text(query, max_results=3)
                return {
                    "type": "search_result",
                    "query": query,
                    "results": results # Contains 'title', 'href', and 'body'
                }
            except Exception as e:
                logger.error(f"Web search attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (BACKOFF_MULTIPLIER ** attempt))
                else:
                    return {"error": f"Search failed after {MAX_RETRIES} attempts: {str(e)}"}
    
    elif tool_name == "read_url":
        required_params = ["url"]
        if not all(param in tool_args for param in required_params):
            return {"error": f"Missing required parameters for read_url: {required_params}"}
            
        target_url = tool_args.get("url")
        if DEBUG_MODE: print(f"\n[INTERNAL] 📖 Reading URL: '{target_url}'")
        
        # Add circuit breaker pattern for URL reading
        for attempt in range(MAX_RETRIES):
            try:
                # Jina AI instantly converts any URL into LLM-ready Markdown
                response = requests.get(f"https://r.jina.ai/{target_url}", timeout=TIMEOUT)
                return {
                    "type": "url_content",
                    "url": target_url,
                    "markdown_content": response.text[:15000] # Cap length to protect context window
                }
            except Exception as e:
                logger.error(f"URL read attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (BACKOFF_MULTIPLIER ** attempt))
                else:
                    return {"error": f"Failed to read URL after {MAX_RETRIES} attempts: {str(e)}"}
            
    return {"error": "Unknown tool"}

async def generate_with_async_streaming(prompt: str, completion_id: str, token_queue: asyncio.Queue) -> None:
    """Generate response asynchronously with a shared multi-turn internal tool loop."""
    start_time = time.perf_counter()
    current_prompt = prompt
    first_token_time = [None]
    total_token_count = [0]
    
    MAX_TURNS = 8 # Avoid edge case hanging/infinite looping
    
    for turn in range(MAX_TURNS):
        class StreamState:
            def __init__(self):
                self.text_buffer = ""
                self.is_tool_mode = False
                self.json_buffer = ""
                self.abort_generation = False
                self.full_generated_text = ""
                self.detected_tool=""
                self.tool_args = {}
        
        stream_state = StreamState()
        
        def ov_streamer(subword: str):
            if first_token_time[0] is None:
                first_token_time[0] = time.perf_counter()
                
            total_token_count[0] += 1
            stream_state.full_generated_text += subword
            
            # --- PHASE 2: TOOL PARAMETER PARSING ---
            if stream_state.is_tool_mode:
                stream_state.json_buffer += subword
                if '{' in stream_state.json_buffer:
                    start_pos = stream_state.json_buffer.find('{')
                    temp_buffer = stream_state.json_buffer[start_pos:]
                    
                    brace_count = 0
                    for i, char in enumerate(temp_buffer):
                        if char == '{': brace_count += 1
                        elif char == '}': brace_count -= 1
                        
                        if brace_count == 0:
                            complete_json = temp_buffer[:i+1]
                            try:
                                parsed_obj = json.loads(complete_json)
                                
                                # DYNAMIC ARGUMENT CHECK
                                # web_search needs 'query', read_url needs 'url'
                                required_keys = {"web_search": "query", "read_url": "url"}
                                expected_key = required_keys.get(stream_state.detected_tool)
                                
                                if expected_key and expected_key in parsed_obj:
                                    stream_state.tool_args = parsed_obj # Store full args
                                    stream_state.abort_generation = True
                                    return True 
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parsing error in tool call: {e}")
                                # Continue processing instead of failing completely
                                pass
                return False
            
            # --- PHASE 1: TEXT BUFFERING & TAG INTERCEPTION ---
            stream_state.text_buffer += subword
            
            # Check if any tool tag is in the buffer
            for tool_name, tag in TOOL_TAGS.items():
                if tag in stream_state.text_buffer:
                    stream_state.is_tool_mode = True
                    stream_state.detected_tool = tool_name # Save which tool was triggered
                    
                    parts = stream_state.text_buffer.split(tag)
                    safe_text = parts[0]
                    stream_state.json_buffer = parts[1] if len(parts) > 1 else ""
                    stream_state.text_buffer = ""
                    
                    if safe_text:
                        try: token_queue.put_nowait(safe_text)
                        except asyncio.QueueFull: pass
                    return False
                
            last_angle_bracket = stream_state.text_buffer.rfind('<')
            if last_angle_bracket == -1:
                try: token_queue.put_nowait(stream_state.text_buffer)
                except asyncio.QueueFull: pass
                stream_state.text_buffer = ""
            else:
                suffix = stream_state.text_buffer[last_angle_bracket:]
                # FIX: Check if suffix matches the start of ANY registered tag
                if any(tag.startswith(suffix) for tag in TOOL_TAGS.values()):
                    safe_text = stream_state.text_buffer[:last_angle_bracket]
                    if safe_text:
                        try: token_queue.put_nowait(safe_text)
                        except asyncio.QueueFull: pass
                    stream_state.text_buffer = suffix
                else:
                    try: token_queue.put_nowait(stream_state.text_buffer)
                    except asyncio.QueueFull: pass
                    stream_state.text_buffer = ""
                    
            return False 

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, lambda: pipe.generate(current_prompt, streamer=ov_streamer))
        except Exception as e:
            print(f"Generation error during turn {turn}: {e}")
            
        # --- PHASE 3: EVALUATE RESULT STATE ---
        if stream_state.abort_generation: 
            if DEBUG_MODE:
                print(f"\n[DEBUG] 🛠️ Tool Call Intercepted!")
                print(f"[DEBUG] Detected Tool: {stream_state.detected_tool}")
                print(f"[DEBUG] Exact LLM Output before abort: {repr(stream_state.full_generated_text)}")
            
            # Use the args we captured during Phase 2
            tool_result = await execute_tool_call({
                "function": {
                    "name": stream_state.detected_tool,
                    "arguments": json.dumps(stream_state.tool_args)
                }
            })
            
            # Linearly append history, inject context into system, trigger next step
            current_prompt += stream_state.full_generated_text
            current_prompt += f"\n<|system|>\nTool Result:\n{json.dumps(tool_result)}\n<|assistant|>\n"
            
            if DEBUG_MODE:
                print(f"\n[DEBUG] 📝 --- START OF TURN {turn + 2} PROMPT ---")
                print(current_prompt)
                print(f"[DEBUG] 📝 --- END OF TURN {turn + 2} PROMPT ---\n")
                
            continue
        else:
            if DEBUG_MODE and turn == 0:
                print(f"\n[DEBUG] ⚠️ No tool call detected in Turn 1! The model chose to answer directly.")
                print(f"[DEBUG] Raw output: {repr(stream_state.full_generated_text)}\n")
                
            if stream_state.text_buffer:
                try: token_queue.put_nowait(stream_state.text_buffer)
                except asyncio.QueueFull: pass
            break
            
    end_time = time.perf_counter()
    await token_queue.put(None) # Sentinel to break receiver loops
    
    # --- Performance Statistics ---
    ttft = (first_token_time[0] - start_time) if first_token_time[0] else 0
    total_time = end_time - start_time
    gen_time = total_time - ttft
    speed = total_token_count[0] / gen_time if gen_time > 0 else 0
    
    try:
        prompt_tokens = pipe.get_tokenizer().encode(prompt).get_shape()[1]
    except:
        prompt_tokens = int(len(prompt) / 3.5)
        
    print("\n" + "=" * 58)
    print(f"Model              : {MODEL_NAME}")
    print(f"Prompt Tokens      : {prompt_tokens}")
    print(f"Completion Tokens  : {total_token_count[0]}")
    print("")
    print(f"TTFT               : {ttft:.2f} s")
    print(f"Generation         : {gen_time:.2f} s")
    print(f"Total Request      : {total_time:.2f} s")
    print("")
    print(f"Generation Speed   : {speed:.1f} tok/s")
    print("=" * 58 + "\n")

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
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        token_queue = asyncio.Queue(maxsize=100)
        generation_task = asyncio.create_task(
            generate_with_async_streaming(prompt, completion_id, token_queue)
        )
        
        if stream:
            # --- STREAMING RESPONSE MECHANISM ---
            async def event_stream() -> AsyncGenerator[str, None]:
                yield (
                    "data: " + json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                    }) + "\n\n"
                )

                while True:
                    try:
                        token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                        
                    if token is None:
                        break
                        
                    yield (
                        "data: " + json.dumps({
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": MODEL_NAME,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                        }) + "\n\n"
                    )

                await generation_task

                yield (
                    "data: " + json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }) + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
            
        else:
            # --- NON-STREAMING RESPONSE MECHANISM ---
            collected_tokens = []
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                collected_tokens.append(token)
                
            await generation_task
            result_text = "".join(collected_tokens)
            
            try:
                prompt_tokens = pipe.get_tokenizer().encode(prompt).get_shape()[1]
            except:
                prompt_tokens = int(len(prompt) / 3.5)
                
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
                            "content": result_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(collected_tokens),
                    "total_tokens": prompt_tokens + len(collected_tokens)
                }
            })
            
    except Exception as e:
        print(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")