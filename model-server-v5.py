import asyncio
import json
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, AsyncGenerator
import logging
import re
import time
import uuid
import queue
import threading
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OpenVINO GenAI
try:
    import openvino_genai as ov_genai
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    logger.warning("OpenVINO GenAI not installed. Model server will not work.")

# Tool schema definition
SEARCH_TOOL_SCHEMA = {
    "name": "search_internet",
    "description": "Searches the internet for current information on topics that require up-to-date data",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for finding current information"
            }
        },
        "required": ["query"]
    }
}

# Define context window sizes for known models
MODEL_CONTEXT_WINDOWS = {
    "qwen3-coder-30b": 131072,
    "qwen2.5-coder": 32768,
    "qwen2-coder": 32768,
    # Add other models as needed
}

# Default context window size
DEFAULT_CONTEXT_WINDOW = 32768

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run model server with custom model name and path')
parser.add_argument('--model-name', type=str, default='qwen3-coder-30b', help='Name of the model')
parser.add_argument('--model-path', type=str, default='./qwen3-coder-30b-a3b-int4', help='Path to the model directory')
parser.add_argument('--cache-dir', type=str, default='../modles/cache', help='Path to the cache directory')
args = parser.parse_args()

MODEL_NAME = args.model_name
MODEL_PATH = args.model_path
CACHE_DIR = args.cache_dir

# Get context window size for the current model
CONTEXT_WINDOW = MODEL_CONTEXT_WINDOWS.get(MODEL_NAME, DEFAULT_CONTEXT_WINDOW)

# Initialize FastAPI app
app = FastAPI(title="Model Server v5 with Search Capabilities")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import duckduckgo-search for web search functionality
# Note: This requires installing with: pip install duckduckgo-search
try:
    from duckduckgo_search.duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False
    logger.warning("duckduckgo-search not installed. Search functionality will be limited.")

async def perform_web_search(query: str) -> Dict[str, Any]:
    """Perform web search using DuckDuckGo HTML scraper"""
    if not HAS_DUCKDUCKGO:
        return {
            "error": "Search functionality not available. Please install duckduckgo-search.",
            "query": query
        }
    
    try:
        results = []
        # Use DDGS as async context manager
        async with DDGS() as ddgs:
            # Get top 5 actual web results
            async for r in ddgs.text(query, max_results=5):
                results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', '')[:300] + "..." if len(r.get('body', '')) > 300 else r.get('body', ''),
                    'url': r.get('href', '')
                })
        
        return {
            "query": query,
            "results": results,
            "timestamp": str(asyncio.get_event_loop().time())
        }
    except Exception as e:
        return {"error": str(e), "query": query}

def extract_search_query(content: str) -> str:
    """Safely extract search query from JSON tool call"""
    if '"name":"search_internet"' in content:
        try:
            # Extract JSON portion containing the tool call
            json_match = re.search(r'\{.*?"name":"search_internet".*?\}', content, re.DOTALL)
            if json_match:
                tool_call_data = json.loads(json_match.group(0))
                # Handle different JSON structures
                if 'arguments' in tool_call_data:
                    return tool_call_data['arguments'].get('query', '')
                elif 'parameters' in tool_call_data:
                    return tool_call_data['parameters'].get('query', '')
                else:
                    # Fallback: check if query is directly in the JSON
                    return tool_call_data.get('query', '')
        except (json.JSONDecodeError, KeyError):
            pass
    return ""

def format_search_results(results: dict) -> str:
    """Format search results properly for LLM consumption"""
    if 'error' in results:
        return f"Search Error: {results['error']}"
    
    formatted = "Search Results for: " + results.get('query', 'Unknown') + "\n\n"
    if results.get('results'):
        for i, result in enumerate(results['results'], 1):
            formatted += f"{i}. {result.get('title', 'Untitled')}\n"
            formatted += f"   {result.get('snippet', 'No content')}\n"
            formatted += f"   Source: {result.get('url', 'Unknown')}\n\n"
    
    return formatted.strip()

def process_tool_calls_and_inject_context(messages: List[Dict]) -> List[Dict]:
    """Process tool calls and properly inject search results"""
    # Check if we have a search tool call
    search_query = None
    search_message_index = -1
    
    # Find the search tool call in recent messages
    for i, msg in enumerate(reversed(messages)):
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            content = msg.get('content', '')
            query = extract_search_query(content)
            if query:
                search_query = query
                search_message_index = len(messages) - 1 - i
                break
    
    # If search was called, execute it and inject results
    if search_query and search_message_index != -1:
        # Execute search asynchronously
        search_results = asyncio.run(perform_web_search(search_query))
        formatted_results = format_search_results(search_results)
        
        # Add search results as a proper tool response message
        messages.append({
            "role": "tool",
            "name": "search_internet",
            "content": formatted_results
        })
    
    return messages

def build_prompt(messages: List[Dict]) -> str:
    """Build enhanced prompt with proper message formatting"""
    # Process tool calls if any exist
    processed_messages = process_tool_calls_and_inject_context(messages)
    
    # Build the prompt from message history
    prompt_parts = []
    
    for msg in processed_messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{content}\n")
        elif role == "tool":
            prompt_parts.append(f"<|tool|>\n{content}\n")

    prompt_parts.append("<|assistant|>\n")
    return "".join(prompt_parts)

# Global variable to hold the model pipeline
pipe = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    if HAS_OPENVINO:
        try:
            print("Loading model...")
            pipe = ov_genai.LLMPipeline(MODEL_PATH, "GPU", CACHE_DIR=CACHE_DIR)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            logger.error(f"Failed to load OpenVINO model: {e}")
    else:
        print("OpenVINO GenAI not available. Model server will not function properly.")
    yield
    # Shutdown logic here if needed

app = FastAPI(lifespan=lifespan)

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

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completions with enhanced search capabilities and streaming support"""
    try:
        data = await request.json()
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        model_name = data.get('model', MODEL_NAME)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        prompt = build_prompt(messages)
        
        if stream:
            async def event_stream():
                # 1. Send the initial role setup chunk
                yield (
                    "data: " + json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model_name,
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
                            "created": int(time.time()), "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                        }) + "\n\n"
                    )

                # 6. Send the final stop sequence
                yield (
                    "data: " + json.dumps({
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model_name,
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
                
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
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
            }
            
    except Exception as e:
        logger.error(f"Error during completion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/v1/search")
async def search(request: Request):
    """Direct search endpoint for testing"""
    try:
        data = await request.json()
        query = data.get('query', '')
        
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        search_results = await perform_web_search(query)
        return {
            "query": query,
            "results": search_results.get('results', []),
            "error": search_results.get('error')
        }
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/v1/tools")
async def get_tools():
    """Return available tools for the LLM"""
    return {
        "tools": [
            SEARCH_TOOL_SCHEMA
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "Model Server v5 with Search Capabilities",
        "version": "5.0",
        "tools": ["search_internet"],
        "features": [
            "Tool call interception",
            "Web search integration",
            "Enhanced context management",
            "Streaming and non-streaming support"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "search_enabled": HAS_DUCKDUCKGO,
        "openvino_enabled": HAS_OPENVINO,
        "version": "5.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
