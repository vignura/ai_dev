from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import json
import time
import uuid
import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
import base64

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run image generation server with custom model name and path')
parser.add_argument('--model-name', type=str, default='stabilityai/stable-diffusion-2-1', help='Name of the model')
parser.add_argument('--model-path', type=str, default='./stable-diffusion-2-1', help='Path to the model directory')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to log incoming requests')
args = parser.parse_args()

DEBUG_MODE = args.debug

MODEL_NAME = args.model_name
MODEL_PATH = args.model_path

app = FastAPI()

print("Loading image generation model...")
try:
    from optimum.intel.openvino import OVDiffusionPipeline
    pipe = OVDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        safety_checker=None,
        device="GPU"  # Use NPU for acceleration
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

@app.get("/v1/image/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
                "max_tokens": 2048,
                "cost_per_token": {
                    "prompt": 0.000001,
                    "completion": 0.000002
                }
            }
        ]
    }

# Global executor for offloading blocking operations
executor = ThreadPoolExecutor(max_workers=4)

async def generate_image_with_async_streaming(prompt: str, image_id: str, image_queue: asyncio.Queue) -> None:
    """Generate image asynchronously with progress updates."""
    start_time = time.perf_counter()
    
    try:
        # Run image generation in thread pool
        loop = asyncio.get_event_loop()
        image_tensor = await loop.run_in_executor(executor, lambda: pipe(prompt, num_inference_steps=20))
        
        # Convert tensor to PIL Image
        image = Image.fromarray(image_tensor.data[0])
        
        # Save image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        image_data = image_bytes.getvalue()
        
        # Put image data into queue
        await image_queue.put(image_data)
        
    except Exception as e:
        print(f"Image generation error: {e}")
        await image_queue.put(None)
    finally:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"\nImage Generation Time: {total_time:.2f} seconds")
        print("=" * 58 + "\n")

@app.post("/v1/image/generations")
async def image_generations(request: Request):
    try:
        data = await request.json()
        
        if DEBUG_MODE:
            print("Debug: Incoming request body:")
            print(json.dumps(data, indent=2))
            
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        
        image_id = f"img-{uuid.uuid4().hex}"
        
        if stream:
            async def event_stream() -> asyncio.AsyncGenerator[str, None]:
                # 1. Send the initial status chunk
                yield (
                    "data: " + json.dumps({
                        "id": image_id, 
                        "object": "image_generation.chunk",
                        "created": int(time.time()), 
                        "model": MODEL_NAME,
                        "status": "generating"
                    }) + "\n\n"
                )

                # 2. Set up an asyncio queue to hold image data
                image_queue = asyncio.Queue(maxsize=1)

                # 3. Start generation in background thread
                generation_task = asyncio.create_task(generate_image_with_async_streaming(prompt, image_id, image_queue))

                # 4. Read from the queue and send to client instantly
                while True:
                    try:
                        # Wait for next image data with timeout
                        image_data = await asyncio.wait_for(image_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                        
                    # If we hit the None signal, break the loop
                    if image_data is None:
                        break
                        
                    # Encode image data to base64
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    
                    # Yield the image data to the client
                    yield (
                        "data: " + json.dumps({
                            "id": image_id, 
                            "object": "image_generation.chunk",
                            "created": int(time.time()), 
                            "model": MODEL_NAME,
                            "image": base64_image
                        }) + "\n\n"
                    )

                # Wait for generation to complete
                await generation_task

                # 5. Send the final completion status
                yield (
                    "data: " + json.dumps({
                        "id": image_id, 
                        "object": "image_generation.chunk",
                        "created": int(time.time()), 
                        "model": MODEL_NAME,
                        "status": "completed"
                    }) + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream"
            )
            
        else:
            # Non-streaming fallback
            start_time = time.perf_counter()
            
            image_tensor = pipe(prompt, num_inference_steps=20)
            image = Image.fromarray(image_tensor.data[0])
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Save image to bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            image_data = image_bytes.getvalue()
            
            # Encode to base64 for response
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Print the stats to your server console
            print("\n" + "=" * 58)
            print(f"Image Generation Time: {total_time:.2f} seconds")
            print("=" * 58 + "\n")
                
            return JSONResponse({
                "id": image_id,
                "object": "image_generation",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "image": base64_image
            })
            
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
