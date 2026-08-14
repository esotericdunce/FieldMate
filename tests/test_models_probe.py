import asyncio
import base64
import os
import time
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI, NotFoundError

load_dotenv()

MODELS_TO_PROBE = [
    # Llama 4 Scout variants (to test existence on Groq API)
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-scout",
    "llama-4-scout",
    # Active available models
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

@pytest.fixture
def groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")
    return AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

@pytest.mark.asyncio
async def test_probe_all_models_text_and_vision(groq_client):
    with open("frontend/src/assets/hero.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    results = []
    print("\n" + "=" * 90)
    print(f"{'MODEL ID':<45} | {'STATUS':<14} | {'TTFT':<9} | {'TOTAL':<9} | {'VISION'}")
    print("=" * 90)

    for model in MODELS_TO_PROBE:
        status = "UNKNOWN"
        ttft_ms = 0.0
        total_ms = 0.0
        vision_support = "NO"
        error_msg = ""

        t0 = time.perf_counter()
        try:
            stream = await groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "PC RAM beep code 3 beeps meaning in one sentence."}],
                max_tokens=30,
                stream=True,
            )
            first_chunk_t = None
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_chunk_t is None:
                        first_chunk_t = time.perf_counter() - t0
            total_t = time.perf_counter() - t0
            ttft_ms = (first_chunk_t or total_t) * 1000.0
            total_ms = total_t * 1000.0
            status = "AVAILABLE"

            # Test Vision capability
            try:
                vis_resp = await groq_client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in 5 words."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }],
                    max_tokens=20
                )
                if vis_resp.choices and vis_resp.choices[0].message.content:
                    vision_support = "YES (Supported)"
            except Exception as vis_err:
                err_str = str(vis_err)
                if "messages[0].content must be a string" in err_str:
                    vision_support = "NO (Text-Only)"
                else:
                    vision_support = f"NO ({type(vis_err).__name__})"

        except NotFoundError:
            status = "404 NOT FOUND"
        except Exception as e:
            status = f"ERR: {type(e).__name__}"
            error_msg = str(e)

        results.append({
            "model": model,
            "status": status,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "vision": vision_support,
            "error": error_msg,
        })

        print(f"{model:<45} | {status:<14} | {ttft_ms:7.1f}ms | {total_ms:7.1f}ms | {vision_support}")

    print("=" * 90)
    assert any(r["status"] == "AVAILABLE" for r in results)

