"""
Quick test script to verify vLLM server is running and accepting requests.
"""

import argparse
import requests
import json


def test_health(base_url: str):
    """Test if server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_basic_generate(base_url: str):
    """Test basic /generate endpoint (non-OpenAI API)."""
    print("\n" + "="*50)
    print("Testing basic /generate endpoint...")
    print("="*50)

    payload = {
        "prompt": "What is 2 + 2? The answer is",
        "max_tokens": 50,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False


def test_models(base_url: str):
    """List available models."""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        print(f"Models endpoint: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {json.dumps(models, indent=2)}")
            return True
        return False
    except Exception as e:
        print(f"Models check failed: {e}")
        return False


def test_chat_text_only(base_url: str, model_name: str):
    """Test chat completion with text only."""
    print("\n" + "="*50)
    print("Testing text-only chat completion...")
    print("="*50)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2? Answer briefly."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False


def test_chat_with_audio(base_url: str, model_name: str, audio_path: str):
    """Test chat completion with audio."""
    print("\n" + "="*50)
    print("Testing chat with audio...")
    print("="*50)

    # Use file:// URL for local files
    audio_url = f"file://{audio_path}" if not audio_path.startswith("http") else audio_path

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                    {"type": "text", "text": "What do you hear in this audio? Describe briefly."}
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test vLLM server")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8901",
                        help="vLLM server base URL")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
                        help="Model name")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to test audio file (optional)")

    args = parser.parse_args()

    print("="*50)
    print("vLLM Server Test")
    print("="*50)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model_name}")
    print()

    # Test health
    print("1. Testing health endpoint...")
    health_ok = test_health(args.base_url)

    # Test basic generate endpoint
    print("\n2. Testing basic /generate endpoint...")
    basic_ok = test_basic_generate(args.base_url)

    # Test models (OpenAI API)
    print("\n3. Testing OpenAI /v1/models endpoint...")
    models_ok = test_models(args.base_url)

    # Test text chat (OpenAI API)
    text_ok = False
    if models_ok:
        print("\n4. Testing OpenAI text-only chat...")
        text_ok = test_chat_text_only(args.base_url, args.model_name)

    # Test audio chat if path provided
    audio_ok = None
    if args.audio_path and text_ok:
        print("\n5. Testing audio chat...")
        audio_ok = test_chat_with_audio(args.base_url, args.model_name, args.audio_path)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Health: {'✓' if health_ok else '✗'}")
    print(f"Basic /generate: {'✓' if basic_ok else '✗'}")
    print(f"OpenAI /v1/models: {'✓' if models_ok else '✗'}")
    print(f"OpenAI chat: {'✓' if text_ok else '✗'}")
    if audio_ok is not None:
        print(f"Audio chat: {'✓' if audio_ok else '✗'}")

    if basic_ok and not text_ok:
        print("\n⚠️  Server is running with BASIC API (not OpenAI-compatible)")
        print("To use OpenAI chat API, restart with: vllm serve <model>")
    elif text_ok:
        print("\n✓ vLLM server is ready with OpenAI-compatible API!")
    else:
        print("\n✗ vLLM server has issues. Check the server logs.")


if __name__ == "__main__":
    main()
