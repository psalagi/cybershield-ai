import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

base_url = os.getenv("OPENAI_BASE_URL", "").strip()
api_key = os.getenv("OPENAI_API_KEY", "").strip()
model = os.getenv("MODEL", "Qwen/Qwen3-8B").strip()

if not base_url or "YOUR-INSTITUTE" in base_url:
    raise SystemExit("Set OPENAI_BASE_URL in .env")
if not api_key or api_key == "YOUR_API_KEY":
    raise SystemExit("Set OPENAI_API_KEY in .env")

ca_cert = os.getenv("SSL_CERT_FILE")

http_client = httpx.Client(
    verify=ca_cert if ca_cert else True,
    timeout=float(os.getenv("TIMEOUT_SECONDS", "90"))
)

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=float(os.getenv("TIMEOUT_SECONDS", "90")),
    http_client=http_client
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant. Reply with one short sentence."},
            {"role": "user", "content": "Reply exactly: CyberShield endpoint connection successful."},
        ],
        temperature=0,
        max_tokens=60,
    )
    print("Connection successful.")
    print("Model returned:", response.choices[0].message.content)
    print("Model field:", response.model)
    if response.usage:
        print("Usage:", response.usage.model_dump())
except Exception as exc:
    print("Connection failed:", repr(exc))
    raise
