import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Qwen API Gateway")

#  ADD THIS CORS BLOCK SO REACT CAN TALK TO FASTAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (ideal for local development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define request schemas matching your payload
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen3-8B"
    messages: List[Message]
    max_tokens: Optional[int] = 256

# Configuration constants based on your working Postman request
TARGET_URL = "https://103.42.50.41:1032/v1/chat/completions"
BEARER_TOKEN = "key"

@app.post("/ask-qwen")
async def ask_qwen(payload: ChatCompletionRequest):
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "FastAPI-Client/1.0"
    }
    
    # verify=False bypasses the SSL check just like disabling SSL verification in Postman
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(
                TARGET_URL,
                json=payload.model_dump(),
                headers=headers,
                timeout=30.0
            )
            
            # Raise an error if the remote server returns a bad status code
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, 
                detail=f"Remote server error: {exc.response.text}"
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=500, 
                detail=f"An error occurred while requesting: {str(exc)}"
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
