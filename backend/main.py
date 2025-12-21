from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="BilalAİ Backend")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    messages: list

class ImageRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=req.messages
    )
    return {
        "reply": response.choices[0].message.content
    }

@app.post("/image")
async def image(req: ImageRequest):
    img = client.images.generate(
        model="gpt-image-1",
        prompt=req.prompt,
        size="1024x1024"
    )
    return {
        "url": img.data[0].url
    }
