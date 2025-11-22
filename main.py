import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pathlib

# === LOAD SYSTEM PROMPT ===
system_prompt_path = pathlib.Path("system_prompt.txt")
SYSTEM_PROMPT = system_prompt_path.read_text()

# === MODEL ===
MODEL_NAME = "google/gemma-2b"   # piccola, gira con CPU su Render

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# === API APP ===
app = FastAPI()

class Query(BaseModel):
    prompt: str
    max_tokens: int = 300

@app.post("/generate")
def generate_text(query: Query):
    full_prompt = SYSTEM_PROMPT + "\n\n" + query.prompt

    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=query.max_tokens,
        temperature=0.7,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Estrarre solo la parte dopo l'ultimo prompt dell'utente
    if query.prompt in text:
        text = text.split(query.prompt)[-1].strip()

    return { "response": text }

@app.get("/")
def root():
    return {"status": "Carter Brain Online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
