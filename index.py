
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model (Note: large models may hit Vercel's serverless size limits, 
# consider using a lightweight model or API wrapper if it exceeds 250MB)
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

@app.get("/")
def home():
    return {"message": "LLM Alignment API is live!"}

@app.get("/generate")
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"prompt": prompt, "response": response}
