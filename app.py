from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the tokenizer and model (LLaMA 3.1 variant from Hugging Face)
model_name = "unsloth/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a pipeline for text generation
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import os

# Optional: set your Hugging Face token here or use CLI login
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"
# login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# Load the tokenizer and model (LLaMA 3.1 variant from Hugging Face)
model_name = "unsloth/Meta-Llama-3.1-8B"

try:
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Simple interactive loop
print("\n🤖 Welcome to LLaMA 3.1 Chat! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    prompt = f"[INST] {user_input} [/INST]"
    try:
        response = chatbot(
            prompt,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = response[0]["generated_text"]
        reply = generated.replace(prompt, "").strip()
        print(f"LLaMA: {reply}\n")
    except Exception as err:
        print(f"⚠️ Error during generation: {err}")


# Simple interactive loop
print("Welcome to LLaMA Chat! Type 'exit' to stop.")

while True:
    user_input = input("You: ")  
    if user_input.lower() == "exit":
        break

    prompt = f"[INST] {user_input} [/INST]"
    response = chatbot(prompt, max_length=200, do_sample=True, temperature=0.7, top_p=0.9)
    print("LLaMA:", response[0]["generated_text"].replace(prompt, "").strip())
