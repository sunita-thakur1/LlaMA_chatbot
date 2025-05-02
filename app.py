from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import os

# Optional: Set your Hugging Face token in the environment or manually login
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"
# login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# Load the tokenizer and model
model_name = "unsloth/Meta-Llama-3.1-8B"

try:
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("✅ Model loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Chat loop
print("🤖 Welcome to LLaMA 3.1 Chat! Type 'exit' to quit.\n")

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
