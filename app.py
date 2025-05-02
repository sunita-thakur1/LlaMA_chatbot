from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the tokenizer and model (LLaMA 3.1 variant from Hugging Face)
model_name = "unsloth/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a pipeline for text generation
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)    

# Simple interactive loop
print("Welcome to LLaMA Chat! Type 'exit' to stop.")

while True:
    user_input = input("You: ")  
    if user_input.lower() == "exit":
        break

    prompt = f"[INST] {user_input} [/INST]"
    response = chatbot(prompt, max_length=200, do_sample=True, temperature=0.7, top_p=0.9)
    print("LLaMA:", response[0]["generated_text"].replace(prompt, "").strip())
