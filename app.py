import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import os

# Hugging Face login (optional)
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"
# login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

@st.cache_resource(show_spinner="Loading model...")
def load_chatbot():
    model_name = "unsloth/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chatbot = load_chatbot()

st.title("🦙 LLaMA 3.1 Chatbot")
st.markdown("Chat with the `unsloth/Meta-Llama-3.1-8B` model.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("chat_form"):
    user_input = st.text_area("Your message:", "", height=100)
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    prompt = f"[INST] {user_input.strip()} [/INST]"
    with st.spinner("LLaMA is thinking..."):
        response = chatbot(
            prompt,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=chatbot.tokenizer.eos_token_id
        )
        reply = response[0]["generated_text"].replace(prompt, "").strip()
        st.session_state.chat_history.append(("You", user_input.strip()))
        st.session_state.chat_history.append(("LLaMA", reply))

# Display chat history
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")
