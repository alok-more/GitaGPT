import os
import streamlit as st
from dotenv import load_dotenv
import requests
from langdetect import detect, DetectorFactory
from typing import List, Dict

DetectorFactory.seed = 0

load_dotenv()

st.set_page_config(
    page_title="GitaGPT - Bhagavad Gita AI",
    page_icon="🕉️",
    layout="centered",
    initial_sidebar_state="expanded"
)

class GeetaGPT:
    """GitaGPT class using Groq API with bilingual support"""
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"
        self.max_history = 6 

    def get_ai_response(self, user_message: str, conversation_history: List[Dict]) -> Dict:
        """Get response from Groq API with language detection"""

        try:
            lang = detect(user_message)
        except Exception:
            lang = 'en'
        
        # System prompt based on language
        system_prompt = (
            """तुम्ही GitaGPT आहात, भगवद्गीतेत प्रवीण असलेले AI सहाय्यक.
            उत्तर देताना मित्रासारखे, सोप्या शब्दात, जीवनात उपयोगी समजावून सांगा.
            जर श्लोक नमूद करायचा असेल, तर तो **सदैव संस्कृतमध्ये** द्या आणि त्याचा अर्थ स्पष्ट करा.
            उत्तरे दयाळू, प्रोत्साहक आणि तत्त्वज्ञानिक असावीत, परंतु जास्त कठीण भाषेत नाहीत. कृपया श्लोकाच्या शब्दशः तपासणीसाठी किंवा त्रुटी शोधण्यासाठी वेळ वाया घालवू नका."""
            if lang == 'mr' else
            """You are GitaGPT, a wise and compassionate AI assistant knowledgeable in the Bhagavad Gita.
            Explain teachings in simple, friendly language applicable in daily life.
            If quoting shlokas, always provide them in **Sanskrit**, and explain their meaning.
            Include practical examples, analogies, or stories to make concepts easy to understand.
            Responses should be thoughtful, helpful, and conversational, not just literal translations. Do not try to verify or correct the shlokas—just provide clear and concise explanations."""
        )


        if not self.api_key:
            return {"success": False, "error": "Groq API key not found in .env file"}
        

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[-self.max_history:])
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data['choices'][0]['message']['content']
                return {"success": True, "response": assistant_message}
            else:
                return {"success": False, "error": f"API Error {response.status_code}: {response.text}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'geeta_gpt' not in st.session_state:
    st.session_state.geeta_gpt = GeetaGPT()

st.markdown("""
<style>
    .main-header { text-align: center; color: white; padding: 1rem; margin-bottom: 1rem; }
    .main-header h1 { font-size: 3rem; margin-bottom: 0.5rem; }
    .main-header p { font-size: 1.2rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🕉️ GitaGPT - Bhagavad Gita AI</h1>
    <p>Your AI Guide to the Wisdom of the Bhagavad Gita</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.subheader("💡 Quick Questions")
    quick_questions = [
        "What is karma yoga?",
        "How to find inner peace?",
        "What is my duty in life?",
        "How to control my mind?",
        "What is true wisdom?"
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()
    
    st.divider()
    
    if st.button("🔄 Clear Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

# Chat Area
st.markdown("### 💬 Chat")

# Initial message
if not st.session_state.messages:
    st.info("""
    🙏 **Namaste!** Welcome to GitaGPT.
    
Ask me anything about the Bhagavad Gita - karma yoga, meditation, dharma, or life's guidance.
💡 Try the quick questions in the sidebar!
""")
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🕉️"):
                st.write(msg["content"])

if 'pending_question' in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question
else:
    user_input = st.chat_input("Ask about karma, dharma, yoga, meditation...")

if user_input:
    if not st.session_state.geeta_gpt.api_key:
        st.error("⚠️ Please add your Groq API key to the .env file!")
        st.info("Get your FREE key at: https://console.groq.com/")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🕉️ Contemplating the wisdom of the Gita..."):
        result = st.session_state.geeta_gpt.get_ai_response(
            user_input,
            st.session_state.messages[:-1]
        )
    
    if result["success"]:
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        st.rerun()
    else:
        st.error(f"Error: {result.get('error')}")
