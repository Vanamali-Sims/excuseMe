import streamlit as st
from apology_generator import generate_apology

st.set_page_config(page_title="excuseMe.ai",page_icon="💬", layout="centered")

st.title("💬 excuseMe.ai")
st.write("Forgot your best friend's birthday? Ate your roommate’s last slice of pizza? Don't worry! This AI has your back. 🎭")
st.markdown("---")

blunder = st.text_input("Enter your social blunder:", placeholder="Forgot my mom's birthday")

if st.button("Generate Apology"):
    if blunder.strip():
        apology = generate_apology(blunder)
        st.success("🎭 Your AI-Crafted Apology 🎭")
        st.write(f"**apology")
        
        st.code(apology,language="markdown")
    else:
        st.warning("⚠️ Please enter a valid social blunder to generate an apology.")

st.markdown("---")
st.write("Crafted with 💖 by [Vanamali Sims]")