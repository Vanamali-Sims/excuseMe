import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 Model and Tokenizer
MODEL_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# GPT-2 doesn't have a padding token by default, so we set one
tokenizer.pad_token = tokenizer.eos_token

def generate_apology(blunder, max_new_tokens=50):
    """
    Generates a witty AI-crafted apology based on the given blunder.
    Uses GPT-2 for text generation.
    """
    # BETTER PROMPT: Gives clear structure and ensures the model generates new text
    prompt_text = f"""
    I am an expert at writing creative, witty apologies. Here are some examples:

    Blunder: "Ate my roommate’s expensive cheese without asking."
    Apology: "Dear roommate, I must confess, I have sinned. The cheese was calling my name, and in a moment of weakness, I answered. Please forgive me. I promise to replace it… with something even stinkier."

    Blunder: "Forgot my best friend's birthday."
    Apology: "Dear bestie, I have committed the ultimate betrayal. I forgot your birthday. I don’t deserve cake. Or friendship. Please allow me to make it up to you with unlimited coffee and my eternal regret."

    Blunder: "{blunder}"
    Apology:
    """

    # Encode input text
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generate response (FINAL FIXED VERSION)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,  # Fixes length issue
        temperature=0.7,  # Lower temperature for more sensible responses
        top_k=50,  # Limits to more relevant words
        top_p=0.9,  # Nucleus sampling (keeps it balanced)
        do_sample=True,  # Ensures diversity
        attention_mask=input_ids.ne(tokenizer.eos_token_id),  # FIXES ERROR
        pad_token_id=tokenizer.eos_token_id,  # Ensures padding works
        eos_token_id=tokenizer.eos_token_id  # Stops at end of response
    )

    # Decode and return the generated text
    apology_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the apology text, not the repeated examples
    if "Apology:" in apology_text:
        apology_text = apology_text.split("Apology:")[-1].strip()

    return apology_text

if __name__ == "__main__":
    print("\n💡 AI-Powered Apology Generator 💡\n")
    blunder = input("Enter your social blunder (e.g., 'Forgot my best friend's birthday'): ")

    apology = generate_apology(blunder)

    print("\n🎭 Your AI-Crafted Apology 🎭\n")
    print(apology)
    print("\n✅ Copy and send this masterpiece!\n")
