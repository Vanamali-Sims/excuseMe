import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

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
    # IMPROVED PROMPT: Forces GPT-2 to generate a clean, structured response
    prompt_text = f"""
    Apology Generator:
    
    Blunder: {blunder}
    Apology: "I sincerely apologize
    """

    # Encode input text
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generate response (FINAL FIXED VERSION)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,  # Limit length
        temperature=0.6,  # Lower temp = more controlled output
        top_k=50,  # Keeps it focused
        top_p=0.9,  # Better word selection
        do_sample=True,
        attention_mask=input_ids.ne(tokenizer.eos_token_id),  # Fixes issue
        pad_token_id=tokenizer.eos_token_id,  # Ensures padding works
        eos_token_id=tokenizer.eos_token_id  # Stops at end of response
    )

    # Decode and return only the generated sentence
    apology_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the generated apology part
    if "Apology:" in apology_sentence:
        apology_sentence = apology_sentence.split("Apology:")[-1].strip()
    
    # Ensure the output is a clean single sentence
    apology_sentence = apology_sentence.split(".")[0] + "."

    return format_apology(blunder, apology_sentence)


def format_apology(blunder, generated_sentence):
    """
    Wraps the generated apology in a structured template for better readability.
    """
    templates = [
        f"Dear {get_target_audience(blunder)}, {generated_sentence} Please accept my deepest regrets and a coffee as a peace offering.",
        f"My sincerest apologies, {get_target_audience(blunder)}. {generated_sentence} I promise to make it up to you with cake and eternal servitude.",
        f"I messed up, {get_target_audience(blunder)}. {generated_sentence} I feel terrible, and I will go above and beyond to fix it!",
        f"Oops, my bad, {get_target_audience(blunder)}. {generated_sentence} I owe you an apology and probably a pizza.",
        f"Hey {get_target_audience(blunder)}, {generated_sentence} Can we forget this happened if I bribe you with food?",
        f"Dear {get_target_audience(blunder)}, {generated_sentence} I'll be making amends by being extra nice for the next 48 hours. No complaints allowed!"
    ]

    return random.choice(templates)


def get_target_audience(blunder):
    """
    Infers the person affected by the blunder.
    """
    keywords = {
        "mom": "Mom",
        "friend": "bestie",
        "boss": "Boss",
        "roommate": "roommate",
        "partner": "love",
        "girlfriend": "girlfriend",
        "boyfriend": "boyfriend",
        "professor": "Professor",
        "dad": "Dad",
        "teacher": "Teacher"
    }
    
    for keyword, audience in keywords.items():
        if keyword in blunder.lower():
            return audience

    return "everyone"


if __name__ == "__main__":
    print("\n💡 AI-Powered Apology Generator 💡\n")
    blunder = input("Enter your social blunder (e.g., 'Forgot my best friend's birthday'): ")

    apology = generate_apology(blunder)

    print("\n🎭 Your AI-Crafted Apology 🎭\n")
    print(apology)
    print("\n✅ Copy and send this masterpiece!\n")
