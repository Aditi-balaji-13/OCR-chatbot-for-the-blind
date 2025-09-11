import os
from huggingface_hub import InferenceClient

# Make sure you have your Hugging Face token set:
# export HF_TOKEN="your_token_here"   (in terminal)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Create inference client
client = InferenceClient(
    model="openaccess-ai-collective/gpt-oss-20b",
    token=HF_TOKEN
)

def chat_with_gptoss(prompt):
    """
    Sends a user prompt to gpt-oss-20b and returns the response.
    """
    response = client.text_generation(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return response

if __name__ == "__main__":
    print("Chatbot running with gpt-oss-20b (HuggingFace API)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break
        answer = chat_with_gptoss(user_input)
        print("Bot:", answer)
