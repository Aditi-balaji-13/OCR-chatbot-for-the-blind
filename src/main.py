import os
from image_processing import extract_text_from_image, generate_image_summary
from chatbot import chat_with_gptoss

if __name__ == "__main__":
    image_path = input("📷 Enter image file path: ").strip()

    if not os.path.exists(image_path):
        print("❌ File not found!")
        exit()

    # Step 1: OCR
    print("\n🔎 Extracting text from image...")
    extracted_text = extract_text_from_image(image_path)
    summary = generate_image_summary(extracted_text)
    print("📝 Image Summary:", summary)
    print("📝 OCR Text:\n", extracted_text if extracted_text else "[No text detected]")

    # Step 2: Chat with GPT-OSS
    if extracted_text and summary:
        print("Chatbot running with gpt-oss-20b (HuggingFace API)\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Bot: Goodbye!")
                break
            prompt = f"The image text says: {extracted_text}. the summary of image is {summary} Answer this question {user_input}.")
            answer = chat_with_gptoss(user_input)
            print("Bot:", answer)

        print("\n🤖 Generating response from GPT-OSS-20B...")
        response = chat_with_gptoss(f"The image text says: {extracted_text}. the summary of image is {summary} Explain or expand this in detail.")
        print("\n💬 GPT-OSS Response:\n", response)
    else:
        print("\n⚠️ No text found in the image to send to GPT-OSS.")
