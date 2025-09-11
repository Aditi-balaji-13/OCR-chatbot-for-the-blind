import os
from PIL import Image
import pytesseract
import torch
import cv2
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ------------------------------
# OCR Function with Preprocessing
# ------------------------------
def preprocess_image_for_ocr(image_path):
    """
    Preprocess the image for better OCR accuracy:
    - Grayscale
    - Gaussian blur
    - Adaptive threshold
    - Deskewing
    - Dilation
    """
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Deskew
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Dilation to connect letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(deskewed, kernel, iterations=1)

    # Convert back to PIL image
    return Image.fromarray(dilated)

def extract_text_from_image(image_path):
    """
    Extracts text from an image using Tesseract OCR with preprocessing.
    """
    try:
        processed_image = preprocess_image_for_ocr(image_path)
        text = pytesseract.image_to_string(
            processed_image, lang="eng", config="--oem 3 --psm 6"
        )
        return text.strip()
    except Exception as e:
        return f"Error during OCR: {e}"

# ------------------------------
# Image Summary Function (BLIP-2)
# ------------------------------
def generate_image_summary(image_path):
    """
    Generates a detailed scene description using BLIP-2.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        token = os.environ.get("HF_TOKEN")  # set your token in env
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_auth_token=False)
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", use_auth_token=False, load_in_8bit=True, device_map=None, dtype=torch.float16
            )       
        model.to(device)

        image = Image.open(image_path).convert("RGB")

        prompt = (
            "Describe this image in extreme detail. "
            "Include all objects, colors, positions, textures, interactions, "
            "overall scene, mood, and any visible text. Aim for a long and thorough description."
        )

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=500)
        summary = processor.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Error generating image summary: {e}"

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    image_path = "/Users/aditib/Desktop/Rice/Hackathon/OpenAI_open/OCR-chatbot-for-the-blind/example/menu_example.jpg"

    # OCR
    text = extract_text_from_image(image_path)
    print("OCR Text:\n", text)

    # Scene Summary
    summary = generate_image_summary(image_path)
    print("\nImage Summary (BLIP-2):\n", summary)
