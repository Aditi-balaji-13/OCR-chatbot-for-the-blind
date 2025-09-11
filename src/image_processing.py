import cv2
import numpy as np
import torch
from PIL import Image
import pytesseract
from transformers import AutoProcessor, AutoModelForVision2Seq

# ------------------------------
# Image Preprocessing for OCR
# ------------------------------
def preprocess_image(image_path):
    """
    Preprocesses the image to improve OCR accuracy.
    """
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding to enhance text visibility
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Detect and correct skew (deskewing)
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = thresh.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(thresh, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Apply dilation to connect broken letters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(deskewed, kernel, iterations=1)

        return Image.fromarray(dilated)
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")


def extract_text_from_image(image_path):
    """
    Extracts text from an image using Tesseract OCR with preprocessing.
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(
            processed_image, lang='eng', config="--oem 3 --psm 6"
        )

        return extracted_text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"


# ------------------------------
# Qwen2.5-VL for OCR + Description
# ------------------------------
# Load Qwen2.5-VL (vision-language model)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-2B-Instruct", use_fast=True)
qwen_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"
)

def generate_image_summary(image_path):
    """
    Generates a detailed summary of the image using Qwen2.5-VL
    (objects + scene description + OCR-like recognition).
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Create a multimodal prompt
        prompt = (
            "Describe this image in detail. "
            "List the objects you see and also transcribe any text present."
        )

        # Preprocess inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(qwen_model.device)

        # Generate description
        output = qwen_model.generate(**inputs, max_new_tokens=512)
        summary = processor.batch_decode(output, skip_special_tokens=True)[0]

        return summary
    except Exception as e:
        return f"Error generating image summary: {e}"


# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    image_path = r"/Users/aditib/Desktop/Rice/Hackathon/OpenAI_open/OCR-chatbot-for-the-blind/example/menu_example.jpg"

    # Extract text from the image
    text = extract_text_from_image(image_path)
    print("Extracted Text (Tesseract):")
    print(text)

    # Generate a scene description with Qwen2.5-VL
    summary = generate_image_summary(image_path)
    print("\nScene Description (Qwen2.5-VL):")
    print(summary)
