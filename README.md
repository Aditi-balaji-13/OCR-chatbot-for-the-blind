# 👁️‍🗨️ Vision-Powered Chatbot for the Blind

This project combines **OCR (Optical Character Recognition)** and **Image Summarization** with a **GPT-OSS-20B chatbot** to create an **accessible AI assistant for visually impaired users**.  

---

## 🌟 Motivation

Blind and visually impaired users often struggle to interpret visual information such as:  
- Menus  
- Street signs  
- Product labels  
- Screenshots or documents  

This system allows users to **upload an image**, automatically **extract text**, generate a **rich scene summary**, and **interact with the content through chat**.

---

## 🧩 How It Works

### 1. OCR – Extract Text
- Uses **Tesseract OCR** with preprocessing for better accuracy:  
  - Grayscale conversion  
  - Noise reduction with Gaussian blur  
  - Adaptive thresholding  
  - Deskewing rotated text  
  - Dilation to sharpen letters  
- Extracts any readable text from the image.  

**Example:**  
```
Input Image → "examples/menu_exampple.png"
```

---

### 2. Image Summarization – Describe Scene
- For images with **little or no text**, a **vision-language model** (BLIP) generates a detailed description of the scene.  

**Example:**  
```
Input Image → "A plate of spaghetti pasta with tomato sauce, served with basil leaves and a fork on the right side."
```

---

### 3. GPT-OSS-20B Chatbot – Conversational Layer
- Combines **OCR text + image summary** into a **chat context**.  
- Powered by the open-source **GPT-OSS-20B** model via Hugging Face API.  
- Users can **ask questions** about the image in natural language:  
  - "What’s on this menu?"  
  - "How much does the spaghetti cost?"  
  - "What objects are in this picture?"

---

## 🔄 Workflow

image -> get OCR and summary -> get chatbot context --> get answers to questions from the chatbot 

# 🛠 Installation & Usage Guide

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ocr-gptoss-chatbot.git
cd ocr-gptoss-chatbot
```

## 2️⃣ Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 3️⃣ Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu:**
```bash
sudo apt install tesseract-ocr -y
```

**Windows:**
Download the installer from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and add it to your system PATH.

## 4️⃣ Set Hugging Face Token

**macOS/Linux:**
```bash
export HF_TOKEN="your_hf_token_here"
```

**Windows:**
```bash
setx HF_TOKEN "your_hf_token_here"
```

---

## ▶️ Run the Chatbot
```bash
python main.py
```

When prompted:
```arduino
📷 Enter image file path: ./example/menu_example.jpg
```

---

## 💡 What It Does
- 🔎 Extracts text from the image using OCR
- 📝 Summarizes image content (if needed)
- 🤖 Generates a conversational response via GPT-OSS-20B

---

## 🧪 Example Output
```vbnet
🔎 Extracting text from image...
📝 OCR Text:
Spaghetti Bolognese - $12.99

🤖 Generating response from GPT-OSS-20B...

💬 GPT-OSS Response:
This appears to be a restaurant menu item. It describes a delicious plate of spaghetti with tomato sauce and garnishes...
```

---
