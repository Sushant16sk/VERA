from flask import Flask, render_template, request, jsonify, send_from_directory
import speech_recognition as sr
from gtts import gTTS
import os
import requests
import pyttsx3
import webbrowser
import cv2
import torch
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration, ViTFeatureExtractor, GPT2Tokenizer, VisionEncoderDecoderModel
from PIL import Image
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from googletrans import Translator
import easyocr
from PyPDF2 import PdfReader
import docx
from pathlib import Path
import string
from transformers import pipeline
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
import re

# Function to read PDF files
def read_pdf(file_path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(file_path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
   
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)
   
    fp.close()
    device.close()
   
    text = retstr.getvalue()
    retstr.close()
 
    # Post-process the text to handle spacing issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Space between lower and upper case letters
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Space between digits and letters
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Space between letters and digits
    text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)    # Space after periods followed by a capital letter
 
    # Ensure consistent spacing
    text = ' '.join(text.split())
 
    return text

# Function to read DOCX files
def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Function to read TXT files
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# General function to read files
def read_file(file_path):
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.txt'):
        return read_txt(file_path)
    else:
        return "Unsupported file type."

def delete_all_audio_files():
    for filename in os.listdir(AUDIO_DIR):
        file_path = os.path.join(AUDIO_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Function to delete all uploaded files
def delete_all_uploaded_files():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
            
index = 0
AUDIO_DIR = 'audio_files'
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)
    
template_dir = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(template_dir))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load pre-trained image captioning model
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

engine = pyttsx3.init()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to extract text from an image using OCR
def ocr_read_image(image_path):
    results = reader.readtext(image_path)
    text = " ".join([result[1] for result in results])
    return text

# Global variable to store voice preference
voice_preference = 'female'

# speech_rate = 150  # Adjust this value as needed
# engine.setProperty('rate', speech_rate)

@app.route('/set_speech_rate', methods=['POST'])
def set_speech_rate():
    data = request.json
    speech_rate = data.get('rate')
    engine.setProperty('rate', speech_rate)
    return jsonify({'status': 'success', 'rate': speech_rate})


def speak(text, lang='en'):
    global voice_preference
    global index
    if voice_preference == 'male':
        voices = engine.getProperty('voices')
        for voice in voices:
            if "male" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break

        file_path = os.path.join(AUDIO_DIR, f"output_{index}.mp3")
        engine.save_to_file(text, file_path)
        try:
            engine.runAndWait()
        except RuntimeError:
            pass        
        index += 1

    elif voice_preference == 'female':
        file_path = os.path.join(AUDIO_DIR, f"output_{index}.mp3")
        tts = gTTS(text=text, lang=lang, tld='us')
        tts.save(file_path)
        index += 1

# Function to listen to the user's voice command
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio, language='en')
        print(f"You said: {query}")
        return query.lower()
    except Exception as e:
        print("Sorry, I didn't get that.")
        return ""
    
def summarize_paragraph(paragraph, model_name="facebook/bart-large-cnn", max_length=130, min_length=30):

    summarizer = pipeline("summarization", model=model_name) 
    summary = summarizer(paragraph, max_length=max_length, min_length=min_length, do_sample=False)
    summarized_text = summary[0]['summary_text']
    speak(summarized_text)
    return summarized_text

@app.route('/summarize-text', methods=['POST'])
def summarize_text_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    summarized_text = summarize_paragraph(text)
    return jsonify({'summary': summarized_text})

# Function to describe an image
def describe_image(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(image_path).convert("RGB")

    # Conditional image captioning
    text = "a picture of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Function to perform tasks based on user's command
def perform_task(command, text=""):
    # Normalize command by removing punctuation and converting to lowercase
    command = command.translate(str.maketrans('', '', string.punctuation)).lower()

    response = ""
    if "search" in command:
        search_query = command.replace("search", "").strip()
        url = f"https://www.google.com/search?q={search_query}"
        response = "Here are the search results for " + search_query
        webbrowser.open(url)

    elif "what is" in command:
        search_query = command.replace("what is", "").strip()
        url = f"https://www.google.com/search?q={search_query}"
        response = "Here are the search results for " + search_query
        webbrowser.open(url)
        
    elif "tell me something about" in command:
        topic = command.replace("tell me something about", "").strip()
        wikipedia_search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        res = requests.get(wikipedia_search_url)
        if res.status_code == 200:
            data = res.json()
            summary = data.get('extract', 'Sorry, I could not find information on that topic.')
            response = summary
        else:
            response = "Sorry, I couldn't find information on that topic."
    
    elif "read this" in command:
        response = text
        speak(response)
        return response

    elif "analyse the image" in command or "analyze the image" in command:
        response = "Please upload the image using the provided option for analysis."
        speak(response)

    elif "read the image" in command:
        response = "Please upload the image using the provided option for reading text."
        speak(response)

    elif "hi" in command or "hello" in command or "hi VERA" in command:
        response = "Hello! I am your voice assistant VERA. How can I help you?"

    elif "what is your name" in command or "who are you" in command:
        response = "Hello! My name is VERA. How can I help you?"
        
    elif "what can you do for me" in command:
        response = "Hi, I can do many things for you, like I can analyze any image for you, I can also read any text for you and summarize it for you and also I can tell you about something. So what you want me to do?"
    elif "exit" in command or "bye" in command:
        response = "Thank you and Goodbye!"
        speak(response)
        os._exit(0)
        
    elif "thank you" in command:
        response = "You are welcome, I am happy that I was able to help you"
        speak(response)
        return response
    elif "file" in command:
        response = "Please upload the file for processing."
        speak(response)
        return response
    else:
        response = "Sorry, I couldn't understand your command."
        speak(response)
        return response
    
    speak(response)
    return response

@app.route('/set-voice-preference', methods=['POST'])
def set_voice_preference():
    global voice_preference
    data = request.json
    voice_preference = data.get('voice', 'male').lower()
    return jsonify({"status": "success"})

# Route to handle the voice command
@app.route("/voice-command", methods=["POST"])
def voice_command():
    command = request.json.get("command")
    text = request.json.get("text", "")  # Default to empty string if text is not provided
    response = perform_task(command, text)
    return jsonify({"response": response})

# Route to handle image upload
@app.route("/upload-image", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"response": "No image uploaded."})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"response": "No image selected."})
    if file:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        
        command = request.form.get('command', '').lower()
        if 'read the image' in command:
            text = ocr_read_image(image_path)
            if text.strip():
                response = f"The text in the image is: {text}"
            else:
                response = "No text detected in the image."
        elif 'analyze the image' in command or 'analyse the image' in command:
            caption = describe_image(image_path)
            response = f"The image is described as: {caption}"
        else:
            response = "Invalid command for image processing."
        
        speak(response)
        return jsonify({"response": response})

@app.route('/text-command', methods=['POST'])
def text_command():
    data = request.get_json()  # Use force=True to parse JSON even if content type is not set to application/json
    text = data.get('text', '')
    response = f"Received text: {text}"
    speak(response)
    return jsonify({'response': response})


@app.route("/upload-file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"response": "No file uploaded."})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"response": "No file selected."})
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        text = read_file(file_path)
        if text.strip():
            response = f"The content of the file is: {text}"
        else:
            response = "No text detected in the file."
        return jsonify({"response": response})

@app.route('/save_audio', methods=['POST'])
def save_audio():
    global index
    file = request.files['audio']
    file_path = os.path.join(AUDIO_DIR, f'output_{index}.mp3')
    file.save(file_path)
    index += 1
    return jsonify({'index': index - 1, 'file_path': file_path})

@app.route('/play_audio/<int:index>', methods=['GET'])
def play_audio(index):
    file_path = os.path.join(AUDIO_DIR, f'output_{index}.mp3')
    if os.path.exists(file_path):
        return send_from_directory(AUDIO_DIR, f'output_{index}.mp3')
    else:
        return "File not found", 404

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text')
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'tl')
    translator = Translator()
    try:
        translated = translator.translate(text, src=source_lang, dest=target_lang)
        translated_text = translated.text
        speak(translated_text,lang=target_lang)
        return jsonify({'translated_text': translated_text, 'Your_Text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reset_index', methods=['POST'])
def reset_index():
    delete_all_audio_files()
    delete_all_uploaded_files()
    global index
    index = 0
    return "Index reset to 0", 200
    
# Home route to render the frontend
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)