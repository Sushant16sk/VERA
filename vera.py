import speech_recognition as sr
from gtts import gTTS
import os
import webbrowser
import requests

import cv2
import torch
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration, ViTFeatureExtractor, GPT2Tokenizer, VisionEncoderDecoderModel
from PIL import Image
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

import pvporcupine
import pyaudio
import struct

# Function to convert text to speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    #os.system("mpg321 output.mp3")  # For Linux
    os.system("start output.mp3")  # For Windows

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

# Function to summarize text
def summarize_text(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# Load pre-trained image captioning model
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

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

# Function to analyze an image
def analyze_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Describe the image
    describe_image(image_path)



# Function to perform tasks based on user's command
def perform_task(command):
    if "search" in command:
        search_query = command.replace("search", "").strip()
        url = f"https://www.google.com/search?q={search_query}"
        speak("Here are the search results for " + search_query)
        webbrowser.open(url)

    elif "what is" in command:
        search_query = command.replace("what is", "").strip()
        url = f"https://www.google.com/search?q={search_query}"
        speak("Here are the search results for " + search_query)
        webbrowser.open(url)
    elif "tell me something about" in command:
        topic = command.replace("tell me something about", "").strip()
        wikipedia_search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        response = requests.get(wikipedia_search_url)
        if response.status_code == 200:
            data = response.json()
            summary = data.get('extract', 'Sorry, I could not find information on that topic.')
            speak(summary)
        else:
            speak("Sorry, I couldn't find information on that topic.")
    elif "read this" in command:
        speak("Ok, what do you want me to read? Please enter your text.")
        text_to_read = input("Enter your text: ")  # Wait for user input
        speak("Sure, here is the text you provided.")
        speak(text_to_read)
        speak("Now, let me summarize what is happening in the text.")
        summary = summarize_text(text_to_read)
        speak(summary)

    elif "analyse this image" in command:
        speak("Please enter the path to the image.")
        image_path = input("Enter the path to the image: ")  # Wait for user input
        caption = describe_image(image_path)
        speak("The image shows: " + caption)

    elif "analyze this image" in command:
        speak("Please enter the path to the image.")
        image_path = input("Enter the path to the image: ")  # Wait for user input
        caption = describe_image(image_path)
        speak("The image shows: " + caption)
        
    elif "hi" in command:
        speak("Hello! I am your voice assistant VERA. How can I help you?")
    elif "hello" in command:
        speak("Hello! I am your voice assistant VERA. How can I help you?")
    elif "what is your name" in command:
        speak("Hello! My name is VERA. How can I help you?")
    elif "exit" in command:
        speak("Thank you and Goodbye!")
        exit()
    elif "bye" in command:
        speak("Thank you and Goodbye!")
        exit()
    elif "thank you" in command:
        speak("You are welcome, I am happy that I was able to help you")
        exit()
    else:
        speak("Sorry, I couldn't understand your command.")

# Main function to run the voice assistant
def main():
    speak("Hello! I am your voice assistant VERA. How can I help you?")
    while True:
        command = listen()
        perform_task(command)

# Function to detect wake word and activate the assistant
def detect_wake_word():
    porcupine = pvporcupine.create(
        access_key='DLD/YzdT8niR7l7htvXaApn2avn7ad1JtZYMTC/RQo2rlzZ63Zf7Sw==',
        keyword_paths=['C:/Users/SushantKumar/Downloads/ok-VERA_en_windows_v3_0_0.ppn']
    )
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for wake word...")
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("Wake word detected!")
            speak("Hello! I am your voice assistant VERA. How can I help you?")
            command = listen()
            perform_task(command)

# Main function to run the voice assistant
def main():
    detect_wake_word()

if __name__ == "__main__":
    main()