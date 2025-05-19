import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import speech_recognition as sr
import threading
import re

# Load ImageNet classes
imagenet_classes = {i: cls for i, cls in enumerate(open("imagenet_classes.txt").read().splitlines())}

# Load pre-trained ResNet model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Function to load customizable responses
def load_custom_responses(file_path="responses.txt"):
    responses = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    responses[key.lower()] = value
    except FileNotFoundError:
        print("Warning: responses.txt not found. Using defaults.")
    return responses

custom_responses = load_custom_responses()

def reload_responses():
    global custom_responses
    custom_responses = load_custom_responses()
    insert_message("GenieBot: Custom responses reloaded successfully.", "bot")

# Function to solve math expressions
def solve_math_expression(expression):
    try:
        if re.match(r'^[0-9+\-*/(). ]+$', expression):
            result = eval(expression)
            return f"The result is: {result}"
        else:
            return "Sorry, I can only solve mathematical expressions."
    except Exception as e:
        return f"Error: {str(e)}"

# Insert messages into chat
def insert_message(message, sender):
    bubble_color = "#DCF8C6" if sender == "user" else "#F4F4F4"
    align_side = "right" if sender == "user" else "left"

    chat_display.tag_configure(sender, justify=align_side, background=bubble_color, lmargin1=10, rmargin=10, spacing3=5)
    chat_display.insert(tk.END, f"{message}\n", sender)
    chat_display.see(tk.END)


# Send user input
def send_message(event=None):
    user_input = user_entry.get().strip()
    if not user_input:
        return
    insert_message(f"You: {user_input}", "user")
    user_entry.delete(0, tk.END)

    # Check for math expressions
    if any(op in user_input for op in ['+', '-', '*', '/', '(', ')']):
        bot_response = solve_math_expression(user_input)
    else:
        bot_response = custom_responses.get(user_input.lower(), "I'm sorry, I don't understand that yet.")
    
    insert_message(f"GenieBot: {bot_response}", "bot")

# Recognize an image and display it in chat
def recognize_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            img = Image.open(file_path).convert("RGB")
            img_resized = img.resize((150, 150))
            img_tk = ImageTk.PhotoImage(img_resized)
            chat_display.image_create(tk.END, image=img_tk)
            chat_display.image_ref = img_tk  

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted_idx = outputs.max(1)
                label = imagenet_classes.get(predicted_idx.item(), "Unknown")

            insert_message(f"GenieBot: The image appears to be a {label}.", "bot")
        except Exception as e:
            insert_message(f"GenieBot: Error processing image: {str(e)}", "bot")

# Voice input
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        insert_message("GenieBot: Listening...", "bot")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            insert_message(f"You (Voice): {text}", "user")
            user_entry.delete(0, tk.END)
            user_entry.insert(0, text)
            send_message()
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            insert_message("GenieBot: Sorry, I couldn't understand that.", "bot")

def start_voice_input():
    threading.Thread(target=voice_input, daemon=True).start()

# UI Setup
app = tk.Tk()
app.title("GenieBot - Chatbot")
app.geometry("700x800")
app.configure(bg="#333333")

# Custom title bar
title_bar = tk.Frame(app, bg="#222222", relief="raised", bd=0)
title_bar.pack(fill=tk.X)
title_label = tk.Label(title_bar, text="GenieBot", fg="white", bg="#222222", font=("Helvetica", 14, "bold"))
title_label.pack(side=tk.LEFT, padx=10, pady=5)

# Chat Display
chat_frame = tk.Frame(app, bg="#FFFFFF", bd=2, relief=tk.GROOVE)
chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

chat_display = tk.Text(chat_frame, wrap=tk.WORD, bg="#FFFFFF", font=("Helvetica", 12), state=tk.NORMAL)
chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Input Frame
input_frame = tk.Frame(app, bg="#333333")
input_frame.pack(fill=tk.X, padx=10, pady=10)

user_entry = tk.Entry(input_frame, font=("Helvetica", 14), relief="flat", bg="#E8E8E8", fg="#333333", bd=2)
user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
user_entry.bind("<Return>", send_message)

# Buttons
button_style = {"bg": "#555555", "fg": "white", "font": ("Helvetica", 12), "bd": 0, "activebackground": "#777777"}

voice_button = tk.Button(input_frame, text="🎙️", command=start_voice_input, **button_style)
voice_button.pack(side=tk.RIGHT, padx=5)

image_button = tk.Button(input_frame, text="📷", command=recognize_image, **button_style)
image_button.pack(side=tk.RIGHT, padx=5)

reload_button = tk.Button(input_frame, text="🔄", command=reload_responses, **button_style)
reload_button.pack(side=tk.RIGHT, padx=5)

send_button = tk.Button(input_frame, text="➤", command=send_message, **button_style)
send_button.pack(side=tk.RIGHT, padx=5)

# Start App
insert_message("GenieBot: Hi there! How can I assist you today?", "bot")
app.mainloop()
