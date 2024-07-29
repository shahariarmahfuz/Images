import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import threading
import time
import requests
from PIL import Image
import pytesseract
from io import BytesIO
from collections import deque
import logging

app = Flask(__name__)

# Set API key directly
GEMINI_API_KEY = "AIzaSyCfOha3zR71EaHfCzSJoxrtZXcAaIBqZ1w"

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_sessions = {}  # Dictionary to store chat sessions per user

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def extract_text_from_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from image: {str(e)}")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    user_id = data.get('user_id')

    if not query or not user_id:
        return jsonify({"error": "Please provide both query and user_id parameters."}), 400

    if user_id not in chat_sessions:
        chat_sessions[user_id] = {
            "chat": model.start_chat(history=[]),
            "history": deque(maxlen=5)  # Stores the last 5 messages
        }

    chat_session = chat_sessions[user_id]["chat"]
    history = chat_sessions[user_id]["history"]

    # Add the user query to history
    history.append(f"User: {query}")
    try:
        response = chat_session.send_message(query)
        # Add the bot response to history
        history.append(f"Bot: {response.text}")
    except Exception as e:
        app.logger.error(f"Error in chat session: {str(e)}")
        return jsonify({"error": "Failed to get response from chat session."}), 500

    return jsonify({"response": response.text})

@app.route('/ask_with_image', methods=['POST'])
def ask_with_image():
    data = request.json
    query = data.get('query')
    user_id = data.get('user_id')
    image_url = data.get('image_url')

    if not query or not user_id or not image_url:
        return jsonify({"error": "Please provide query, user_id, and image_url parameters."}), 400

    if user_id not in chat_sessions:
        chat_sessions[user_id] = {
            "chat": model.start_chat(history=[]),
            "history": deque(maxlen=5)  # Stores the last 5 messages
        }

    chat_session = chat_sessions[user_id]["chat"]
    history = chat_sessions[user_id]["history"]

    # Download and extract text from image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
        extracted_text = extract_text_from_image(image_data)
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

    # Add the user query to history
    history.append(f"User: {query} (Image Text: {extracted_text})")
    
    # Prepare the message with the extracted text
    if extracted_text.strip():
        message_with_text = f"{query} [Extracted Text: {extracted_text}]"
    else:
        message_with_text = f"{query} [Image Description Needed]"
    
    # Send the query with the extracted text
    try:
        response = chat_session.send_message(message_with_text)
        # Add the bot response to history
        history.append(f"Bot: {response.text}")
    except Exception as e:
        app.logger.error(f"Error in chat session with image: {str(e)}")
        return jsonify({"error": "Failed to get response from chat session with image."}), 500

    return jsonify({"response": response.text})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

def keep_alive():
    url = "http://localhost:8080/ping"  # Replace with your actual URL
    while True:
        time.sleep(600)  # Ping every 10 minutes
        try:
            response = requests.get(url)
            if response.status_code == 200:
                app.logger.info("Ping successful")
            else:
                app.logger.warning("Ping failed with status code %d", response.status_code)
        except Exception as e:
            app.logger.error("Ping failed with exception: %s", str(e))

if __name__ == '__main__':
    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
