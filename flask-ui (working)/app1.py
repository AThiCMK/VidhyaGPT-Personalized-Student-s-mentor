from flask import Flask, render_template, request, jsonify, send_file, Response
from vidhya_core import initialize_conversation_bot, talk_with_bot, translate_audio_hindi_to_english
import os
import time
import json
from scipy.io import wavfile
import numpy as np
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = 'Data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
query_engine = initialize_conversation_bot()

@app.route('/')
def home():
    return render_template('index1.html', timestamp=int(time.time()))

@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    if 'audio' not in request.files:
        print("Audio file missing in request")
        return jsonify({'error': 'No audio part in the request'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        print("No filename found in uploaded file")
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the audio file temporarily
        temp_filepath = r'K:\EttA\temp_input.wav'
        audio_file.save(temp_filepath)
        print("File saved successfully:", temp_filepath)

        def generate():
            try:
                translated_text = translate_audio_hindi_to_english(temp_filepath)
                yield json.dumps({"type": "translated_text", "content": translated_text}) + "\n"
            except Exception as e:
                print("Error in translating audio:", e)
                yield json.dumps({"error": "Translation failed"}) + "\n"
                return

            try:
                response_stream = talk_with_bot(query_engine, translated_text)
                for response_chunk in response_stream:
                    yield response_chunk
            except Exception as e:
                print("Error in bot response:", e)
                yield json.dumps({"error": "Bot response generation failed"}) + "\n"

        # Return a streaming response
        return Response(generate(), content_type="text/event-stream")
    except Exception as e:
        print("Error in processing audio:")
        return jsonify({'error': 'Failed to process audio'}), 500

    
if __name__ == '__main__':
    app.run(debug=False)

