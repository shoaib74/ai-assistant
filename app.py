import os
import mlflow
import requests
import json
import tempfile
import base64
from flask import Flask, request, jsonify, render_template, session
import speech_recognition as sr
import subprocess
import time
import wave
import io

app = Flask(__name__)
app.secret_key = os.urandom(24)  # for session handling

# Initialize MLflow with error handling
try:
    mlflow.set_tracking_uri("http://localhost:5000")
except Exception as e:
    print(f"Warning: Could not connect to MLflow: {str(e)}")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Store chat history
chat_history = []

# Load the latest model configuration from MLflow
def load_latest_model_config():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("ollama-finetuning")
        if experiment is None:
            print("Warning: No MLflow experiment found")
            return None
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            print("Warning: No MLflow runs found")
            return None
        
        latest_run = runs[0]
        artifact_uri = latest_run.info.artifact_uri
        
        # Load model configuration
        config_path = os.path.join(artifact_uri, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Warning: Error loading model config: {str(e)}")
        return None

model_config = load_latest_model_config()

def text_to_speech(text):
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as temp_file:
            # Use macOS say command to generate speech in AIFF format
            subprocess.run(['say', '-o', temp_file.name, '-f', 'aiff', text], check=True)
            
            # Convert AIFF to WAV using ffmpeg
            wav_file = temp_file.name + '.wav'
            subprocess.run(['ffmpeg', '-i', temp_file.name, '-acodec', 'pcm_s16le', '-ar', '44100', wav_file], 
                         check=True, capture_output=True)
            
            # Read the WAV file
            with open(wav_file, 'rb') as audio_file:
                audio_data = audio_file.read()
                
            # Clean up temporary files
            os.unlink(temp_file.name)
            os.unlink(wav_file)
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return audio_base64
            
    except subprocess.CalledProcessError as e:
        print(f"Error generating speech: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status')
def status():
    ollama_available = check_ollama_availability()
    return jsonify({
        "ollama_status": "running" if ollama_available else "not running",
        "model_config": model_config is not None
    })

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})
        
        # Prepare the messages for the API call
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
        
        # Make request to Ollama API
        response = requests.post('http://localhost:11434/api/chat', 
                               json={
                                   "model": "llama3.2:3b",
                                   "messages": messages,
                                   "stream": False  # Disable streaming to get a single response
                               })
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if not isinstance(response_data, dict):
                    raise ValueError("Invalid response format from Ollama")
                
                assistant_message = response_data.get('message', {}).get('content', '')
                if not assistant_message:
                    return jsonify({'error': 'Empty response from model'}), 500
                
                # Add assistant response to chat history
                chat_history.append({"role": "assistant", "content": assistant_message})
                
                # Generate audio for the response
                audio_data = text_to_speech(assistant_message)
                
                return jsonify({
                    'response': assistant_message,
                    'audio': audio_data
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing Ollama response: {str(e)}")
                print(f"Raw response: {response.text}")
                return jsonify({'error': 'Invalid response from model'}), 500
            except Exception as e:
                print(f"Error processing Ollama response: {str(e)}")
                return jsonify({'error': str(e)}), 500
        else:
            error_msg = f'Failed to get response from model: {response.status_code}'
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error']
            except:
                pass
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    global chat_history
    chat_history = []
    return jsonify({'status': 'success'})

def convert_audio_to_wav(input_file, output_file):
    try:
        subprocess.run(['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_file], 
                      check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e.stderr.decode()}")
        return False

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': 'No audio file selected'}), 400

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input, \
         tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        
        try:
            # Save the uploaded audio
            audio_file.save(temp_input.name)
            
            # Convert to WAV format with specific parameters
            subprocess.run([
                'ffmpeg', '-i', temp_input.name,
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',          # 16kHz sample rate
                '-ac', '1',              # Mono
                '-y',                    # Overwrite output file
                temp_wav.name
            ], check=True, capture_output=True)
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Read the WAV file
            with sr.AudioFile(temp_wav.name) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record audio
                audio = recognizer.record(source)
                
                # Perform speech recognition
                text = recognizer.recognize_google(audio)
                
                if not text:
                    return jsonify({'error': 'No speech detected'}), 400
                    
                return jsonify({'text': text})
                
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError as e:
            return jsonify({'error': f'Error with speech recognition service: {str(e)}'}), 500
        except subprocess.CalledProcessError as e:
            print(f"Error converting audio: {e.stderr.decode()}")
            return jsonify({'error': 'Failed to convert audio format'}), 500
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_input.name)
                os.unlink(temp_wav.name)
            except Exception as e:
                print(f"Error cleaning up temporary files: {str(e)}")

def check_ollama_availability():
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def wait_for_ollama(timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_ollama_availability():
            return True
        print("Waiting for Ollama to become available...")
        time.sleep(5)
    return False

if __name__ == '__main__':
    if not wait_for_ollama():
        print("Warning: Ollama is not running. Some features may not work.")
    app.run(debug=True, port=5002)  # Using port 5002 to avoid conflicts 
