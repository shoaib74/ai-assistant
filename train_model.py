import mlflow
import requests
import json
import os
import sys
import time

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

def check_model_availability(model_name):
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def train_model():
    # Check if Ollama is running
    if not wait_for_ollama():
        print("Error: Ollama is not running. Please start Ollama first.")
        sys.exit(1)

    # Initialize MLflow
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("ollama-finetuning")
    except Exception as e:
        print(f"Error initializing MLflow: {str(e)}")
        sys.exit(1)

    # Model configuration
    model_name = "llama3.2:3b"

    # Check if model is available
    if not check_model_availability(model_name):
        print(f"Error: Model {model_name} is not available. Please run 'ollama pull {model_name}' first.")
        sys.exit(1)

    try:
        with mlflow.start_run():
            # Log model parameters
            mlflow.log_params({
                "model_name": model_name,
                "model_type": "ollama",
                "ollama_endpoint": "http://localhost:11434"
            })

            # Save model configuration
            model_config = {
                "model_name": model_name,
                "ollama_endpoint": "http://localhost:11434"
            }
            
            # Save configuration to file
            config_path = "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(model_config, f)

            # Log the configuration file
            mlflow.log_artifact(config_path)
            print("Successfully configured and logged model settings to MLflow")

    except Exception as e:
        print(f"Error during model configuration: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_model() 
