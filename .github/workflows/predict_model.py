import openai
import requests

# Constants
DATA_URL = 'https://raw.githubusercontent.com/threeniinesix/G3/main/train.txt'
DATA_PATH = 'train.txt'

# Download the dataset
def download_dataset(url, path):
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    with open(path, 'wb') as file:
        file.write(response.content)
    print(f"Dataset downloaded and saved to {path}")

# Download the dataset
download_dataset(DATA_URL, DATA_PATH)

# Load the dataset
with open(DATA_PATH, 'r') as file:
    data = file.read().splitlines()

# Prepare prompts
def prepare_prompt(history):
    prompt = "Given the following history of numbers, predict the next number: " + ", ".join(history)
    return prompt

# Set up OpenAI API key
openai.api_key = 'your-api-key'

# Function to get prediction from GPT-4
def get_prediction(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",  # Use the appropriate engine for GPT-4
        prompt=prompt,
        max_tokens=10
    )
    prediction = response.choices[0].text.strip()
    return prediction

# Simulate predictions using the dataset
history_window = 10  # Number of past entries to consider for the prompt
predictions = []

for i in range(history_window, len(data)):
    history = data[i-history_window:i]
    prompt = prepare_prompt(history)
    prediction = get_prediction(prompt)
    predictions.append(prediction)
    print(f"History: {history}")
    print(f"Prediction: {prediction}")

# Save predictions to a file
with open('predictions.txt', 'w') as file:
    for prediction in predictions:
        file.write(prediction + '\n')

print("Predictions saved to 'predictions.txt'")
