import os
import re
import sys

import sounddevice as sd
import soundfile as sf
import torch
from transformers import MusicgenForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoProcessor, logging

# set the log level to be CRITICAL
logging.set_verbosity(logging.CRITICAL)

# switch to GPU if possible
if torch.cuda.is_available():
    print('GPU available:', torch.cuda.get_device_name(), sep=" ")
    device = torch.device("cuda")
else:
    print('CPU available')
    device = torch.device("cpu")


# a review find on Amazon use as the example input text
data = sys.argv[1]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)

music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)


# Emotion labels as per GoEmotions dataset (positive, neutral, negative)
emotion_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", 
                  "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", 
                  "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", 
                  "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", 
                  "neutral"]

# Functions
# Preprocess input text by cleaning and normalizing.
# Remove special characters and extra spaces
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# Predict emotion from text and improve accuracy by using logits and thresholding.
def predict_emotion(text, threshold=0.5):
    preprocessed_text = preprocess_text(text)

    # Tokenize input
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Get model output
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Extract the predicted emotion
    top_prob, top_indices = torch.topk(probabilities, k=3)

    # Filter based on a threshold for better accuracy
    top_emotions = []
    for prob, idx in zip(top_prob[0], top_indices[0]):
        if prob > threshold:
            top_emotions.append((emotion_labels[idx], prob.item()))

    if top_emotions:
        return top_emotions
    else:
        # Return neutral if no emotion passes the threshold
        return [("neutral", 1.0)]

# Generate music based on the extracted emotion.
def generate_music_from_emotion(emotion):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    inputs = processor(
        text=["music that sounds " + emotion[0][0]],
        padding=True,
        return_tensors="pt"
    ).to(device)

    generated_audio = music_model.generate(**inputs, do_sample=True)

    sampling_rate = music_model.config.audio_encoder.sampling_rate

    # Save audio to a temporary file
    file_name = f"{emotion}_music.wav"
    sf.write(file_name, generated_audio[0, 0].cpu().numpy(), sampling_rate)
    return file_name

# Play the generated music and delete the file after playback.
def play_and_delete_music(file_name):
    try:

        # Read the file
        data, samplerate = sf.read(file_name)
        # Play the file
        sd.play(data, samplerate)
        sd.wait()  # Wait until playback finishes
    finally:
        # Delete the file after playback
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"File {file_name} deleted.")

# RUN

# Generate emotion based on input text
predicted_emotions = predict_emotion(data, threshold=0.3)

print(f"Generating music based on current emotion: {predicted_emotions[0][0]}")

# Generate music based on emotion
music_file = generate_music_from_emotion(predicted_emotions)
print(f"Generated music file: {music_file}")

# Play the music and delete the file after playing
play_and_delete_music(music_file)
