import sys
import os
import numpy as np
import librosa
import tensorflow as tf


# =====================
# Constants (same as training)
# =====================

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
MAX_FRAMES = 130

EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


# =====================
# Preprocessing (same as training)
# =====================

def trim_silence(audio):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
    return trimmed_audio


def extract_log_mel(audio, sr):

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad / truncate
    if log_mel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]

    return log_mel


# =====================
# Load Model
# =====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ser_cnn_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)


# =====================
# Prediction Function
# =====================

def predict_emotion(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    audio = trim_silence(audio)

    mel = extract_log_mel(audio, sr)

    mel = mel[np.newaxis, ..., np.newaxis]

    prediction = model.predict(mel)[0]

    index = np.argmax(prediction)

    emotion_code = str(index + 1).zfill(2)

    emotion = EMOTION_MAP[emotion_code]

    confidence = np.max(prediction) * 100

    return emotion, confidence


# =====================
# Main (Command Line Part)
# =====================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]

    emotion, confidence = predict_emotion(audio_file)

    print(f"Predicted Emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}%")
