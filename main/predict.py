import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model


# =============================
# CONSTANTS (Same as Training)
# =============================

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


# =============================
# LOAD TRAINED MODEL
# =============================

MODEL_PATH = "ser_cnn_model.h5"

try:
    model = load_model(MODEL_PATH)
except:
    print("❌ Error: Model file not found!")
    print("Make sure 'ser_cnn_model.h5' is in the same folder.")
    sys.exit(1)


# =============================
# PREPROCESSING FUNCTIONS
# =============================

def trim_silence(audio):
    trimmed, _ = librosa.effects.trim(audio, top_db=20)
    return trimmed


def extract_log_mel(audio, sr):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad or trim
    if log_mel.shape[1] < MAX_FRAMES:
        pad = MAX_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]

    return log_mel


# =============================
# PREDICTION FUNCTION
# =============================

def predict_emotion(file_path):

    # Load audio
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    # Remove silence
    audio = trim_silence(audio)

    # Extract features
    mel = extract_log_mel(audio, sr)

    # Reshape for CNN
    mel = mel[np.newaxis, ..., np.newaxis]

    # Predict
    prediction = model.predict(mel)[0]

    # Get emotion
    idx = np.argmax(prediction) + 1
    emotion = EMOTION_MAP[str(idx).zfill(2)]

    confidence = np.max(prediction) * 100

    print(f"\n🎯 Predicted Emotion: {emotion}")
    print(f"📊 Confidence: {confidence:.2f}%\n")


# =============================
# ONE-LINE EXECUTION
# =============================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("  python predict.py audio.wav\n")
        sys.exit(1)

    audio_file = sys.argv[1]

    predict_emotion(audio_file)
