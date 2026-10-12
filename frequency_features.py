import os
import librosa
import numpy as np

def extract_features(audio_dir):
    features = []
    labels = []

    for category in os.listdir(audio_dir):
        category_path = os.path.join(audio_dir, category)

        if not os.path.isdir(category_path):
            continue  # Skip files

        print(f"🔍 Processing category: {category}")

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if not file.lower().endswith(".wav"):
                print(f"❌ Skipping non-audio file: {file}")
                continue

            try:
                y, sr = librosa.load(file_path, sr=None)
                fft = np.abs(np.fft.fft(y))[:1000]
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                feature_vector = np.concatenate([fft[:50], [centroid, bandwidth, zcr]])
                features.append(feature_vector)
                labels.append(category)
            except Exception as e:
                print(f"⚠️ Failed to process {file_path}: {e}")

    return features, labels
