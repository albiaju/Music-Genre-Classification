import os
import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATASET_PATH = r"C:\Users\albia\Downloads\Music_Genre_dataset\Data\genres_original"
MODEL_PATH = os.path.join("model", "best_yamnet_bilstm.h5")
OUTPUT_PATH = os.path.join("model", "song_embeddings_db.joblib")
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5  # seconds
AUDIO_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION
ANALYZE_DURATION = 30 # Analyze first 30 seconds for better accuracy

# --- MODEL LOADING ---
class YAMNetFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yamnet = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)

    def call(self, inputs):
        def apply_yamnet(waveform):
            _, embeddings, _ = self.yamnet(waveform)
            return embeddings
        return tf.map_fn(apply_yamnet, inputs, 
                         fn_output_signature=tf.TensorSpec(shape=(None, 1024), dtype=tf.float32))

print("⏳ Loading model...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"YAMNetFeatureExtractor": YAMNetFeatureExtractor}
    )
    
    # Extract just the YAMNet part for embeddings
    yamnet_layer = model.get_layer("yamnet_extractor")
    embedding_model = tf.keras.Model(model.input, yamnet_layer.output)
    
    # Trace the function for performance (handling variable batch size)
    @tf.function
    def predict_embedding(segment):
        return embedding_model(segment)
        
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --- EMBEDDING GENERATION ---
def get_file_embedding(file_path):
    try:
        # Load audio (first 30s)
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=ANALYZE_DURATION)
        
        segments = []
        for start in range(0, len(audio) - AUDIO_SAMPLES + 1, AUDIO_SAMPLES):
            segments.append(audio[start:start + AUDIO_SAMPLES])
            
        if not segments: return None
        
        segments_array = np.array(segments)
        
        # Predict embeddings (Batch, Frames, 1024)
        embeddings_3d = predict_embedding(segments_array)
        
        # Flatten: Average over Batch (0) and Frames (1) -> (1024,)
        # Use .numpy() to convert tensor to array
        embeddings_val = embeddings_3d.numpy()
        
        if len(embeddings_val.shape) == 3:
            avg_embedding = np.mean(embeddings_val, axis=(0, 1))
        else:
            avg_embedding = np.mean(embeddings_val, axis=0)
            
        return avg_embedding
        
    except Exception as e:
        # print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

# --- MAIN LOOP ---
song_database = {
    "filenames": [],
    "embeddings": [],
    "genres": [],
    "dataset_path": DATASET_PATH # Store path for reference
}

print(f"🚀 Starting embedding generation from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset folder not found! Please check the path.")
    exit(1)

genres = sorted(os.listdir(DATASET_PATH))
total_files = 0

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    if not os.path.isdir(genre_path): continue
    
    files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
    
    for file in tqdm(files, desc=f"Processing {genre}"):
        file_path = os.path.join(genre_path, file)
        emb = get_file_embedding(file_path)
        
        if emb is not None:
            song_database["filenames"].append(file)
            song_database["embeddings"].append(emb)
            song_database["genres"].append(genre)
            total_files += 1

# Convert and Save
if total_files > 0:
    song_database["embeddings"] = np.array(song_database["embeddings"])
    joblib.dump(song_database, OUTPUT_PATH)
    print(f"\n✅ Done! Processed {total_files} songs.")
    print(f"💾 Saved database to: {OUTPUT_PATH}")
else:
    print("\n❌ No songs processed. Check if your .wav files are valid.")
