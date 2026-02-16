import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import joblib
import os
from .utils import load_audio, segment_audio

# Define the custom layer needed for loading the model
class YAMNetFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yamnet = hub.KerasLayer(
            "https://tfhub.dev/google/yamnet/1",
            trainable=False
        )

    def call(self, inputs):
        def apply_yamnet(waveform):
            _, embeddings, _ = self.yamnet(waveform)
            return embeddings  # (frames, 1024)

        return tf.map_fn(
            apply_yamnet,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, 1024), dtype=tf.float32)
        )

class GenreClassifier:
    def __init__(self, model_path, encoder_path, db_path=None):
        """
        Initializes the classifier with model and label encoder.
        Optionally loads a song database for recommendations.
        """
        self.model = self._load_model(model_path)
        self.encoder = self._load_encoder(encoder_path)
        self.song_db = self._load_db(db_path) if db_path else None

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        return tf.keras.models.load_model(
            path,
            custom_objects={"YAMNetFeatureExtractor": YAMNetFeatureExtractor}
        )

    def _load_encoder(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoder file not found at {path}")
        return joblib.load(path)

    def _load_db(self, path):
        if not os.path.exists(path):
            print(f"Warning: Song database not found at {path}")
            return None
        db = joblib.load(path)
        
        # FIX: Handle 3D embeddings (N, Frames, 1024) -> (N, 1024)
        if "embeddings" in db:
            embs = db["embeddings"]
            if len(embs.shape) == 3:
                # Average over the frames dimension (axis 1)
                print(f"Flattening 3D embeddings from {embs.shape}...")
                db["embeddings"] = np.mean(embs, axis=1)
                
        return db

    def recommend(self, query_embedding, top_k=5):
        """
        Finds similar songs using cosine similarity.
        """
        if self.song_db is None:
            return []

        # Database embeddings shape: (N_songs, 1024)
        db_embeddings = self.song_db["embeddings"]
        filenames = self.song_db["filenames"]
        genres = self.song_db["genres"]

        # Calculate Cosine Similarity
        # Sim(A, B) = (A . B) / (||A|| * ||B||)
        
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0: return []
        query_vec = query_embedding / query_norm

        # Normalize DB (pre-calculating this in init would be faster, but doing here for simplicity)
        db_norms = np.linalg.norm(db_embeddings, axis=1)
        # Avoid division by zero
        db_norms[db_norms == 0] = 1 
        db_vecs = db_embeddings / db_norms[:, np.newaxis]

        # Dot product
        similarities = np.dot(db_vecs, query_vec)

        # Get top K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                "filename": filenames[idx],
                "genre": genres[idx],
                "score": similarities[idx]
            })
            
        return recommendations

    def predict(self, audio_path, top_k=3):
        """
        Predicts genre for a given audio file.
        Returns (predictions, embedding)
        """
        # Load and segment audio
        audio = load_audio(audio_path)
        segments = segment_audio(audio)

        if not segments:
            return [], None

        segments_array = np.array(segments) # (num_segments, AUDIO_SAMPLES)
        
        # Get embeddings first (we need them for recommendation)
        # We can use the intermediate layer output
        embedding_model = tf.keras.Model(
            self.model.input, 
            self.model.get_layer("yamnet_extractor").output
        )
        
        # Get embeddings for each segment
        segment_embeddings = embedding_model.predict(segments_array) # (num_segments, Frames, 1024) likely
        
        # FIX: Handle 3D output (Segments, Frames, 1024) -> (1024,)
        if len(segment_embeddings.shape) == 3:
             # Average over segments (0) AND frames (1)
             mean_embedding = np.mean(segment_embeddings, axis=(0, 1)) 
        else:
             mean_embedding = np.mean(segment_embeddings, axis=0)
        
        # Now get predictions from the full model
        predictions = self.model.predict(segments_array) # (num_segments, num_classes)
        mean_pred = np.mean(predictions, axis=0) # (num_classes,)
        
        # Get top K results
        top_indices = np.argsort(mean_pred)[::-1][:top_k]
        top_genres = self.encoder.inverse_transform(top_indices)
        top_scores = mean_pred[top_indices]

        return list(zip(top_genres, top_scores)), mean_embedding
