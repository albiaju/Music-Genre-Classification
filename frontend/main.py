import streamlit as st
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd

# Add parent directory to path to import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.inference import GenreClassifier

# Page Configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css(os.path.join(os.path.dirname(__file__), "styles.css"))

# Title and Description
st.title("🎵 Music Genre Classifier")
st.markdown("""
<div style="text-align: center;">
    Upload an audio file (WAV or MP3) to identify its music genre using a YAMNet + LSTM model.
</div>
""", unsafe_allow_html=True)

# Paths to model and encoder
MODEL_PATH = os.path.join("model", "best_yamnet_bilstm.h5")
ENCODER_PATH = os.path.join("model", "label_encoder.joblib")
DB_PATH = os.path.join("model", "song_embeddings_db.joblib")

@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        st.error(f"Model files not found! Please ensure '{MODEL_PATH}' and '{ENCODER_PATH}' exist.")
        return None
    try:
        # Load with database if it exists
        return GenreClassifier(MODEL_PATH, ENCODER_PATH, db_path=DB_PATH if os.path.exists(DB_PATH) else None)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

classifier = load_classifier()

# File Uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None and classifier:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Waveform Visualization")
        try:
            # Load for visualization (librosa)
            y, sr = librosa.load(tmp_file_path, sr=None)
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error visualizing audio: {e}")

    with col2:
        st.subheader("Genre Prediction")
        
        if st.button("Analyze Audio"):
            with st.spinner("Analyzing..."):
                try:
                    # Predict results AND embedding
                    predictions, embedding = classifier.predict(tmp_file_path)
                    
                    if predictions:
                        top_genre, top_score = predictions[0]
                        st.success(f"**Top Prediction:** {top_genre} ({top_score:.2%})")
                        
                        # Create a dataframe for the chart
                        df = pd.DataFrame(predictions, columns=["Genre", "Confidence"])
                        st.bar_chart(df.set_index("Genre"))
                        
                        st.write("### Detailed Results")
                        for genre, score in predictions:
                            st.write(f"- **{genre}**: {score:.4f}")
                        
                        # RECOMMENDATION SECTION
                        if embedding is not None:
                            st.markdown("---")
                            st.subheader("🎶 Recommended Songs")
                            recommendations = classifier.recommend(embedding, top_k=3)
                            
                            # Define local dataset path
                            DATASET_PATH = r"C:\Users\albia\Downloads\Music_Genre_dataset\Data\genres_original"
                            
                            if recommendations:
                                for i, rec in enumerate(recommendations, 1):
                                    col_info, col_audio = st.columns([3, 2])
                                    
                                    with col_info:
                                        st.write(f"**{i}. {rec['filename']}**")
                                        st.caption(f"Genre: {rec['genre']} | Similarity: {rec['score']:.2f}")
                                    
                                    with col_audio:
                                        # Construct full path
                                        song_path = os.path.join(DATASET_PATH, rec['genre'], rec['filename'])
                                        if os.path.exists(song_path):
                                            st.audio(song_path)
                                        else:
                                            st.warning("File not found locally.")
                            else:
                                st.info("No recommendations available (Database not loaded).")

                    else:
                        st.warning("Could not process audio.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                finally:
                    # cleanup temp file
                     try:
                        os.remove(tmp_file_path)
                     except:
                        pass

# Model Performance Section
st.markdown("---")
st.header("📊 Model Performance")

tabs = st.tabs(["Accuracy & Loss", "Confusion Matrix", "t-SNE Plot", "Recall"])

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")

def display_plot(filename, caption):
    path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.info(f"Plot not found: `{filename}`. Please save the plot from your notebook to the `images/` folder.")

with tabs[0]:
    st.subheader("Training History")
    # User provided a single image which likely contains both plots or is one of them.
    # Displaying it once here.
    display_plot("Training and validation.png", "Training & Validation Metrics")

with tabs[1]:
    st.subheader("Confusion Matrix")
    display_plot("confusion matrinx.png", "Confusion Matrix (Test Set)")

with tabs[2]:
    st.subheader("t-SNE Visualization")
    display_plot("tsne.png", "t-SNE Projection of Embeddings")

with tabs[3]:
    st.subheader("Per-Genre Recall")
    display_plot("recall score.png", "Recall per Genre")

