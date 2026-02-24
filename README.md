# 🎵 Music Genre Classification & Recommendation App

A Streamlit web app that classifies music genres using a **YAMNet + BiLSTM** deep learning model, recommends similar songs, and builds a personalized listening profile.

---

## 🚀 Features

- **Genre Classification** — Upload any `.wav` or `.mp3` file to predict its genre across 10 classes: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.
- **Waveform Visualization** — Displays the audio waveform of your uploaded track.
- **Song-to-Song Recommendations** — Suggests 3 similar songs from the dataset based on audio embedding (cosine similarity).
- **Personalized Recommendations** — Recommends songs based on your overall listening history (user behavior embeddings).
- **Listening History** — Sidebar tracks your last 5 analyzed songs with delete support.
- **Audio Playback** — Listen to recommended songs directly in the app.
- **Model Performance Dashboard** — View Training/Validation curves, Confusion Matrix, t-SNE plot, and per-genre Recall scores.

---

## 🛠️ Prerequisites

- **Python 3.10** (required for TensorFlow compatibility)
- **GTZAN Dataset** downloaded locally at:
  `C:\Users\albia\Downloads\Music_Genre_dataset\Data\genres_original`

---

## 📦 Installation

1. **Clone / Open the project** in your terminal.

2. **Install dependencies** (if not already installed):
   ```powershell
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Ensure model files** exist in the `models/` directory:
   - `best_yamnet_bilstm.h5` — Trained Keras model
   - `label_encoder.joblib` — Label encoder for genre names

---

## ⚙️ First-Time Setup: Generate Song Embeddings

To enable song recommendations, you need to pre-compute embeddings for the entire dataset. This is a **one-time step**.

1. **Verify** your dataset path is correct in `scripts/generate_embeddings.py`:
   ```python
   DATASET_PATH = r"C:\Users\albia\Downloads\Music_Genre_dataset\Data\genres_original"
   ```

2. **Run** the embedding generation:
   ```powershell
   # Double-click:
   run_generation.bat

   # Or manually:
   .\venv\Scripts\activate
   python scripts/generate_embeddings.py
   ```
   ⏳ *This processes all songs and takes ~10–15 minutes. Saves output to `models/song_embeddings_db.joblib`.*

---

## ▶️ How to Run the App

**Option 1 — Double-click:**
> 👉 `run_app.bat`

**Option 2 — Manual (PowerShell):**
```powershell
.\venv\Scripts\streamlit.exe run frontend/main.py
```

The app will open at **http://localhost:8501**

> ⚠️ Use `venv\Scripts\streamlit.exe` directly — `streamlit` alone may not be recognized in PowerShell if it's not in your system PATH.

---

## 📊 Model Performance Plots

Place the following images (exported from your training notebook) into the `images/` folder:

| File Name | Description |
|---|---|
| `Training and validation.png` | Accuracy & Loss curves |
| `confusion matrinx.png` | Confusion Matrix (test set) |
| `tsne.png` | t-SNE projection of embeddings |
| `recall score.png` | Per-genre Recall scores |

---

## 🗂️ Project Structure

```
Music Genre Classificatioin_V1/
├── frontend/
│   ├── main.py           # Streamlit UI
│   └── styles.css        # Custom CSS
├── backend/
│   ├── inference.py      # GenreClassifier: prediction & recommendations
│   ├── user_profile.py   # Listening history & user embeddings
│   └── utils.py          # Audio loading & segmentation
├── scripts/
│   └── generate_embeddings.py  # Offline: builds song_embeddings_db.joblib
├── models/
│   ├── best_yamnet_bilstm.h5
│   ├── label_encoder.joblib
│   └── song_embeddings_db.joblib
├── images/               # Model performance plots
├── user_history.json     # Auto-generated listening history
├── run_app.bat           # Quick launch script
└── run_generation.bat    # Embedding generation script
```

---

## 🧠 How It Works

```
Audio File
  → Resampled to 16kHz (librosa)
  → Split into 5-second segments
  → YAMNet (TF Hub) → 1024-dim audio embeddings per segment
  → BiLSTM → learns temporal patterns
  → Softmax → genre probabilities

Recommendations:
  Song embedding  ──┐
                    ├── Cosine Similarity → Top-K similar songs
  User embedding  ──┘  (average of all listened songs)
```
