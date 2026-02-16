# 🎵 Music Genre Classification & Recommendation App

This application classifies music genres using a **YAMNet + LSTM** deep learning model and recommends similar songs from the **GTZAN dataset**.

## 🚀 Features
-   **Audio Classification**: Upload any `.wav` or `.mp3` file to predict its genre (10 classes: Rock, Jazz, Pop, etc.).
-   **Music Recommendation**: Suggests 3 similar songs from the dataset based on audio content similarity.
-   **Audio Playback**: Listen to the uploaded track and the recommended songs directly in the app.
-   **Visualizations**: View Waveforms, Confidence Scores, and Model Performance plots.

## 🛠️ Prerequisites
-   **Python 3.10** (Required for TensorFlow compatibility)
-   **GTZAN Dataset**: Downloaded locally (e.g., in `Downloads/Music_Genre_dataset/Data/genres_original`).

## 📦 Installation

1.  **Project Setup**:
    The project comes with a virtual environment (`venv`) set up for Python 3.10.

2.  **Install Dependencies**:
    If you haven't already:
    ```powershell
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Model Files**:
    Ensure the following files are in the `model/` directory:
    -   `best_yamnet_bilstm.h5` (Trained Keras Model)
    -   `label_encoder.joblib` (Label Encoder)

## ⚙️ Data Setup (First Time Only)
To enable recommendations and playback, you must generate the song database from your local dataset.

1.  **Configure Path**:
    Ensure your dataset is at `C:\Users\albia\Downloads\Music_Genre_dataset\Data\genres_original` (or update `generate_embeddings_local.py`).

2.  **Generate Embeddings**:
    Double-click **`run_generation.bat`**
    *This processes all songs (takes ~10-15 mins) and saves `model/song_embeddings_db.joblib`.*

## ▶️ How to Run
Simply double-click:
👉 **`run_app.bat`**

Or run manually in terminal:
```powershell
.\venv\Scripts\activate
streamlit run frontend/main.py
```
The app will open at `http://localhost:8501`.

## 📊 Performance Plots
To view training metrics (Accuracy, Loss, Confusion Matrix), place your saved images from Colab into the `images/` folder with these names:
-   `Training and validation.png`
-   `confusion matrinx.png`
-   `recall score.png`
-   `tsne.png`
