import json
import os
import numpy as np
from datetime import datetime

HISTORY_FILE = "user_history.json"

class UserProfile:
    def __init__(self, history_file=HISTORY_FILE):
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self):
        """Loads user history from JSON file."""
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_history(self):
        """Saves current history to JSON file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=4)
        except IOError as e:
            print(f"Error saving history: {e}")

    def add_entry(self, filename, genre, embedding):
        """
        Adds a song to the user's history.
        
        Args:
            filename (str): Name of the audio file.
            genre (str): Predicted or actual genre.
            embedding (np.array or list): The embedding vector of the song.
        """
        # Convert numpy array to list for JSON serialization
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding

        # Check if song is already in history
        for item in self.history:
            if item["filename"] == filename:
                # Update timestamp instead of adding new entry
                item["timestamp"] = datetime.now().isoformat()
                self._save_history()
                return

        entry = {
            "filename": filename,
            "genre": genre,
            "embedding": embedding_list,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history.append(entry)
        self._save_history()

    def get_history(self):
        """Returns the list of songs in history."""
        # Sort by timestamp descending (newest first)
        return sorted(self.history, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_user_embedding(self):
        """
        Calculates the 'User Embedding' by averaging the embeddings of all songs in history.
        Returns:
            np.array: The user vector (1024,), or None if history is empty.
        """
        if not self.history:
            return None

        # Extract all embeddings
        embeddings = [np.array(item["embedding"]) for item in self.history]
        
        # Calculate mean vector
        if embeddings:
            user_vector = np.mean(embeddings, axis=0)
            return user_vector
        return None

    def delete_entry(self, index):
        """Deletes an entry at the given index."""
        if 0 <= index < len(self.history):
            self.history.pop(index)
            self._save_history()

    def clear_history(self):
        """Clears the user history."""
        self.history = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
