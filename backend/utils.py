import librosa
import numpy as np

SAMPLE_RATE = 16000
SEGMENT_DURATION = 5  # seconds
AUDIO_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION

def segment_audio(audio):
    """
    Splits audio into 5-second segments.
    """
    segments = []
    total_len = len(audio)

    # Note: the original loop range might exclude the last bit if it's less than AUDIO_SAMPLES.
    # We might want to pad the last chunk, but following the original logic for now which discards remainders
    # or requires exact chunks.
    # The original loop: range(0, total_len - AUDIO_SAMPLES + 1, AUDIO_SAMPLES)
    
    for start in range(0, total_len - AUDIO_SAMPLES + 1, AUDIO_SAMPLES):
        segment = audio[start:start + AUDIO_SAMPLES]
        segments.append(segment)

    return segments

def load_audio(file_path):
    """
    Loads audio file with target sample rate.
    """
    # Load with librosa
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio
