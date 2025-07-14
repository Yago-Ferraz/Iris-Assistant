import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from wakeword import WakeWordCNN
from joblib import load
from wakeword import WakeWordCNN  # ajuste para onde está sua classe do modelo

class Inferencia:
    """
    Classe para realizar a inferência de uma wake word em um arquivo de áudio ou em memória.
    """
    def __init__(self):
        from joblib import load
        import torch
        import torchaudio
        self.model = WakeWordCNN
        self.scaler = load("scaler.pkl")
        self.SAMPLERATE = 16000
        self.SEGMENT_DURATION_SECONDS = 3
        self.N_MFCC = 40     
        self.N_FFT = 400       
        self.HOP_LENGTH = 160  
        self.MODEL_LOAD_PATH = "models/wake_word_model_2.pth"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.SAMPLERATE,
            n_mfcc=self.N_MFCC,
            melkwargs={
                "n_fft": self.N_FFT,
                "hop_length": self.HOP_LENGTH,
                "n_mels": self.N_MFCC
            }
        )

        # carrega modelo
        
        self.model = WakeWordCNN(self.N_MFCC, self._compute_num_frames()).to(self.DEVICE)
        self.model.load_state_dict(torch.load(self.MODEL_LOAD_PATH, map_location=self.DEVICE))
        self.model.eval()

    def _compute_num_frames(self):
        import torch
        dummy_waveform = torch.randn(1, int(self.SAMPLERATE * self.SEGMENT_DURATION_SECONDS))
        dummy_mfccs = self.mfcc_transform(dummy_waveform)
        return dummy_mfccs.shape[2]

    def predict_wake_word(self, audio_input, threshold=0.5):
        """
        Detecta wake word em:
        - caminho para arquivo `.wav`
        - ou `AudioSegment`
        - ou `torch.Tensor` waveform
        """
        import torch
        import torchaudio
        import numpy as np
        import torch.nn.functional as F
        from pydub import AudioSegment
        import io

        waveform = None

        if isinstance(audio_input, str):
            # caminho para arquivo
            waveform, sr = torchaudio.load(audio_input)
        elif isinstance(audio_input, AudioSegment):
            buf = io.BytesIO()
            audio_input.export(buf, format='wav')
            buf.seek(0)
            waveform, sr = torchaudio.load(buf)
        elif isinstance(audio_input, torch.Tensor):
            # já é um waveform
            waveform = audio_input
            sr = self.SAMPLERATE
        else:
            raise ValueError("audio_input deve ser caminho, AudioSegment ou Tensor")

        # pré-processa
        mfccs = self.preprocess_audio_for_inference(waveform, sr)
        if mfccs is None:
            return None, None

        mfccs = mfccs.unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(mfccs)
            prob = torch.sigmoid(outputs).item()

        prediction = 1 if prob >= threshold else 0
        return prediction, prob

    def preprocess_audio_for_inference(self, waveform, sr):
        """
        Pré-processa o áudio já carregado como Tensor.
        """
        import torch
        import torch.nn.functional as F
        import numpy as np

        # mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # resample
        if sr != self.SAMPLERATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLERATE)
            waveform = resampler(waveform)

        expected_samples = int(self.SAMPLERATE * self.SEGMENT_DURATION_SECONDS)
        if waveform.shape[1] < expected_samples:
            padding = expected_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > expected_samples:
            waveform = waveform[:, :expected_samples]

        mfccs = self.mfcc_transform(waveform)
        mfccs_np = mfccs.squeeze(0).numpy()
        mfccs_normalized = self.scaler.transform(mfccs_np.T).T
        mfccs = torch.from_numpy(mfccs_normalized).float().unsqueeze(0)
        return mfccs
