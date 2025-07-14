import os
from pydub import AudioSegment
import tqdm
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler


class WakeWordDataset(Dataset):
    def __init__(self, root_dir, segment_duration_seconds, samplerate, n_mfcc, n_fft, hop_length):
        self.root_dir = root_dir
        self.segment_duration_seconds = segment_duration_seconds
        self.samplerate = samplerate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.audio_files = []
        self.labels = [] # 0 para negativo, 1 para positivo

        # Carrega os caminhos dos arquivos e seus rótulos
        self._load_audio_paths()

        # Inicializa o transformador MFCC
        # MFCCs são extraídos de espectrogramas Mel-scale, que imitam a percepção humana do som.
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.samplerate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mfcc # Geralmente n_mels é igual a n_mfcc
            }
        )
        # Normalizador para as características MFCC (opcional, mas recomendado)
        self.scaler = None # Será ajustado após carregar todos os dados para calcular média/desvio padrão

    def _load_audio_paths(self):
        """
        Carrega os caminhos de todos os arquivos de áudio e seus rótulos
        das pastas 'positive' e 'negative'.
        """
        positive_dir = os.path.join(self.root_dir, "positiva normalizada")
        negative_dir = os.path.join(self.root_dir, "negativa normalizada")

        # Carrega amostras positivas (rótulo 1)
        if os.path.exists(positive_dir):
            for filename in os.listdir(positive_dir):
                if filename.endswith(".wav"):
                    self.audio_files.append(os.path.join(positive_dir, filename))
                    self.labels.append(1)
            print(f"Carregadas {len([f for l, f in zip(self.labels, self.audio_files) if l == 1])} amostras positivas.")
        else:
            print(f"Aviso: Diretório de amostras positivas não encontrado: {positive_dir}")

        # Carrega amostras negativas (rótulo 0)
        if os.path.exists(negative_dir):
            for filename in os.listdir(negative_dir):
                if filename.endswith(".wav"):
                    self.audio_files.append(os.path.join(negative_dir, filename))
                    self.labels.append(0)
            print(f"Carregadas {len([f for l, f in zip(self.labels, self.audio_files) if l == 0])} amostras negativas.")
        else:
            print(f"Aviso: Diretório de amostras negativas não encontrado: {negative_dir}")

        if not self.audio_files:
            raise RuntimeError("Nenhum arquivo de áudio encontrado nos diretórios especificados. Verifique os caminhos e se os arquivos .wav existem.")

    def _pad_audio(self, waveform):
        """
        Preenche ou corta o waveform para a duração esperada.
        Isso é crucial para garantir que todas as entradas MFCC tenham as mesmas dimensões.
        """
        expected_num_samples = int(self.samplerate * self.segment_duration_seconds)
        if waveform.shape[1] < expected_num_samples:
            # Preenche com zeros se o áudio for muito curto
            padding = expected_num_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > expected_num_samples:
            # Corta se o áudio for muito longo
            waveform = waveform[:, :expected_num_samples]
        return waveform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Carrega o waveform
        # torchaudio.load retorna (waveform, sample_rate)
        waveform, sr = torchaudio.load(audio_path)

        # Garante que o áudio seja mono (já deve ser do pré-processamento, mas é uma verificação)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Garante que a taxa de amostragem esteja correta (já deve ser do pré-processamento)
        if sr != self.samplerate:
            # Isso não deveria acontecer se o pré-processamento foi feito corretamente
            # Mas é uma salvaguarda.
            resampler = torchaudio.transforms.Resample(sr, self.samplerate)
            waveform = resampler(waveform)

        # Preenche ou corta o áudio para a duração esperada
        waveform = self._pad_audio(waveform)

        # Extrai as características MFCC
        # O mfcc_transform espera um tensor (canais, amostras)
        mfccs = self.mfcc_transform(waveform)

        # Normaliza as MFCCs usando o scaler ajustado
        # O scaler espera (num_frames, num_mfccs), então precisamos transpor
        # mfccs.shape é (n_mfcc, num_frames)
        if self.scaler is not None:
            mfccs_np = mfccs.squeeze(0).numpy() # Remove o dim de canal (se houver) e converte para numpy
            mfccs_normalized = self.scaler.transform(mfccs_np.T).T # Normaliza e volta para (n_mfcc, num_frames)
            mfccs = torch.from_numpy(mfccs_normalized).float().unsqueeze(0) # Adiciona dim de canal de volta

        return mfccs, torch.tensor(label, dtype=torch.float32)

    def fit_scaler(self):
        """
        Calcula a média e o desvio padrão das MFCCs de todo o dataset
        para normalização. Isso deve ser chamado ANTES de criar os DataLoaders.
        """
        print("Ajustando normalizador (scaler) para as MFCCs...")
        all_mfccs = []
        for i in range(len(self)):
            # Carrega o áudio sem normalização ainda
            audio_path = self.audio_files[i]
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.samplerate:
                resampler = torchaudio.transforms.Resample(sr, self.samplerate)
                waveform = resampler(waveform)
            waveform = self._pad_audio(waveform)

            mfccs = self.mfcc_transform(waveform)
            all_mfccs.append(mfccs.squeeze(0).numpy()) # Remove dim de canal e adiciona

        # Concatena todos os MFCCs e ajusta o scaler
        all_mfccs_concatenated = np.concatenate(all_mfccs, axis=1).T # Transpõe para (num_frames_total, n_mfcc)
        self.scaler = StandardScaler()
        self.scaler.fit(all_mfccs_concatenated)
        print("Normalizador ajustado.")



   



class normalizate:
    def __init__(self, input_filepath , output_directory, segment_duration_ms=3000):
        self.input_filepath = input_filepath
        self.output_directory = output_directory
        self.segment_duration_ms = segment_duration_ms

    def normalizar(self):
        for k in tqdm.tqdm(os.listdir(self.input_filepath), desc="Normalizando arquivos de áudio", unit="arquivo"):
            self.preprocess_and_segment_audio(self.input_filepath + "/" + k, self.output_directory)

    def preprocess_and_segment_audio(self,input_filepath, output_directory):
        """
        Processa um arquivo de áudio:
        1. Converte para mono.
        2. Resample para 16kHz.
        3. Fatia em segmentos de duração fixa.

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Diretório de saída criado: {output_directory}")

        try:
            # Carrega o arquivo de áudio
          
            audio = AudioSegment.from_file(input_filepath)

            # 1. Converte para mono (se já não for)
            if audio.channels != 1:
              
                audio = audio.set_channels(1)

            # 2. Resample para 16kHz (se já não for)
            if audio.frame_rate != 16000:
              
                audio = audio.set_frame_rate(16000)

            

            # 3. Fatiar em segmentos de duração fixa
            total_length_ms = len(audio)
            num_segments = 0

            # Obtém o nome base do arquivo de entrada sem extensão
            base_filename = os.path.splitext(os.path.basename(input_filepath))[0]

           
            for i in range(0, total_length_ms, self.segment_duration_ms):
                start_ms = i
                end_ms = i + self.segment_duration_ms

                # Garante que o último segmento não ultrapasse o final do áudio
                if end_ms > total_length_ms:
                    if (total_length_ms - start_ms) < (self.segment_duration_ms * 0.3):
                        
                        continue
                    else:
                        # Se o segmento final for significativo, preenchemos com silêncio
                        segment = audio[start_ms:total_length_ms]
                        padding_needed = self.segment_duration_ms - len(segment)
                        if padding_needed > 0:
                            # Cria silêncio (0 volume) e adiciona ao segmento
                            silent_segment = AudioSegment.silent(duration=padding_needed, frame_rate=audio.frame_rate)
                            segment += silent_segment
                            
                else:
                    segment = audio[start_ms:end_ms]

                num_segments += 1
                output_filename = os.path.join(output_directory, f"{base_filename}_segment_{num_segments:04d}.wav")
                segment.export(output_filename, format="wav")
             

        

        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em '{input_filepath}'. Verifique o caminho.")
        except Exception as e:
            print(f"Ocorreu um erro durante o processamento: {e}")
            print("Certifique-se de que FFmpeg está instalado e no seu PATH.")

    
    def process_direct(self, audio_input):
        """
        Processa um áudio (arquivo ou AudioSegment em memória)
        e devolve uma lista com os segmentos processados.
        """

        segmentos = None

        try:
            # Carrega o áudio
            if isinstance(audio_input, str):
                audio = AudioSegment.from_file(audio_input)
            elif isinstance(audio_input, AudioSegment):
                audio = audio_input
            else:
                raise ValueError("audio_input deve ser caminho de arquivo ou AudioSegment")

            # 1. Converte para mono
            if audio.channels != 1:
                audio = audio.set_channels(1)

            # 2. Resample para 16kHz
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)

            # 3. Fatiar em segmentos
            total_length_ms = len(audio)
            num_segments = 0

            base_filename = "in_memory_audio"

            for i in range(0, total_length_ms, self.segment_duration_ms):
                start_ms = i
                end_ms = i + self.segment_duration_ms

                if end_ms > total_length_ms:
                    if (total_length_ms - start_ms) < (self.segment_duration_ms * 0.3):
                        continue
                    else:
                        segment = audio[start_ms:total_length_ms]
                        padding_needed = self.segment_duration_ms - len(segment)
                        if padding_needed > 0:
                            silent_segment = AudioSegment.silent(duration=padding_needed, frame_rate=audio.frame_rate)
                            segment += silent_segment
                else:
                    segment = audio[start_ms:end_ms]

                num_segments += 1
             

            return segment

        except Exception as e:
            print(f"Ocorreu um erro durante o processamento: {e}")
            print("Certifique-se de que FFmpeg está instalado e no seu PATH.")
            return []



