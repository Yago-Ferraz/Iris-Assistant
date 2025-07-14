import sounddevice as sd
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment
from dataset import normalizate
from inferencia import Inferencia
import webrtcvad
import tempfile
import whisper
import os
import requests
import pyttsx3
import pygame
import time
from dotenv import load_dotenv

# Configurações
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
fs = 16000   # taxa de amostragem (Hz)
wake_word_duration = 3  # segundos para gravar wake word
vad_level = 0  # sensibilidade do VAD
max_silence_frames = 50  # para detectar fim da fala
frame_duration_ms = 30
samples_per_frame = int(fs * frame_duration_ms / 1000)

# Inicializa modelos
audio_obj = normalizate(None, "datas/")
model = whisper.load_model("turbo")
infer = Inferencia()
vad = webrtcvad.Vad(vad_level)



url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": API_KEY
}

def play_audio_pygame(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    # Espera até a música terminar de tocar
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def gravar_audio(duracao, fs):
    print(f"Gravando áudio por {duracao} segundos...")
    audio = sd.rec(int(duracao * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Gravação finalizada.")
    return audio

def processar_audio(audio):
    audio_bytes_io = io.BytesIO()
    sf.write(audio_bytes_io, audio, fs, format='WAV')
    audio_bytes_io.seek(0)
    audio_segment = AudioSegment.from_file(audio_bytes_io, format='wav')
    return audio_obj.process_direct(audio_segment)

def detectar_wake_word():
    audio = gravar_audio(wake_word_duration, fs)
    audio_processado = processar_audio(audio)
    pred, prob = infer.predict_wake_word(audio_processado)
    print(f"Predição wake word: {pred}, Probabilidade: {prob:.4f}")
    return pred == 1
def falar_texto(texto):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 225)
    engine.say(texto)
    engine.runAndWait()

def gravar_fala_ate_silencio():
    #aqui deveria ter um audio de wake word
    play_audio_pygame("sound/ui-sounds-pack-2-sound-5-358890.mp3")
    print("Fale agora...")
    frames = []
    silence_counter = 0

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=samples_per_frame) as stream:
        while True:
            frame, _ = stream.read(samples_per_frame)
            frame_bytes = frame.flatten().tobytes()
            if vad.is_speech(frame_bytes, fs):
                silence_counter = 0
                frames.append(frame.copy())
                print(".", end="", flush=True)
            else:
                silence_counter += 1
                if silence_counter > max_silence_frames:
                    break

    print("\nFim da fala.")

    if not frames:
        print("Nenhuma fala detectada.")
        return None

    audio_np = np.concatenate(frames, axis=0)
    audio_float32 = audio_np.astype(np.float32) / 32768.0
    return audio_float32.squeeze()


def transcrever_audio(audio_float32):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    sf.write(tmp_name, audio_float32, fs)
    result = model.transcribe(tmp_name, language="pt")
    os.remove(tmp_name)
    return result["text"]

def chamar_gemini_api(prompt_text):
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text":f"Você é a Iris, uma assistente virtual inteligente e eficiente. Seu objetivo é ajudar o usuário com respostas claras e úteis. No entanto, você tem um toque de sarcasmo meio acido, que usa ocasionalmente para tornar a conversa mais leve e descontraída. Sempre entenda que o usuário espera tanto informação precisa quanto um pouco de personalidade na resposta. Não responda com muitos elementos de texto pois a resposta sera lida por uma voz sintetica nao use * ou qualquer elemento visual e responda brevemente {prompt_text}"
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        resposta = response.json()
        texto = resposta['candidates'][0]['content']['parts'][0]['text']
        return texto
    else:
        print(f"Erro na API Gemini: {response.status_code} - {response.text}")
        return None

def sessao_ativa(prompt_text):
    """
    Enquanto houver fala a cada <=2s, a sessão continua.
    Quando silêncio >2s, a sessão termina.
    """
    
    while True:
        fala = gravar_fala_ate_silencio()
        if fala is None:
            print("Nenhuma fala detectada por mais de 2s. Encerrando sessão.\n")
            return prompt_text

        texto = transcrever_audio(fala)
        prompt_text += f" usuario: {texto}"
        print("Você disse:", texto)

        resposta = chamar_gemini_api(prompt_text)
        if resposta:
            prompt_text += f" assistente: {resposta}"
            falar_texto(str(resposta))
            print("Assistente:", resposta)


def main_loop():
    prompt_text = ''
    print("Aguardando o wake word...")
    while True:
        if detectar_wake_word():
            print("Wake word detectada. Sessão iniciada!")
            prompt_text = sessao_ativa(prompt_text)
            print("\nAguardando novo wake word...\n")
        else:
            print("Wake word não detectado. Tentando novamente...")


if __name__ == "__main__":
    main_loop()
