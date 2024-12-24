from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import librosa
import torch
import os
import json
from tqdm import tqdm
import re

# ASR 모델 및 프로세서 로드
def load_asr_model(device):
    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    return processor, model

# 오디오 파일 로드 및 전처리
def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=16000)  # 16kHz로 샘플링
    return audio

# ASR을 사용해 음성을 텍스트로 변환
def transcribe_audio(file_path, processor, model, device):
    try:
        audio = load_audio(file_path)
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription.strip()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""  # 빈 텍스트 반환

# 디렉토리 내 모든 오디오 파일에 대해 ASR 수행
def transcribe_directory(audio_dir, output_file, processor, model, device):
    results = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    for file_name in tqdm(audio_files, desc="Processing Audio Files", unit="file"):
        file_path = os.path.join(audio_dir, file_name)
        transcription = transcribe_audio(file_path, processor, model, device)
        if transcription:  # 빈 결과 건너뛰기
            results.append((file_name, transcription))
        else:
            print(f"Warning: Skipping {file_name} due to empty transcription.")

    # 결과 저장
    with open(output_file, "w") as f:
        for file_name, transcription in results:
            f.write(f"{file_name}\t{transcription}\n")

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 알파벳, 숫자, 공백 외 제거
    text = text.strip()
    return text

# JSON 파일에서 참조 텍스트 로드
def load_references_from_directory(reference_dir, audio_files):
    references = {}
    for audio_file in audio_files:
        json_file = os.path.join(reference_dir, audio_file.replace(".wav", ".json"))
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                json_data = json.load(f)
                if "dialogs" in json_data and "text" in json_data["dialogs"]:
                    references[audio_file] = json_data["dialogs"]["text"]
                else:
                    print(f"Warning: 'text' field not found in {json_file}")
        else:
            print(f"Warning: Reference JSON for {audio_file} not found.")
    return references

# BLEU 점수 계산
def calculate_bleu_from_transcriptions(reference_dir, transcription_file):
    # Hypothesis 텍스트 로드
    hypotheses = {}
    with open(transcription_file, "r") as hyp_file:
        for line in hyp_file.readlines():
            if "\t" not in line.strip():
                print(f"Warning: Skipping invalid line: {line.strip()}")
                continue
            file_name, transcription = line.strip().split("\t")
            hypotheses[file_name] = preprocess_text(transcription).split()

    # Reference 텍스트 로드
    references = load_references_from_directory(reference_dir, hypotheses.keys())

    # Reference와 Hypothesis 매칭
    ref_texts = [[preprocess_text(references[file_name]).split()] for file_name in hypotheses.keys() if file_name in references]
    hyp_texts = [hyp for file_name, hyp in hypotheses.items() if file_name in references]

    # BLEU 점수 계산
    chencherry = SmoothingFunction()
    bleu_score = corpus_bleu(ref_texts, hyp_texts, smoothing_function=chencherry.method1)
    print(f"Corpus BLEU Score: {bleu_score:.4f}")

# 실행 함수
if __name__ == "__main__":
    # 경로 설정
    audio_directory = "/experiment/output_wavs"  # ASR 입력 오디오 파일 디렉토리
    output_transcriptions = "/experiment/transcriptions.txt"  # ASR 출력 파일
    reference_dir = "/experiment/data/VL_en"  # 참조 텍스트가 저장된 디렉토리

    # GPU 사용 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ASR 모델 및 프로세서 로드
    processor, model = load_asr_model(device)

    # 1. 오디오 파일에서 텍스트 추출
    transcribe_directory(audio_directory, output_transcriptions, processor, model, device)

    # 2. BLEU 점수 계산
    calculate_bleu_from_transcriptions(reference_dir, output_transcriptions)
