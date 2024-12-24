from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os
import json

def preprocess_text(text):
    # 소문자로 변환, 구두점 제거, 공백 정리
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 알파벳, 숫자, 공백 외 제거
    text = text.strip()
    return text


# JSON 파일에서 참조 텍스트 로드
def load_references_from_directory(reference_dir, audio_files):
    references = {}
    for audio_file in audio_files:
        # JSON 파일 경로 추정 (.wav -> .json 변환)
        json_file = os.path.join(reference_dir, audio_file.replace(".wav", ".json"))
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                json_data = json.load(f)
                # "dialogs" -> "text" 필드에서 참조 텍스트 로드
                if "dialogs" in json_data and "text" in json_data["dialogs"]:
                    references[audio_file] = json_data["dialogs"]["text"]
                else:
                    print(f"Warning: 'text' field not found in {json_file}")
        else:
            print(f"Warning: Reference JSON for {audio_file} not found.")
    return references

# BLEU 점수 계산
def calculate_bleu_from_transcriptions(reference_dir, transcription_file):
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
    reference_dir = "/experiment/data/VL_en"  # 참조 텍스트 디렉토리
    transcription_file = "/experiment/transcriptions.txt"  # ASR 결과 파일

    # BLEU 점수 계산
    calculate_bleu_from_transcriptions(reference_dir, transcription_file)
