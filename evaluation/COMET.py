from comet import download_model, load_from_checkpoint
import os
import json

def preprocess_text(text):
    return text.lower().strip()

# COMET 모델 로드
def load_comet_model():
    model_path = download_model("Unbabel/wmt20-comet-da")
    model = load_from_checkpoint(model_path)
    return model

# JSON 파일에서 참조 텍스트 로드
def load_references_from_directory(reference_dir, audio_files):
    references = {}
    for audio_file in audio_files:
        # JSON 파일 경로 추정 (.wav -> .json 변환)
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

# COMET 점수 계산
def calculate_comet_scores(reference_dir, transcription_file, comet_model):
    # Hypothesis 텍스트 로드
    hypotheses = {}
    with open(transcription_file, "r") as hyp_file:
        for line in hyp_file.readlines():
            if "\t" not in line.strip():
                print(f"Warning: Skipping invalid line: {line.strip()}")
                continue
            file_name, transcription = line.strip().split("\t")
            hypotheses[file_name] = transcription

    # Reference 텍스트 로드
    references = load_references_from_directory(reference_dir, hypotheses.keys())

    # 데이터 준비
    data = []
    for file_name, hypothesis in hypotheses.items():
        if file_name in references:
            data.append({
                "src": "",  # 원본 소스는 비워둠
                "mt": preprocess_text(hypothesis),  # Hypothesis
                "ref": preprocess_text(references[file_name])  # Reference
            })

    # COMET 점수 계산
    predictions = comet_model.predict(data, batch_size=8, gpus=1)

    # 점수 추출 및 평균 계산
    if isinstance(predictions, dict) and "scores" in predictions:
        scores = predictions["scores"]
        average_score = sum(scores) / len(scores) if scores else 0
        print(f"Average COMET Score: {average_score:.4f}")
        return average_score
    else:
        print(f"Error: Unexpected predictions format: {predictions}")
        return 0

# 실행 함수
if __name__ == "__main__":
    # 경로 설정
    reference_dir = "/experiment/data/VL_en"  # 참조 텍스트가 저장된 디렉토리
    transcription_file = "/experiment/transcriptions.txt"  # ASR 결과 파일

    # COMET 모델 로드
    comet_model = load_comet_model()

    # COMET 점수 계산
    calculate_comet_scores(reference_dir, transcription_file, comet_model)
