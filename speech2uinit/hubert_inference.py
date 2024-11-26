import argparse
import tqdm
import joblib
import torch
import os
from glob import glob
import librosa
import tempfile
import soundfile as sf

from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)

'''
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_ko_1_1 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/ko --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_ko_1_2 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/ko --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_ko_2 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/ko --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_ko_3 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/ko --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_ko_5 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/ko --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km

python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_en_1_1 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/en --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_en_1_2 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/en --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_en_2 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/en --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_en_3 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/en --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km
python inference.py --in-wav-path /home/jhkim/audio/dataset/kr/train/ko/TS_en_5 --out-unit-path /home/jhkim/audio/dataset/kr/units/train/en --mhubert-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/checkpoint_best.pt --kmeans-path /home/jhkim/audio/audio2unit/pretrained/ckpt_2nd/train.km

'''

def process_units(units, reduce=False):
    if not reduce:
        return units

    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out

def save_unit(unit, unit_path):
    os.makedirs(os.path.dirname(unit_path), exist_ok=True)
    with open(unit_path, "w") as f:
        f.write(unit)

def save_speech(speech, speech_path, sampling_rate=16000):
    os.makedirs(os.path.dirname(speech_path), exist_ok=True)
    sf.write(
        speech_path,
        speech,
        sampling_rate,
    )

def load_model(model_path, kmeans_path, use_cuda=False):
    hubert_reader = HubertFeatureReader(
        checkpoint_path=model_path,
        layer=11,
        use_cuda=use_cuda,
    )
    kmeans_model = joblib.load(open(kmeans_path, "rb"))
    kmeans_model.verbose = False

    return hubert_reader, kmeans_model

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    hubert_reader, kmeans_model = load_model(args.mhubert_path, args.kmeans_path, use_cuda=use_cuda)

    in_wav_paths = glob(os.path.join(args.in_wav_path, "*.wav"))
    out_unit_paths = [os.path.join(args.out_unit_path, os.path.splitext(os.path.basename(p))[0] + ".unit") for p in in_wav_paths]

    for in_wav_path, out_unit_path in tqdm.tqdm(
        zip(in_wav_paths, out_unit_paths),
        total=len(in_wav_paths)
    ):
        with sf.SoundFile(in_wav_path) as f:
            sr = f.samplerate

        if sr != 16000:
            speech, _ = librosa.load(in_wav_path, sr=16000)
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                sf.write(tmp.name, speech, 16000)
                feats = hubert_reader.get_feats(tmp.name)
        else:
            feats = hubert_reader.get_feats(in_wav_path)

        feats = feats.cpu().numpy()

        pred = kmeans_model.predict(feats)
        pred_str = " ".join(str(p) for p in pred)

        save_unit(pred_str, out_unit_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-wav-path", type=str, required=True, help="The path to the directory where the audio input is stored"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, help="The path to the directory where the unit output will be stored"
    )
    parser.add_argument(
        "--mhubert-path",
        type=str,
        required=True,
        help="Checkpoint pre-trained mHuBERT models"
    )
    parser.add_argument(
        "--kmeans-path",
        type=str,
        required=True,
        help="Path to the K-means model file to use for inference",
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
