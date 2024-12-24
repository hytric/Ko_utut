# Vocoder Training and Inference Guide

## **Vocoder Config**

- [[Ko config link](https://drive.google.com/file/d/1rEnXZzrJJnrJmumDpBF67vQBAksadaFD/view?usp=sharing)]
- [[En config link](https://drive.google.com/file/d/1xVc6SNmicGRGEN15eQMOD64k4yiJ1cH_/view?usp=sharing)]

## **How to Use**

Repository: [https://github.com/facebookresearch/speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis)

Use the code available in the above repository for vocoder tasks.

---

## **1. Create Manifest File**

Place `create_manifest.py` in the `fairseq` folder.

```bash
python scripts/create_manifest.py \
    --wav_dir /audio/dataset/kr/wav/ko \
    --units_dir /audio/dataset/kr/units_2nd/units/ko \
    --manifest_path /audio/unit2audio/speech-resynthesis/train_manifest.json \
    --code_type hubert

python scripts/create_manifest.py \
    --wav_dir /audio/dataset/kr/wav/en \
    --units_dir /audio/dataset/kr/units_2nd/units/en \
    --manifest_path /audio/unit2audio/speech-resynthesis/train_en_manifest.json \
    --code_type hubert
```

---

## **2. Split Validation Data**

Place `split_manifest.py` in the `fairseq` folder.

```bash
python scripts/split_manifest.py \
    --input_manifest /audio/unit2audio/speech-resynthesis/train_en_manifest.json \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --output_dir /audio/unit2audio/speech-resynthesis/split_en_manifests \
    --seed 42
```

---

## **3. Configure JSON File**

Create a `config.json` file in the following directory:

```
/audio/unit2audio/speech-resynthesis/examples/speech_to_speech_translation/configs
```

Example `config.json`:

```json
{
    "input_training_file": "/audio/unit2audio/speech-resynthesis/split_manifests/train_manifest.json",
    "input_validation_file": "/audio/unit2audio/speech-resynthesis/split_manifests/val_manifest.json",
    "resblock": "1",
    "num_gpus": 1,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    "upsample_rates": [5,4,4,2,2],
    "upsample_kernel_sizes": [11,8,8,4,4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "num_embeddings": 100,
    "embedding_dim": 128,
    "model_in_dim": 128,
    "segment_size": 8960,
    "code_hop_size": 320,
    "f0": false,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "dur_prediction_weight": 1.0,
    "dur_predictor_params": {
        "encoder_embed_dim": 128,
        "var_pred_hidden_dim": 128,
        "var_pred_kernel_size": 3,
        "var_pred_dropout": 0.5
    },
    "sampling_rate": 16000,
    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": null,
    "num_workers": 4,
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "env://"
    }
}
```

---

## **4. Training**

### Korean Model

```bash
python examples/speech_to_speech_translation/train.py \
    --checkpoint_path checkpoints/ko_hubert_vocoder \
    --config examples/speech_to_speech_translation/configs/hubert100_dw1.0.json \
    --training_epochs 2000 \
    --training_steps 500000 \
    --stdout_interval 5 \
    --checkpoint_interval 50000 \
    --summary_interval 100 \
    --validation_interval 5000 \
    --fine_tuning False
```

### English Model

```bash
python examples/speech_to_speech_translation/train.py \
    --checkpoint_path checkpoints/en_hubert_vocoder \
    --config examples/speech_to_speech_translation/configs/hubert100_dw1.0en.json \
    --training_epochs 2000 \
    --training_steps 500000 \
    --stdout_interval 5 \
    --checkpoint_interval 50000 \
    --summary_interval 100 \
    --validation_interval 5000 \
    --fine_tuning False
```

---

## **5. Inference**

### Example 1

```bash
python -m examples.speech_to_speech_translation.inference \
    --checkpoint_file /home/jhkim/audio/unit2audio/speech-resynthesis/checkpoints/ko_hubert_vocoder_4th \
    -n 20 \
    --output_dir generations \
    --num-gpu 1 \
    --input_code_file /speech-resynthesis/val_ko_manifest_4th.json \
    --dur-prediction
```

### Example 2

```bash
python -m examples.speech_to_speech_translation.inference \
    --checkpoint_file /home/jhkim/audio/unit2audio/speech-resynthesis/checkpoints/ko_hubert_vocoder \
    -n 10 \
    --output_dir generations \
    --num-gpu 1 \
    --input_code_file /speech-resynthesis/split_ko_manifests/val_manifest.json \
    --dur-prediction
