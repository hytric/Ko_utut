# Transformer Data Preprocessing and Training Guide

## **1. HuBERT Inference**

Convert a multilingual dataset into units using a pre-trained HuBERT model.

Place the following files into the `fairseq` folder:
- `inference.py`
- `util.py`

Run the inference script:

```bash
cd fairseq
python inference.py \
    --in-wav-path <wav-file-path> \
    --out-unit-path <unit-output-path> \
    --mhubert-path <model-parameter-path> \
    --kmeans-path <k-means-file-path>
```

---

## **2. Save Unit File Paths to Text Files**

Create text files listing `.unit` files for each language.

```bash
find /units/en/ -maxdepth 1 -name '*.unit' | sort > en_files.txt
find /units/es/ -maxdepth 1 -name '*.unit' | sort > es_files.txt
find /units/fr/ -maxdepth 1 -name '*.unit' | sort > fr_files.txt
```

---

## **3. Prepare Transformer Training Data**

Organize dataset files into the following structure:

```
dataset/
└── units/
    ├── en/
    ├── ko/
    └── fr/
```

Use the following scripts for data preparation:
- `unit_txt_gen.py`
- `unit_TR_preprocessing.py`

Ensure the number of files for each language matches (**parallel corpus**).

### **Command-line Arguments for Preprocessing Script**

```python
parser = argparse.ArgumentParser(description='Multilingual Data Preparation Script')
parser.add_argument('--base_dir', type=str, default='/home/jskim/audio/dataset/units',
                    help='Input data directory path')
parser.add_argument('--output_base_dir', type=str, default='/home/jskim/audio/dataset/units',
                    help='Output data directory path')
parser.add_argument('--languages', type=str, nargs='+', default=['en', 'es', 'fr'],
                    help='List of languages to process (e.g., en es fr)')
parser.add_argument('--train_ratio', type=float, default=0.9,
                    help='Training data ratio (0.0 to 1.0)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed value')
args = parser.parse_args()
```

Ensure files match this pattern:

```bash
filename_pattern = re.compile(r'(.+)_([a-z]{2})_(\d+)\.unit$')
```

### **Expected Output Files**
After running the script, you should have:
- `train.src`
- `train.tgt`
- `valid.src`
- `valid.tgt`

---

## **4. Fairseq Data Preprocessing**

Convert data into Fairseq-compatible format:

```bash
fairseq-preprocess \
  --source-lang src \
  --target-lang tgt \
  --trainpref "/path/to/train" \
  --validpref "/path/to/valid" \
  --destdir "/path/to/data-bin" \
  --workers 4 \
  --joined-dictionary \
  --dataset-impl 'mmap'
```

- A single multilingual HuBERT model uses **one dictionary file (`dict.txt`)**.

---

## **5. Transformer Training**

Run Fairseq training with the following command:

```bash
fairseq-train '/shared/home/milab/users/jskim/multilingual' \
  --arch transformer \
  --share-decoder-input-output-embed \
  --encoder-layers 12 \
  --decoder-layers 12 \
  --encoder-embed-dim 512 \
  --decoder-embed-dim 512 \
  --encoder-attention-heads 8 \
  --decoder-attention-heads 8 \
  --encoder-ffn-embed-dim 2048 \
  --decoder-ffn-embed-dim 2048 \
  --max-tokens 24000 \
  --update-freq 2 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --lr 5e-4 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --dropout 0.3 \
  --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-epoch 50 \
  --save-dir '/shared/home/milab/users/jskim/results' \
  --patience 10 \
  --tensorboard-logdir '/shared/home/milab/users/jskim/tensorboard_logs' \
  --max-source-positions 3000 \
  --max-target-positions 3000 \
  --amp
```

### **Parameter Explanation:**
- **`--arch`:** Transformer model architecture.
- **`--max-tokens`:** Maximum number of tokens per batch.
- **`--optimizer`:** Optimization algorithm.
- **`--lr`:** Learning rate.
- **`--warmup-updates`:** Number of warmup steps.
- **`--save-dir`:** Directory to save model checkpoints.
- **`--tensorboard-logdir`:** Directory for TensorBoard logs.
- **`--amp`:** Enable Automatic Mixed Precision for better GPU utilization.

---

Your Transformer training pipeline is now ready. Adjust parameters if needed for optimal results!
