# HuBERT Guide

## **KoEn HuBERT Pretrained Model**
- [[HuBERT Model Link](https://drive.google.com/file/d/1w9fNZ1-Np1RurPKUtjYXJRtTBpVvcWrp/view?usp=sharing)]
- [[KM File Link](https://drive.google.com/file/d/1huzDxhoMlRFiZOxmC8lTkUp-jB3CpM1r/view?usp=sharing)]

---

## **HuBERT Specifications**
- **Parameters:**
  - 1st Iteration: K=100, 250K Steps
  - 2nd, 3rd Iteration: K=500, 400K Steps
  - Adam Optimizer
- **Mask Probability:** 0.5
- **Dropout:** 0.3
- **Learning Rate:** 0.001
- **Warmup Steps:** 10,000

---

## **Preprocessing Steps**

### **Step 1: Manifest Creation**
Audio files should have durations between **10 to 30 seconds**.

Install the Python soundfile library and run:

```bash
python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

- **`/path/to/waves`:** Directory containing audio files.
- **`--dest`:** Path to save the manifest.
- **`--ext`:** File extension (e.g., `wav`, `flac`).
- **`--valid-percent`:** Validation data ratio (e.g., `0.01` = 1%).

---

### **Step 2: MFCC Extraction**

Extract MFCC features and save them in `.npy` or `.len` format.

```bash
python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
```

- **`${tsv_dir}`:** Manifest file directory.
- **`${split}`:** Data split (e.g., `train`, `valid`).
- **`${nshard}`:** Number of shards.
- **`${rank}`:** Shard index (0-based).
- **`${feat_dir}`:** Feature storage directory.

---

### **Step 3: K-means Clustering**

Run K-means clustering:

```bash
python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_clusters} --percent 0.1
```

- **`${feat_dir}`:** Feature directory.
- **`${split}`:** Dataset split.
- **`${nshard}`:** Number of shards.
- **`${km_path}`:** Path to save the trained model.
- **`${n_clusters}`:** Number of clusters (e.g., `100`, `500`, `1000`).
- **`--percent`:** Dataset fraction (e.g., `0.1` for 10%).

---

### **Step 4: K-means Application**

Assign cluster labels to the dataset:

```bash
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
```

Merge sharded label files:

```bash
for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km
```

The final label file will be:
```
${lab_dir}/${split}.km
```

---

### **Step 5: Create a Dummy Dictionary**

Create a dummy dictionary:

```bash
for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt
```

The dictionary will be saved as:
```
${lab_dir}/dict.km.txt
```

---

## **HuBERT Training**

### **Required Files:**
- `{train, valid}.tsv`: **Manifest files**
- `{train, valid}.km`: **K-means label files**
- `dict.km.txt`: **Dummy dictionary file**

### **Pre-Training**

```bash
python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/path/to/data \
  task.label_dir=/path/to/labels \
  task.labels='["km"]' \
  model.label_rate=100
```

### **Fine-Tuning**

```bash
python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/finetune \
  --config-name base_10h \
  task.data=/path/to/data \
  task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```

---

## **HuBERT Inference**

### **Required Files:**
- `.wav` files
- Pre-trained HuBERT model
- KM files

### **Run Inference**

Place `inference.py` and `util.py` in the `fairseq` folder.

Run the inference script:

```bash
cd fairseq
python inference.py \
  --in-wav-path <wav-file-path> \
  --out-unit-path <unit-output-path> \
  --mhubert-path <model-parameter-path> \
  --kmeans-path <k-means-file-path>
