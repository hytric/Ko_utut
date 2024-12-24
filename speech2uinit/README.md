
# Hubert preprocessing

## KoEn Hubert Pretrained model
[[Hubert model link](https://drive.google.com/file/d/1w9fNZ1-Np1RurPKUtjYXJRtTBpVvcWrp/view?usp=sharing)] , 
[[KM file link](https://drive.google.com/file/d/1huzDxhoMlRFiZOxmC8lTkUp-jB3CpM1r/view?usp=sharing)]


## Hubert Spec

Parameters
1st Iter: K=100, 250K Steps 2nd, 3rd Iter: K=500, 400K Steps Adam Optimizer
Mask Prob: 0.5 Dropout: 0.3
Learning Rate: 0.001 Warmup: 10000

1. HuBERT를 통해 Waveform에서 MFCC 음성 Feature 추출
2. K-means를 이용하여 Feature를 Clustering -> Quantization
3. 데이터의 각 Frame을 특정 Unit에 배정 -> Pseudo-labeling


## Preprocess


wave2vec manifest → mmfc → sample kmean

출처  
https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md
https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/README.md


<br>

1. **manifest 생성**

    논문에서 10~30초 길이의 오디오 파일을 사용
    
    Python의 `soundfile` 라이브러리 설치 (`pip install soundfile`)
    
    ```bash
    python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
    ```
    
    - **`/path/to/waves`**: 오디오 파일이 있는 디렉토리.
    - **`-dest /manifest/path`**: 생성된 manifest 파일을 저장할 경로.
    - **`-ext $ext`**: 오디오 파일의 확장자 (예: wav, flac 등).
    - **`-valid-percent $valid`**: 학습 데이터에서 검증 데이터로 사용할 비율 (예: 0.01 = 1%).

<br>

2. **mmfc 생성**

    `.npy` 또는 `.len` 형식으로 저장
    
    ```bash
    python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
    ```
    
    - **`${tsv_dir}`**: 데이터셋 manifest 파일의 디렉토리.
    - **`${split}`**: 데이터를 분리한 종류(e.g., train, valid).
    - **`${nshard}`**: 전체 데이터를 몇 개의 샤드(shard)로 나눌지 설정.
    - **`${rank}`**: 처리할 샤드 번호 (0부터 시작).


<br>


3. **K-means clustering**
    
    feature를 데이터를 기반으로 K-means 클러스터링을 실행
    
    ```bash
    python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_clusters} --percent 0.1
    ```
    
    ### **파라미터 설명**
    
    - **`${feat_dir}`**: 추출된 특징(feature)이 저장된 디렉토리.
    - **`${split}`**: 데이터 분할 이름 (e.g., train, valid).
    - **`${nshard}`**: 데이터셋을 나누는 샤드 개수.
    - **`${km_path}`**: 학습된 K-means 모델이 저장될 경로.
    - **`${n_clusters}`**: K-means 클러스터 개수 (e.g., 100, 500, 1000).
    - **`-percent`**:
        - 사용할 데이터의 비율. 10% 데이터를 사용할 경우 `0.1`.
        - `1`로 설정하면 전체 데이터를 사용.
    

<br>

4. **K-means application**
    
    학습된 K-means 모델을 사용해 특징 데이터에 클러스터 레이블 할당
    
    ```bash
    python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
    ```
    
    - **`${feat_dir}`**: 특징(feature)이 저장된 디렉토리.
    - **`${split}`**: 데이터 분할 이름 (e.g., train, valid).
    - **`${km_path}`**: 학습된 K-means 모델 경로.
    - **`${nshard}`**: 샤드 개수.
    - **`${rank}`**: 처리할 샤드의 순번 (0부터 시작).
    - **`${lab_dir}`**: 레이블이 저장될 디렉토리
    
    `${lab_dir}/${split}_${rank}_${shard}.km` 형식으로 저장
    
    샤드별로 저장된 레이블 파일을 하나로 병합.
    
    ```bash
    for rank in $(seq 0 $((nshard - 1))); do
      cat $lab_dir/${split}_${rank}_${nshard}.km
    done > $lab_dir/${split}.km
    ```
    
     `${lab_dir}/${split}.km`
    

<br>

5. **Create a dummy dict**
    
    레이블과 가중치를 HuBERT 훈련에 사용할 수 있도록 사전 형식으로 저장
    
    ```bash
    for x in $(seq 0 $((n_clusters - 1))); do
      echo "$x 1"
    done >> $lab_dir/dict.km.txt
    ```
    
    더미 사전 파일 `${lab_dir}/dict.km.txt`
     
<br>


## Hubert training

학습을 위해 필요한 데이터 목록

 `{train, valid}.tsv` : **manifest** 파일

`{train, valid}.km` : **K-means application** 파일

`dict.km.txt`: **dummy dict** 파일

hubert pre-train

```bash
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/path/to/data task.label_dir=/path/to/labels task.labels='["km"]' model.label_rate=100
```

hubert fine-tuning

```bash
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/finetune \
  --config-name base_10h \
  task.data=/path/to/data task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```


<br>

## Hubert inference


필요한 데이터 : .wav 파일(언어별로 파일 분류) , 사전에 학습된 hubert 모델 , KM 파일


1. **Hubert inference 돌리기**

이전에 학습된 Hubert 모델을 가지고 Multilingual dataset을 Unit으로 변경

`inference.py` , `util.py` 

위 2개 파일 fairseq 폴더에 넣기

```bash
cd fairseq
python inference.py \
	--in-wav-path <wav파일경로> \
	--out-unit-path <unit출력저장경로> \
	--mhubert-path <모델파라미터경로> \
	--kmeans-path <k-means파일경로>
```