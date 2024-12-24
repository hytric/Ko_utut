## Transformer 학습을 위한  Data preprocessing


1. **Hubert inference 돌리기**

이전에 학습된 Hubert 모델을 가지고 Multilingual dataset을 Unit으로 변경

`inference.py`
`util.py`

위 2개 파일 fairseq 폴더에 넣기

```bash
cd fairseq
python inference.py \
	--in-wav-path <wav파일경로> \
	--out-unit-path <unit출력저장경로> \
	--mhubert-path <모델파라미터경로> \
	--kmeans-path <k-means파일경로>
```

<br>

2. **데이터 파일 안에 내용 txt 파일로 저장**

파일이 해당 Unit 데이터를 쉽게 찾을 수 있도록 함

```bash
find /units/en/ -maxdepth 1 -name '*.unit' | sort > en_files.txt
find /units/es/ -maxdepth 1 -name '*.unit' | sort > es_files.txt
find /units/fr/ -maxdepth 1 -name '*.unit' | sort > fr_files.txt
```

<br>

3. **TR 학습 데이터 준비**

각 언어 쌍에 대해 학습 및 검증 데이터를 생성, Fairseq에서 요구하는 형식으로 준비

```bash
dataset/
└── units/
    ├── en/
    ├── ko/
    └── fr/
```

다음과 같이 파일 구조를 만들기

`unit_txt_gen.py`
`unit_TR_preprocessing.py`

각 언어의 파일 수가 동일한지 확인 → ***parallel corpus*** 맞추기

학습과 검증 데이터로 데이터를 분할. 일반적으로 90%를 학습용으로, 10%를 검증용으로 사용

```bash
# 명령행 인자 파서 설정
parser = argparse.ArgumentParser(description='다국어 데이터 준비 스크립트')
parser.add_argument('--base_dir', type=str, default='/home/jskim/audio/dataset/units',
                    help='입력 데이터의 기본 디렉토리 경로')
parser.add_argument('--output_base_dir', type=str, default='/home/jskim/audio/dataset/units',
                    help='출력 데이터가 저장될 기본 디렉토리 경로')
parser.add_argument('--languages', type=str, nargs='+', default=['en', 'es', 'fr'],
                    help='처리할 언어 목록 (예: en es fr)')
parser.add_argument('--train_ratio', type=float, default=0.9,
                    help='학습 데이터 비율 (0과 1 사이의 실수)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='랜덤 시드 값')
args = parser.parse_args()
```

다음 코드로 동일한 ***parallel corpus*** 를 찾을 수 있도록 해야함

```bash
filename_pattern = re.compile(r'(.+)_([a-z]{2})_(\d+)\.unit$')
``` 
파일에 맞는 패턴을 입력

위 파일 실행 했을 때 4개 txt 파일이 나와야함

train.src, train.tgt, valid.src, valid.tgt

<br>

4. fairseq Data Preprocessing

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

위 코드를 통해 실제 TR에 들어가는 data 구조 완성 

dict 파일도 같이 들어감

여기에서는 하나의 multilingual Hubert가 들어가기 때문에 1개의 dict만 필요함

<br>

## Trasformer Training code

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
 