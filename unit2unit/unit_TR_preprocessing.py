import os
import glob
import re
import random
import argparse


def main():
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

    # 언어별 디렉토리 설정
    lang_dirs = {lang: os.path.join(args.base_dir, lang) for lang in args.languages}

    # 파일 이름에서 언어 코드와 식별자를 추출하기 위한 패턴
    filename_pattern = re.compile(r'(.+)_([a-z]{2})_(\d+)\.unit$')

    # 각 언어의 파일을 저장할 딕셔너리 초기화
    lang_files = {lang: {} for lang in lang_dirs}

    # Step 1: 모든 파일 목록을 생성하고 기본 파일 이름 추출
    for lang, dir_path in lang_dirs.items():
        file_paths = glob.glob(os.path.join(dir_path, '*.unit'))
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            match = filename_pattern.match(filename)
            if match:
                base_name = f"{match.group(1)}_{match.group(3)}"  # 언어 코드를 제외한 기본 이름
                lang_files[lang][base_name] = file_path
            else:
                print(f"파일 이름 패턴이 일치하지 않습니다: {file_path}")

    # Step 2: 모든 언어에 공통으로 존재하는 파일 찾기
    common_basenames = set(lang_files[args.languages[0]].keys())
    for lang in args.languages[1:]:
        common_basenames &= set(lang_files[lang].keys())

    print(f"모든 언어에 공통된 파일 수: {len(common_basenames)}")

    # Step 3: 공통 파일의 내용을 병합하고 언어 토큰 추가
    # 출력 디렉토리 설정
    merged_files_dir = os.path.join(args.output_base_dir, 'multilingual')
    os.makedirs(merged_files_dir, exist_ok=True)

    # 소스와 타겟 문장을 저장할 리스트 초기화
    all_src_lines = []
    all_tgt_lines = []

    # 언어 목록 및 언어 쌍 생성
    languages = args.languages
    language_pairs = [(src, tgt) for src in languages for tgt in languages if src != tgt]

    for base_name in sorted(common_basenames):
        # 각 언어의 파일 내용을 읽어옴
        file_contents = {}
        for lang in lang_dirs:
            file_path = lang_files[lang][base_name]
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                file_contents[lang] = content

        # 각 언어 쌍에 대해 소스와 타겟 문장 생성
        for src_lang, tgt_lang in language_pairs:
            src_line = f'<{src_lang}> {file_contents[src_lang]}'
            tgt_line = f'<{tgt_lang}> {file_contents[tgt_lang]}'
            all_src_lines.append(src_line + '\n')
            all_tgt_lines.append(tgt_line + '\n')

    # Step 4: 데이터를 섞고 학습 및 검증 세트로 분할
    num_samples = len(all_src_lines)
    indices = list(range(num_samples))
    random.seed(args.random_seed)
    random.shuffle(indices)

    # 학습 및 검증 세트 분할
    train_size = int(args.train_ratio * num_samples)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    print(f"총 샘플 수: {num_samples}")
    print(f"학습 샘플 수: {len(train_indices)}")
    print(f"검증 샘플 수: {len(valid_indices)}")

    # Step 5: 병합된 내용을 파일로 저장
    # 학습 데이터 저장
    train_src_path = os.path.join(merged_files_dir, 'train.src')
    train_tgt_path = os.path.join(merged_files_dir, 'train.tgt')
    with open(train_src_path, 'w', encoding='utf-8') as src_f, \
            open(train_tgt_path, 'w', encoding='utf-8') as tgt_f:
        for idx in train_indices:
            src_f.write(all_src_lines[idx])
            tgt_f.write(all_tgt_lines[idx])

    # 검증 데이터 저장
    valid_src_path = os.path.join(merged_files_dir, 'valid.src')
    valid_tgt_path = os.path.join(merged_files_dir, 'valid.tgt')
    with open(valid_src_path, 'w', encoding='utf-8') as src_f, \
            open(valid_tgt_path, 'w', encoding='utf-8') as tgt_f:
        for idx in valid_indices:
            src_f.write(all_src_lines[idx])
            tgt_f.write(all_tgt_lines[idx])

    print(f"학습 및 검증 데이터가 {merged_files_dir}에 저장되었습니다.")
    print("데이터 준비가 완료되었습니다.")


if __name__ == '__main__':
    main()