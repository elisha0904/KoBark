import os
import glob
import random
import numpy
import uuid

from bark import text_to_semantic
from bark.generation import load_model

from data import get_sentence_from_book

print('Loading semantics model')
load_model(use_gpu=True, use_small=False, force_reload=False, model_type='text')

def extract_parts_from_filename(filename):
    parts = filename.split('-')
    return parts[0], parts[1][:2], parts[3], parts[4][:3]

output = 'output'
if not os.path.isdir(output):
    os.mkdir(output)

for book_path in glob.glob(f'txt/*.txt'):
    style, emotion, intensity, reciter = extract_parts_from_filename(os.path.basename(book_path))
    sentence_index = 51  # 시작 문장 번호 설정
    while True:
        sentence = get_sentence_from_book(book_path, sentence_index - 51)  # 인덱스 조정
        if sentence is None:
            break  # 모든 문장을 처리한 경우 루프 종료
        sentence = sentence.strip()
        if sentence:
            print(f'Generating semantics for text: {sentence}')
            semantics = text_to_semantic(sentence, temp=round(random.uniform(0.6, 0.8), ndigits=2))
            file_name = f"{style}-{emotion}-{intensity}-{reciter}-{str(sentence_index).zfill(4)}.npy"
            file_path = os.path.join(output, file_name)
            numpy.save(file_path, semantics)
            sentence_index += 1