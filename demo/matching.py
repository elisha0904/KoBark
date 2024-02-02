import os
import glob
import zipfile
import random

def get_voices(style):
    # 속성과 맞는 폴더 내의 모든 zip 파일 호출
    path = f'../demo/voice/character/{style[0]}_{style[1]}_{style[2]}'

    # 후보 npz 경로 리스트 생성
    # candidates = glob.glob(os.path.join(path, '*.zip'))

    return path

def match_character(characters):
    voices = {}

    # for name in names:
    #     candidates = get_voice(characters[name])
    #     idx = random.choice([i for i in range(len(candidates))])
    #     voice = candidates.pop(idx)
    #     voices[name] = voice

    for name in [*characters]:
        voices[name] = get_voices(characters[name])

    return voices

def match_emotion(name, emotion, voices):
    # 캐릭터 zip 안에서 해당하는 감정 npz 호출
    # zip_path = voices[name]

    # with zipfile.Zipfile(zip_path, 'r') as czip:
    #     for npz in czip.namelist():
    #         if emotion in npz:
    #             with czip.open(npz) as f:
    #                 voice = f.read()

    path = voices[name]
    print(f'{name} : {path}')
    file_list = glob.glob(f'{path}/*.npz')
    print(f'file_list : {file_list} ')
    for npz_file in file_list:
        if emotion in npz_file:
            print(f'{npz_file}')
            return npz_file

def preset_narr(gender, age, random):
    if random:
        path = None
    else:
        path = f'../demo/voice/narrator/{gender}_{age}.npz'
 
    return path
