## 원본 wav 파일을 16khz로 downsampling

import torchaudio
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 원본 데이터와 다운샘플링 데이터 폴더 경로 정의
source_folder = '원천데이터'
downsampled_folder = 'downsampling data'

# 다운샘플링할 샘플레이트 정의
new_sample_rate = 16000

# 원본 데이터 폴더 내의 모든 하위 폴더 순회
for subdir, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.wav'):
            # 원본 파일의 전체 경로
            original_file_path = os.path.join(subdir, file)
            # 대상 폴더 경로 생성
            target_subdir = subdir.replace(source_folder, downsampled_folder)
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)
            # 대상 파일의 전체 경로
            downsampled_file_path = os.path.join(target_subdir, file)
            
            # 오디오 파일 로드
            waveform, orig_sample_rate = torchaudio.load(original_file_path)
            waveform = waveform.to(device)
            
            # resampler 생성
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate).to(device)
            
            # 다운샘플링 진행
            downsampled_waveform = resampler(waveform)
            
            # 다운샘플링된 파일 저장
            torchaudio.save(downsampled_file_path, downsampled_waveform.cpu(), new_sample_rate)

print("다운샘플링 완료")

# check sampling rate
sr, wav = wavfile.read('downsampling data/VS_독백체_071/M-A1-A-071-0051.wav')
print(f"The sample rate is: {sr} Hz")