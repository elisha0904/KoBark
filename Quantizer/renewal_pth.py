## pth 파일 내부의 디렉토리 오류 수정하기 위한 파일

import zipfile
import os
import shutil

original_pth = 'model_epoch_5.pth' # 여기서 원하는 epoch의 파일 이름으로 바꾸기!
temp_dir = 'temp_model'
new_pth = 'new_' + ''.join(original_pth.split('.')[:-1]) + '.pth'

# 원본 .pth 파일에서 모든 내용을 임시 디렉토리에 추출
with zipfile.ZipFile(original_pth, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

print('move info file now...')

# info 파일 이동 및 info 파일 있던 폴더 삭제
model_path = os.path.join(temp_dir, ''.join(original_pth.split('.')[:-1]))
info_path = os.path.join(model_path, '.info')
archive_path = os.path.join(temp_dir, 'archive')

if os.path.exists(info_path):
    shutil.move(info_path, archive_path)
    shutil.rmtree(os.path.join(model_path))

print('generate new pth file now...')

# 새로운 pth 파일 생성
with zipfile.ZipFile(new_pth, 'w') as zip_ref:
    for foldername, subfolders, filenames in os.walk(temp_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            zip_path = os.path.relpath(file_path, temp_dir)
            zip_ref.write(file_path, zip_path)

# 임시 디렉토리 삭제
shutil.rmtree(temp_dir)
print('Finish!')