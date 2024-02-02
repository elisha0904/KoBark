import os
import shutil
import zipfile

import numpy
import torchaudio

# from hubert.pre_kmeans_hubert import CustomHubert
from transformers import HubertModel

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare(path):
    """
    Put all the training data in one folder
    :param path: The path to the training data, with 2 subdirectories with zips, "semantic" and "wav", with equal pairs in both directories
    """
    path = os.path.abspath(path)
    raw_data_paths = {
        'semantic': os.path.join(path, 'semantic'),
        'wav': os.path.join(path, 'wav')
    }
    prepared_path = os.path.join(path, 'prepared')

    if not os.path.isdir(prepared_path):
        os.mkdir(prepared_path)

    # offset 초기값 원래 0인데 우리는 zip 6개 돌려야 해서 초기값 계속 바꿔줬음
    # 출력되는 'now offset' + 1 더한 걸로 offset 변경해주면 됨
    offset = 27425

    for zip_file in os.listdir(raw_data_paths['semantic']):
        print(f'Extracting {os.path.basename(zip_file)}')
        offset = extract_files({
            'semantic': os.path.join(raw_data_paths['semantic'], zip_file),
            'wav': os.path.join(raw_data_paths['wav'], zip_file)
        }, prepared_path, offset)


def extract_files(zip_files: dict[str, str], out: str, start_offset: int = 0) -> int:
    new_offset = start_offset
    with zipfile.ZipFile(zip_files['semantic'], 'r') as semantic_zip:
        with zipfile.ZipFile(zip_files['wav'], 'r') as wav_zip:
            for file in semantic_zip.infolist():
                for file2 in wav_zip.infolist():
                    if ''.join(file.filename.split('.')[:-1]) == ''.join(file2.filename.split('.')[:-1]):
                        semantic_zip.extract(file, out)
                        shutil.move(os.path.join(out, file.filename), os.path.join(out, f'{new_offset}_semantic.npy'))
                        wav_zip.extract(file2, out)
                        shutil.move(os.path.join(out, file2.filename), os.path.join(out, f'{new_offset}_wav.wav'))
                        new_offset += 1
                        print(f'now offset : {new_offset - 1}')
            wav_zip.close()
        semantic_zip.close()

    return new_offset


def prepare2(path, model):
    prepared = os.path.join(path, 'prepared')
    ready = os.path.join(path, 'ready')
    # hubert_model = CustomHubert(checkpoint_path=model, device=device)
    hubert_model = HubertModel.from_pretrained("team-lucid/hubert-base-korean")
    hubert_model.to(device)

    if not os.path.isdir(ready):
        os.mkdir(ready)

    wav_string = '_wav.wav'
    sem_string = '_semantic.npy'

    for input_file in os.listdir(prepared):
        input_path = os.path.join(prepared, input_file)
        if input_file.endswith(wav_string):
            file_num = int(input_file[:-len(wav_string)])
            fname = f'{file_num}_semantic_features.npy'
            print('Processing', input_file)
            if os.path.isfile(fname):
                continue
            wav, sr = torchaudio.load(input_path)
            wav = wav.to(device)

            if wav.shape[0] == 2:  # Stereo to mono if needed
                wav = wav.mean(0, keepdim=True)

            ## output = hubert_model.forward(wav, input_sample_hz=sr) 
            output = hubert_model.forward(input_values = wav).last_hidden_state
            # out_array = output.cpu().detach().numpy() ## numpy() -> detach().numpy()
            out_array = torch.from_numpy(output.cpu().detach().numpy())

            numpy.save(os.path.join(ready, fname), out_array)
        elif input_file.endswith(sem_string):
            fname = os.path.join(ready, input_file)
            if os.path.isfile(fname):
                continue
            shutil.copy(input_path, fname)
    print('All set! We\'re ready to train!')
