import os
from einops import pack, unpack
import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import HubertModel
from customtokenizer import CustomTokenizer

def load_model(narr=True):
    device='cuda'

    print('Loading HuBERT...')
    hubert_model = HubertModel.from_pretrained("team-lucid/hubert-base-korean")
    hubert_model.to(device)

    print('Loading Quantizer...')
    if narr: 
        quant_model = CustomTokenizer.load_from_checkpoint("../KOR-HuBERT-Quantizer/Literature/new_model_epoch_37.pth", device)
    else:
        quant_model = CustomTokenizer.load_from_checkpoint("../KOR-HuBERT-Quantizer/Literature/new_model_epoch_8.pth", device)
        

    print('Loading Encodec...')
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model.to(device)

    print('Downloaded and loaded models!')

def create_voice(wav_file, hubert_model, quant_model, encodec_model):
    device='cuda'

    out_file = ''.join(wav_file.split('.')[:-1])

    wav, sr = torchaudio.load(wav_file)
    print(f'sampling rate : {sr}')

    wav_hubert = wav.to(device)

    if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
        wav_hubert = wav_hubert.mean(0, keepdim=True)

    resampler = torchaudio.transforms.Resample(
    orig_freq= sr,
    new_freq= 16000).to(device)

    if sr > 16000:
        wav_hubert = resampler(wav_hubert)

    print('Extracting semantics...')
    semantic_vectors = hubert_model.forward(wav_hubert).last_hidden_state

    embed, packed_shape = pack(semantic_vectors, '* d')
    semantic_vectors = torch.from_numpy(embed.cpu().detach().numpy()).to(device)

    print('Tokenizing semantics...')
    semantic_tokens = quant_model.get_token(semantic_vectors)

    print('Creating coarse and fine prompts...')
    wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)

    wav = wav.to(device)

    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    codes = codes.cpu()
    semantic_tokens = semantic_tokens.cpu()

    np.savez(out_file,
            semantic_prompt=semantic_tokens.squeeze(),
            fine_prompt=codes,
            coarse_prompt=codes[:2, :]
            )

    print('Done!')

    return out_file + '.npz'

    