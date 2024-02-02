from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io import wavfile
import os
import wave
from matching import *
import re
# from pydub import AudioSegment

def bark_generate(text, speaker, number):
    audio_array = generate_audio(text, history_prompt=speaker)
    wavfile.write(f"../demo/voice/audio_book/{number}", SAMPLE_RATE, audio_array)

def full_bark(scenario, voices):
    generated = []

    for idx in range(len(scenario)):
        name, content, emotion = scenario[idx]
        
        if name == 'Narrator':
            speaker = voices[name]
        else:
            speaker = match_emotion(name, emotion, voices)
        sentences = re.split(r'[.!?]\s+', content)

        if len(sentences) > 0 and len(content) > 50:
            for snum in range(len(sentences)):
                number = '%02d_%02d' % (idx, snum) + '.wav'
                bark_generate(sentences[snum], speaker, number)
                generated.append([f"../demo/voice/audio_book/{number}", name, sentences[snum]])
        else:
            number = '%02d_00' % (idx) + '.wav'
            bark_generate(content, speaker, number)
            generated.append([f"../demo/voice/audio_book/{number}", name, content])

    return generated


def regenerate(generated):
    outputs = []
    for subset in generated:
        path, speaker, sentence, is_checked = subset
        if is_checked:
            number = path.split('/')[-1]
            bark_generate(sentence, speaker, number)
            outputs.append([path, speaker, sentence])

    return outputs

def convert_to_pcm(folder):
    files = os.listdir(folder)
    wav_files = [f for f in files if f.endswith('.wav')]

    wav_files.sort(key=lambda fname: [int(num) for num in re.findall(r'\d+', fname)])

    for fname in wav_files:
        file_path = os.path.join(folder, fname)
        audio = AudioSegment.from_file(file_path, format="wav")

        # PCM 형식으로 변환하여 저장
        output_file_path = os.path.join(folder, fname)
        audio.export(output_file_path, format="wav", codec="pcm_s16le")
        print(f'Converted {fname} to PCM format.')

def combine(folder):
    convert_to_pcm(folder)

    files = os.listdir(folder)
    print(f'files : {files}')
    wav_files = [f for f in files if f.endswith('.wav')]
    print(f'wav_files : {wav_files}')

    wav_files.sort(key=lambda fname: [int(num) for num in re.findall(r'\d+', fname)])
    print(f'wav_files_sort : {wav_files}')

    output = 'audio_book.wav'
    with wave.open(os.path.join(folder, output), 'wb') as outfile:
        with wave.open(os.path.join(folder, wav_files[0]), 'rb') as infile:
            outfile.setparams(infile.getparams())

        for fname in wav_files:
            with wave.open(os.path.join(folder, fname), 'rb') as infile:
                audio_data = infile.readframes(infile.getnframes())
                outfile.writeframes(audio_data)

    return os.path.join(folder, output)