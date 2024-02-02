# Korean custom TTS Audio Book with Bark 😎📗🎤
**소설 속 인물의 속성과 감정을 반영한 사용자 맞춤형 한국어 TTS 오디오북 with Bark**

Langchain을 활용하여 소설 속 인물의 속성과 감정을 파악한 뒤, 그에 맞는 음성으로 TTS를 생성하는 한국어 오디오북 프로젝트입니다.

### Environment

아래 환경에서 학습 및 구현되었으며, 이외의 환경에서는 테스트를 진행하지 않았습니다.

| python   | cuda | torch        | torchaudio |
| -------- | ---- | ------------ | ---------- |
| 3.10     | 11.3 | 1.12.1+cu113 | 0.12.1     |

**Fork, clone, or download!**
```
git clone https://github.com/elisha0904/koBark.git
```

이후 과정을 진행하기 전, 아래의 **bash command**를 통해 필요한 패키지를 다운로드 해주세요.
```
pip install -r requirement.txt
```

학습시킨 Quantizer 파일은 용량 문제로 업로드하지 않습니다.

## ① Data Preprocessing 

gitmylo의 코드를 참고하여, 우선 Quantizer를 학습시켜야 합니다.

Quantizer 학습을 위해 사전 학습된 [hubert-base-korean](https://huggingface.co/team-lucid/hubert-base-korean)을 불러와 사용하였습니다.

학습 데이터로는 [AIHub 감성 및 발화스타일 동시 고려 음성 합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=71349)를 사용하였습니다.

먼저, 데이터셋의 json 파일을 txt 파일로 포맷을 변경해주어야 합니다. (이를 위해 `json2txt.ipynb`를 사용해도 좋습니다.)

그런 다음 아래의 코드를 실행하여 txt 파일을 npy 포맷의 semantic 파일로 만들어줍니다.
```
python create_data_fixed.py
```

생성된 semantic 파일과 wav 파일들을 각각 `semantic.zip`, `wav.zip` 파일로 합쳐줍니다.

압축한 파일을 `Literature` 폴더 아래의 `semantic`, `wav` 폴더에 각각 넣어주어야 합니다.

이때, **`wav.zip` 파일의 이름도 `semantic.zip`으로 변경**해주어야 합니다.

폴더에 파일을 넣는 작업이 끝났다면, 아래의 코드를 차례대로 실행합니다.

```
python process.py --path Literature --mode prepare
python process.py --path Literature --mode prepare2
```

이제 Quantizer 학습을 위한 데이터 전처리가 모두 끝났습니다.

## ② Training Quantizer

아래의 코드를 실행하여 학습을 진행합니다.

```
process.py --path Literature --mode train
```

이제 학습된 Quantizer를 통해 원하는 음성으로 npz를 생성합니다.

이 과정은 `notebook.ipynb`에서 진행할 수 있습니다.

이때, 학습된 Quantizer의 경로를 주석이 달린 위치에 복사해서 넣어주고, 원하는 음성의 경로를 `wav_file`과 `out_file` 변수를 통해 지정해주어야 합니다.

notebook 파일을 끝까지 실행했다면, `out_file`의 경로에 npz 파일이 생성되어 있을 것입니다.

생성된 npz가 제대로 음성 스타일을 카피하였는지 테스트하기 위해, [Bark demo Colab](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)을 사용하셔도 좋습니다.

## ③ Let's AudioBook!

이제 소설을 읽을 차례입니다.

`app_modified.ipynb` 파일을 실행하여, 노트북 파일 내의 과정을 차례대로 따라하면 됩니다.

### Acknowledgement

아래의 깃허브 코드를 참고하였음을 밝힙니다.

- [suno-ai](https://github.com/suno-ai/bark)
- [gitmylo](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer)
