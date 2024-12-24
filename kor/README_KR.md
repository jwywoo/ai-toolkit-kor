# AI Toolkit by Ostris BUT in Korean

## 필독!!
이건 저의 연구 레포입니다. 여러 실험을 진행중이니 제 실수로 뭔갈 고장낼 수 있습니다.
뭔가 안된다면 저의 이전 Commit을 참고해주시기 바랍니다.
그리고 이 저장소가 많은 모델들을 학습시킬 수 있습니다. 그만큼 모든걸 최신에 맞출 수 없을 수 있어요.

## Support my work

<a href="https://glif.app" target="_blank">
<img alt="glif.app" src="https://raw.githubusercontent.com/ostris/ai-toolkit/main/assets/glif.svg?v=1" width="256" height="auto">
</a>

이 프로젝트는 [Glif](https://glif.app/)와 팀원들의 지원 없이는 불가능했을 것입니다.
저를 응원하고 싶으시면 Glif를 응원해 주세요. [Glif 가입](https://glif.app/),
[디스코드에서 함께하기](https://discord.com/invite/nuR9zZ2nsh), [트위터 팔로우](https://x.com/heyglif)로 우리와 함께 멋진 것들을 만들어 보세요!

## Installation

Requirements:
- python >3.10
- Nvidia GPU와 충분한 양의 RAM
- python venv
- git

Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
# .\venv\Scripts\activate on windows
# 토치먼저 설치하세요!
pip3 install torch
pip3 install -r requirements.txt
```

Windows:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## FLUX.1 Training

### Tutorial

빠르게 시작하려면, [@araminta_k](https://x.com/araminta_k)의 24GB VRAM [Finetuning Flux Dev on a 3090](https://www.youtube.com/watch?v=HzGW_Kyermg) 튜토리얼로 시작해보세요.

### Requirements
현재 FLUX.1 학습을 위해서는 **최소 24GB의 VRAM**이 탑재된 GPU가 필요합니다.
모니터를 제어하기 위해 GPU로 사용하는 경우, 설정 파일의 'model:' 아래에 'low_vram: true'로 플래그를 설정해야 할 것입니다.
이렇게 하면 모델이 CPU에서 양자화되고 모니터가 부착된 상태에서 학습할 수 있습니다.
(양자화는 계산과 메모리 비용을 낮추는것을 말합니다.)
사용자들은 WSL을 사용하여 Windows에서 작동하도록 했지만, 기본적으로 Windows에서 실행될 때 버그가 있다는 보고가 있습니다.
지금은 리눅스에서만 테스트해 보았습니다. 이것은 여전히 매우 실험적입니다. 그래서 24GB에 맞추기 위해서는 많은 양자화와 트릭이 필요합니다.

### FLUX.1-dev

FLUX.1-dev는 비상업적 라이선스를 가지고 있습니다. 즉, FLUX.1-dev를 기반으로 만든 모든 모델은 똑같은 라이센스가 적요이됩니다. 또한 게이트 모델이므로 사용하기 전에 HuggingFace에서 라이선스에 대한 동의를 진행해야 합니다. 라이선스에 대한 동의 절차는 다음과 같습니다.
FLUX.1-dev has a non-commercial license. Which means anything you train will inherit the
non-commercial license. It is also a gated model, so you need to accept the license on HF before using it.
Otherwise, this will fail. Here are the required steps to setup a license.
1. HuggingFace에 로그인한 다음 [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)으로 이동하세요.(처음들어가면 사용동의 관련된 설정을 볼 수 있습니다.)
2. `.env`파일을 만드시기 바랍니다.
3. [HuggingFace Read Key 가져오기](https://huggingface.co/settings/tokens/new?) 그리고 `.env`에 본인의 Read Key를 `HF_TOKEN=your_key_here`와 같이 추가해주시기 바랍니다.

### FLUX.1-schnell

죄송하지만 `FLUX.1-schnell`의 경우 안된다고 보시는게 좋습니다. 그래서 번역하지 않았습니다.

FLUX.1-schnell is Apache 2.0. Anything trained on it can be licensed however you want and it does not require a HF_TOKEN to train.
However, it does require a special adapter to train with it, [ostris/FLUX.1-schnell-training-adapter](https://huggingface.co/ostris/FLUX.1-schnell-training-adapter).
It is also highly experimental. For best overall quality, training on FLUX.1-dev is recommended.

To use it, You just need to add the assistant to the `model` section of your config file like so:

```yaml
      model:
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"
        is_flux: true
        quantize: true
```

You also need to adjust your sample steps since schnell does not require as many

```yaml
      sample:
        guidance_scale: 1  # schnell does not do guidance
        sample_steps: 4  # 1 - 4 works well
```

### Training

1. `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` schnell)에 있는 파일을 복사해서 `config` 폴더 붙여 넣으세요. 그 다음에 `whatever_you_want.yml`와 같이 이름을 다시 지어주세요.
2. yaml 파일에 코멘트를 따라 수정해주시면 됩니다.(`kor/storyboard-scene-generation-config.yaml`에 제가 작성한 한국어 Comment가 있으니 그걸 복사하여 수정하시는걸 추천드립니다.)
3. 실행하기 위해서 `python run.py config/whatever_you_want.yml`를 입력하셔서 실행하면됩니다.

시작할 때 config에 지정된 파일의 학습 폴더가 포함된 폴더가 생성됩니다. 모든 체크포인트와 체크포인트에서 생성된 이미지도 포함되며 CTRL+c를 사용하여 언제든지 학습을 중지할 수 있고 다시 시작하면 마지막 체크포인트에서부터 다시 시작합니다.

IMPORTANT 만약 체크포인트를 저장하는 중에 Ctrl+C를 누른다면 체크포인트에 임시저장된 내용들에 문제가 있을 수 있습니다.

### Need help?

실제로 코드에서 버그가 발견되지 않았다면 Bug리포트를 열어보지 마세요. 도움이 필요하다면 디스코드 채널에 들어와서 언제든지 도움을 요청하셔도 됩니다. [Join my Discord](https://discord.gg/VXmU2f5WEU)
하지만 가급적이면 저에게 개인메세지로 도움을 요청하시지는 말아주세요. 디스코드 채널에 올려주신다면 가능할 때 답변드리도록 하겠습니다.

## Gradio UI

위의 모든 과정을 따라했고 `ai-toolkit` 설치된 상태에서 만약 GUI툴을 활용하고 싶다면은:

```bash
cd ai-toolkit # ai-toolkit 디렐토리로 이동
huggingface-cli login #huggingface의 write 토큰으로 로그인합니다. .env에 넣어놔도 괜찮습니다.
python flux_train_ui.py
```

위에 같이 실행한다면 아래 사진과 GUI를 활용하여 이미지를 업로드하고, 이미지의 캡션을 생성하고 훈련한 다음 당신이 만든 LoRA를 Huggingface로 올리는거까지 가능합니다.

![image](/assets/lora_ease_ui.png)

## Training in RunPod(유료)
Example RunPod template: **runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04**
> You need a minimum of 24GB VRAM, pick a GPU by your preference.

#### Example config ($0.5/hr):
- 1x A40 (48 GB VRAM)
- 19 vCPU 100 GB RAM

#### Custom overrides (you need some storage to clone FLUX.1, store datasets, store trained models and samples):
- ~120 GB Disk
- ~120 GB Pod Volume
- Start Jupyter Notebook

### 1. Setup
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
### 2. Upload your dataset
- Create a new folder in the root, name it `dataset` or whatever you like.
- Drag and drop your .jpg, .jpeg, or .png images and .txt files inside the newly created dataset folder.

### 3. Login into Hugging Face with an Access Token
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run ```huggingface-cli login``` and paste your token.

### 4. Training
- Copy an example config file located at ```config/examples``` to the config folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file.
- Change ```folder_path: "/path/to/images/folder"``` to your dataset path like ```folder_path: "/workspace/ai-toolkit/your-dataset"```.
- Run the file: ```python run.py config/whatever_you_want.yml```.

### Screenshot from RunPod
<img width="1728" alt="RunPod Training Screenshot" src="https://github.com/user-attachments/assets/53a1b8ef-92fa-4481-81a7-bde45a14a7b5">

## Training in Modal(번역작업중)

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

데이터셋은 이미지와 이미지를 설명할 수 있는 텍스트 파일이 포함된 폴더여야 합니다.
현재 지원되는 형식은 jpg, jpeg, png입니다. 현재 Webp에 문제가 있습니다.
텍스트와 이미지 파일들의 이름은 확장자 빼고 동일해야합니다.
예를 들어 'image2.jpg'와 'image2.txt'입니다. 텍스트 파일에는 캡션만 포함되어야 합니다.
캡션 파일에 '[trigger]'를 추가할 수 있으며, config에 'trigger_word'로 지정한 단어가 있음 자동으로 교체됩니다.

이미지는 결코 업스케일링되지 않고, 다운스케일링되어 버킷에 배치됩니다. **이미지를 크롭하거나 크기를 조정할 필요는 없습니다**.
로더는 자동으로 크기를 조정하고 다양한 종횡비를 처리할 수 있습니다.

## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

---

## EVERYTHING BELOW THIS LINE IS OUTDATED: 이 밑으로는 신경ㄴㄴ

It may still work like that, but I have not tested it in a while.

---

### Batch Image Generation

A image generator that can take frompts from a config file or form a txt file and generate them to a 
folder. I mainly needed this for an SDXL test I am doing but added some polish to it so it can be used
for generat batch image generation.
It all runs off a config file, which you can find an example of in  `config/examples/generate.example.yaml`.
Mere info is in the comments in the example

---

### LoRA (lierla), LoCON (LyCORIS) extractor

It is based on the extractor in the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) tool, but adding some QOL features
and LoRA (lierla) support. It can do multiple types of extractions in one run.
It all runs off a config file, which you can find an example of in  `config/examples/extract.example.yml`.
Just copy that file, into the `config` folder, and rename it to `whatever_you_want.yml`.
Then you can edit the file to your liking. and call it like so:

```bash
python3 run.py config/whatever_you_want.yml
```

You can also put a full path to a config file, if you want to keep it somewhere else.

```bash
python3 run.py "/home/user/whatever_you_want.yml"
```

More notes on how it works are available in the example config file itself. LoRA and LoCON both support
extractions of 'fixed', 'threshold', 'ratio', 'quantile'. I'll update what these do and mean later.
Most people used fixed, which is traditional fixed dimension extraction.

`process` is an array of different processes to run. You can add a few and mix and match. One LoRA, one LyCON, etc.

---

### LoRA Rescale

Change `<lora:my_lora:4.6>` to `<lora:my_lora:1.0>` or whatever you want with the same effect. 
A tool for rescaling a LoRA's weights. Should would with LoCON as well, but I have not tested it.
It all runs off a config file, which you can find an example of in  `config/examples/mod_lora_scale.yml`.
Just copy that file, into the `config` folder, and rename it to `whatever_you_want.yml`.
Then you can edit the file to your liking. and call it like so:

```bash
python3 run.py config/whatever_you_want.yml
```

You can also put a full path to a config file, if you want to keep it somewhere else.

```bash
python3 run.py "/home/user/whatever_you_want.yml"
```

More notes on how it works are available in the example config file itself. This is useful when making 
all LoRAs, as the ideal weight is rarely 1.0, but now you can fix that. For sliders, they can have weird scales form -2 to 2
or even -15 to 15. This will allow you to dile it in so they all have your desired scale

---

### LoRA Slider Trainer

<a target="_blank" href="https://colab.research.google.com/github/ostris/ai-toolkit/blob/main/notebooks/SliderTraining.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This is how I train most of the recent sliders I have on Civitai, you can check them out in my [Civitai profile](https://civitai.com/user/Ostris/models).
It is based off the work by [p1atdev/LECO](https://github.com/p1atdev/LECO) and [rohitgandikota/erasing](https://github.com/rohitgandikota/erasing)
But has been heavily modified to create sliders rather than erasing concepts. I have a lot more plans on this, but it is
very functional as is. It is also very easy to use. Just copy the example config file in `config/examples/train_slider.example.yml`
to the `config` folder and rename it to `whatever_you_want.yml`. Then you can edit the file to your liking. and call it like so:

```bash
python3 run.py config/whatever_you_want.yml
```

There is a lot more information in that example file. You can even run the example as is without any modifications to see
how it works. It will create a slider that turns all animals into dogs(neg) or cats(pos). Just run it like so:

```bash
python3 run.py config/examples/train_slider.example.yml
```

And you will be able to see how it works without configuring anything. No datasets are required for this method.
I will post an better tutorial soon. 

---

## Extensions!!

You can now make and share custom extensions. That run within this framework and have all the inbuilt tools
available to them. I will probably use this as the primary development method going
forward so I dont keep adding and adding more and more features to this base repo. I will likely migrate a lot
of the existing functionality as well to make everything modular. There is an example extension in the `extensions`
folder that shows how to make a model merger extension. All of the code is heavily documented which is hopefully
enough to get you started. To make an extension, just copy that example and replace all the things you need to.


### Model Merger - Example Extension
It is located in the `extensions` folder. It is a fully finctional model merger that can merge as many models together
as you want. It is a good example of how to make an extension, but is also a pretty useful feature as well since most
mergers can only do one model at a time and this one will take as many as you want to feed it. There is an 
example config file in there, just copy that to your `config` folder and rename it to `whatever_you_want.yml`.
and use it like any other config file.

## WIP Tools


### VAE (Variational Auto Encoder) Trainer

This works, but is not ready for others to use and therefore does not have an example config. 
I am still working on it. I will update this when it is ready.
I am adding a lot of features for criteria that I have used in my image enlargement work. A Critic (discriminator),
content loss, style loss, and a few more. If you don't know, the VAE
for stable diffusion (yes even the MSE one, and SDXL), are horrible at smaller faces and it holds SD back. I will fix this.
I'll post more about this later with better examples later, but here is a quick test of a run through with various VAEs.
Just went in and out. It is much worse on smaller faces than shown here.

<img src="https://raw.githubusercontent.com/ostris/ai-toolkit/main/assets/VAE_test1.jpg" width="768" height="auto"> 

---

## TODO
- [X] Add proper regs on sliders
- [X] Add SDXL support (base model only for now)
- [ ] Add plain erasing
- [ ] Make Textual inversion network trainer (network that spits out TI embeddings)

---

## Change Log

#### 2023-08-05
 - Huge memory rework and slider rework. Slider training is better thant ever with no more
ram spikes. I also made it so all 4 parts of the slider algorythm run in one batch so they share gradient
accumulation. This makes it much faster and more stable. 
 - Updated the example config to be something more practical and more updated to current methods. It is now
a detail slide and shows how to train one without a subject. 512x512 slider training for 1.5 should work on 
6GB gpu now. Will test soon to verify. 


#### 2021-10-20
 - Windows support bug fixes
 - Extensions! Added functionality to make and share custom extensions for training, merging, whatever.
check out the example in the `extensions` folder. Read more about that above.
 - Model Merging, provided via the example extension.

#### 2023-08-03
Another big refactor to make SD more modular.

Made batch image generation script

#### 2023-08-01
Major changes and update. New LoRA rescale tool, look above for details. Added better metadata so
Automatic1111 knows what the base model is. Added some experiments and a ton of updates. This thing is still unstable
at the moment, so hopefully there are not breaking changes. 

Unfortunately, I am too lazy to write a proper changelog with all the changes.

I added SDXL training to sliders... but.. it does not work properly. 
The slider training relies on a model's ability to understand that an unconditional (negative prompt)
means you do not want that concept in the output. SDXL does not understand this for whatever reason, 
which makes separating out
concepts within the model hard. I am sure the community will find a way to fix this 
over time, but for now, it is not 
going to work properly. And if any of you are thinking "Could we maybe fix it by adding 1 or 2 more text
encoders to the model as well as a few more entirely separate diffusion networks?" No. God no. It just needs a little
training without every experimental new paper added to it. The KISS principal. 


#### 2023-07-30
Added "anchors" to the slider trainer. This allows you to set a prompt that will be used as a 
regularizer. You can set the network multiplier to force spread consistency at high weights

