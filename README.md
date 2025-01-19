
  <a href='https://arxiv.org/abs/2211.12194'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; <a href='https://sadtalker.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vinthony/SadTalker) &nbsp; [![sd webui-colab](https://img.shields.io/badge/Automatic1111-Colab-green)](https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab/blob/main/video/stable/stable_diffusion_1_5_video_webui_colab.ipynb)  [![Replicate](https://replicate.com/cjwbw/sadtalker/badge)](https://replicate.com/cjwbw/sadtalker) 


![sadtalker](https://user-images.githubusercontent.com/4397546/222490039-b1f6156b-bf00-405b-9fda-0c9a9156f991.gif)



OUR Twitter :[VIDAI](https://x.com/VID__AI)

OUR WebSite : [VIDAI](http://vidai.world/)

## About US

> VIDAI: Intelligent Video-Audio Fusion Tool

### Versatile Applications

Perfect for film production, educational training, and social media content creation.

### Effortless Use

##### User-friendly interface allows anyone to generate high-quality results with ease.

### Multi-Language Support

Enables cross-language audio synchronization, making content globally accessible.

## To-Do

We're tracking new updates in [issue #280](https://github.com/OpenTalker/SadTalker/issues/280).

## Troubleshooting

If you have any problems, please read our [FAQs](docs/FAQ.md) before opening an issue.

## 1. Installation.

Community tutorials: [中文Windows教程 (Chinese Windows tutorial)](https://www.bilibili.com/video/BV1Dc411W7V6/) | [日本語コース (Japanese tutorial)](https://br-d.fanbox.cc/posts/5685086).

### Linux/Unix

1. Install [Anaconda](https://www.anaconda.com/), Python and `git`.

2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/JosephJosdqa/VIDAI

  cd VIDAI 

  conda create -n VIDAI python=3.8

  conda activate VIDAI

  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

  conda install ffmpeg

  pip install -r requirements.txt
  ```
### Windows

A video tutorial in chinese is available [here](https://www.bilibili.com/video/BV1Dc411W7V6/). You can also follow the following instructions:

1. Install [Python 3.8](https://www.python.org/downloads/windows/) and check "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win) manually or using [Scoop](https://scoop.sh/): `scoop install git`.
3. Install `ffmpeg`, following [this tutorial](https://www.wikihow.com/Install-FFmpeg-on-Windows) or using [scoop](https://scoop.sh/): `scoop install ffmpeg`.
4. Download the VIDAI repository by running `https://github.com/JosephJosdqa/VIDAI.git`.
5. Download the checkpoints and gfpgan models in the [downloads section](#2-download-models).
6. Run `start.bat` from Windows Explorer as normal, non-administrator, user, and a Gradio-powered WebUI demo will be started.

### macOS

A tutorial on installing VIDAI on macOS can be found [here](docs/install.md).

### Docker, WSL, etc

Please check out additional tutorials [here](docs/install.md).
