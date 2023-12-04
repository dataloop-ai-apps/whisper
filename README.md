# Whisper Model Adapter

## Introduction

This repo is a model integration between OpenAI/Whisper and Dataloop.

Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains **without** the need for fine-tuning.

Whisper was proposed in the paper [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) by Alec Radford et al. from OpenAI. The original code repository can be found [here](https://github.com/openai/whisper).

Whisper `large-v3` has the same architecture as the previous large models except the following minor differences:

1. The input uses 128 Mel frequency bins instead of 80
2. A new language token for Cantonese

The Whisper `large-v3` model is trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper `large-v2`. The model was trained for 2.0 epochs over this mixture dataset.

## Requirements

* dtlpy
* torch
* transformers
* An account in the Dataloop platform

## Installation

Installation process of openai/whisper is seemleasly simple, you should fill in your project name in the model apdapter and run the code to deploy the model to your project.

Next you shuold visit the Dataloop platform, a new model will apper in the Models menu, press on the 3 dots and select deploy.

By default the model is configured to minimum replicas of '0', therefor it will take a few minutes for the model to warm-up, in case you need it in real-time you can change it to 1 (or more depending on your load).
