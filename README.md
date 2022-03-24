# Emotion Recognition ABAW3 Competition

This repo collects the code for our entry to the third competition on the Affwild2 database (ABAW3)

https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/

It allows training and evaluation of unimodal and multimodal models for time-continuous valence and arousal estimation on in the wild data.

## Installation

Requirements:

Runs on Python >=3.6

We recommend installing the following packages via conda:

liac-arff
librosa >= 0.8.1
matplotlib
numpy
pip
pandas >= 1.4.0
pytorch-lightning
ray
scikit-learn
scipy
seaborn
tabulate
tensorboard
tensorboardx
torch
torchvision
torchaudio
tqdm
facenet-pytorch
opencv-python >= 4.5.0
pytorch-model-summary
pynvml


You will also need to install this fork of the End2You toolkit: 

https://github.com/VincentKaras/end2you