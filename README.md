# Automatic Speech Recognition Using TensorFlow
Code is writen in Python 2.7, TensorFlow 1.0.
The high-level network structure is demonstrated in below figure. 

<img src="https://github.com/brianlan/automatic-speech-recognition/blob/master/ASR%20Network%20Structure.PNG" width="480">

## Dataset
The dataset used in this repo is TIMIT. The training set contains sentences, while the test set contains sentences. 

## WAV Format Conversion

## Feature Extraction
MFCC is used here to extract features out of raw sound wav.

## Bi-directional GRU

## CTC

## PER
A PER calculation wrapper of leven edit distance is implemented.
