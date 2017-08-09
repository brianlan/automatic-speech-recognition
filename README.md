# Automatic Speech Recognition Using TensorFlow
Code is writen in Python 2.7, TensorFlow 1.0.
The high-level network structure is demonstrated in below figure. 

<img src="https://github.com/brianlan/automatic-speech-recognition/blob/master/ASR%20Network%20Structure.PNG" width="480">

## Dataset
The dataset used in this repo is TIMIT. The training set contains 3699 utterances, while the test set contains 1347 utterances ('sa' files are removed from the original dataset to prevent bias to this system). 

## WAV Format Conversion
Original wav files are actually NIST format. So conversion must be made beforehand using script [nist2wav.sh](https://github.com/brianlan/automatic-speech-recognition/blob/master/src/nist2wav.sh). But please ensure you have [libsndfile](http://www.mega-nerd.com/libsndfile/) installed first in your machine. 

## Feature Extraction
MFCC is used here to extract features out of raw sound wav. I'm using code [here](https://github.com/zzw922cn/Automatic_Speech_Recognition/blob/master/feature/core/calcmfcc.py) to calculate the features.

## Model
4-layer Bi-directional GRU is used as the acoustic model, and CTC is used to calculate the loss and backpropagate the gradient to the previous network layers. Dropout and Gradient Clipping are used to prevent overfitting and gradient explosion.

## PER
A PER calculation wrapper of leven edit distance is implemented ([code](https://github.com/brianlan/automatic-speech-recognition/blob/master/src/utils/PER_merge_phn.py)), so based on this distance, we can calculate PER arbitrarily without using TensorFlow's sub-graph. To be specific in this case, as suggested in [Speaker-independent phone recognition using hidden Markov models](http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci), we merge original 61 phonemes into 39 to gain more robust predictions.

Below figure is generated using TensorBoard during training phase.
<img src="https://github.com/brianlan/automatic-speech-recognition/blob/master/tensorboard_train_error.png" width="960">
