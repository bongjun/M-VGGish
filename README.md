# M-VGGish embedding

This repository contains example codes to extract audio features using M-VGGish model presented in the following papers

* Bongjun Kim and Bryan Pardo, “Improving Content-based Audio Retrieval by Vocal Imitation Feedback,” IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Brighton, UK, 2019. [[pdf]](https://www.bongjunkim.com/pages/files/papers/icassp19_Kim.pdf) 

```
@inproceedings{kim2019improving,
  title={Improving content-based audio retrieval by vocal imitation feedback},
  author={Kim, Bongjun and Pardo, Bryan},
  booktitle={ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4100--4104},
  year={2019},
  organization={IEEE}
}
```

* Some of the feature extraction codes (`vggish_utils/`) are from the repository of [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset). (`vggish_input_clipwise.py` was newly added to extract a clip-level feature vector.)

* It takes a mel-spectrogram of a recording (any length) and outputs a 8192-dimensional feature vector.

* You will need `python 3` with `PyTorch` and `librosa`

* Run the example code (It runs on CPU).
```shell
python run.py
```
