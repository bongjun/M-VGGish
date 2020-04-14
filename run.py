import librosa
import numpy as np
import torch
from torch.autograd import Variable

from vggish_utils import vggish_input_clipwise
from vggish_model_architecture import MVGGish

AUDIO_PATH = './test_audio.wav'
MODEL_PATH = './conv_weights/vggish_pretrained_convs.pth'

model = MVGGish()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

y, sr = librosa.load(AUDIO_PATH, sr=44100)

# the audio should be at least 1-second long.
if y.shape[0] < 1*sr:
	pad = np.zeros((1*sr-y.shape[0]))
	y = np.append(y, pad)

mel_input = vggish_input_clipwise.waveform_to_examples(y, sr)
mel_input = mel_input.astype('float32')

mel_input = mel_input.reshape(1, mel_input.shape[0], mel_input.shape[1])

representation = model(Variable(torch.from_numpy(mel_input)))

representation = representation.detach().numpy()

print(representation.shape)
