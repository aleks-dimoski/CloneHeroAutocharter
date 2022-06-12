import os
import librosa
import skimage.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def generate_spectrograms(self):
        hop_length = 512
        n_mels = 128
        dir_path = self.img_dir

        lowest_dirs = list()

        for root, dirs, files in os.walk(dir_path):
            if not dirs:
                lowest_dirs.append(root)
                song_name = root.split('\\')[-1]

                audio_fpaths = [fpath for fpath in os.listdir(root) if fpath.endswith('.ogg')]


                y, sr = librosa.load(root)
                spec = spectrogram_image(y, sr, hop_length, n_mels)
                spec_image = self.preprocess(spec).numpy()
                im = Image.fromarray(spec_image)
                im.save(os.path.join('dataset', song_name+'.png'))


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, hop_length, n_mels, out = None):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255-img   # invert. make black==more energy

    # save as PNG
    if out:
        skimage.io.imsave(out, img)

    return img
