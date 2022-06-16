import os
import torch
import shutil
from functools import reduce
from docarray import Document
from pydub import AudioSegment
from itertools import zip_longest
from torch.utils.data import DataLoader, Dataset


def batch(iterable, n=1):
    args = [iter(iterable)] * n
    return zip_longest(*args)


def pad_tensor(vec, pad, value=0, dim=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def pad_collate(batch, values=(0, 0), dim=0):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
        ws - a tensor of sequence lengths
    """

    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])
    sequence_lengths, xids = sequence_lengths.sort(descending=True)
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    target_lengths, yids = target_lengths.sort(descending=True)
    # find longest sequence
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    # pad according to max_len
    batch = [(pad_tensor(x, pad=src_max_len+100, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for (x, y) in batch]

    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.stack([x[1] for x in batch]).int()
    xs = xs[xids]
    ys = ys[yids]
    return xs, ys, sequence_lengths.int(), target_lengths.int()


def load_midi_to_tensor(fpath):
    print("TODO")
    return None


class AudioDataset(Dataset):
    def __init__(self, data_dir='dataset', reload_data=False):
        super(AudioDataset, self).__init__()
        self.audio_dir = data_dir
        self.lowest_dirs = list()

        if reload_data:
            for root, dirs, files in os.walk(self.audio_dir):
                files = os.listdir(root)
                chart_fname = None
                for file in files:
                    if file.endswith('.midi'):
                        chart_fname = file

                if not dirs and chart_fname:
                    self.lowest_dirs.append(root)
                    song_name = root.split('\\')[-1]

                    audio = [AudioSegment.from_ogg(os.path.join(root, fname)) for fname in files
                             if os.path.join(root, fname).endswith('.ogg')]

                    combined_audio = reduce(lambda a, b: a+b, audio)
                    combined_audio.export(os.path.join('dataset', song_name, song_name+'.ogg'), format='ogg')
                    shutil.copyfile(os.path.join(root, chart_fname), os.path.join('dataset', song_name, song_name+'.midi'))

        self.audio_dir = 'dataset'
        self.songs = os.listdir('dataset')

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        # Load from file with docarray
        song_name = self.songs[idx]
        song = Document(uri=os.path.join('dataset', song_name, song_name+'.ogg')).load_uri_to_audio_tensor()
        label = load_midi_to_tensor(os.path.join('dataset', song_name, song_name+'.midi'))

        return song, label


# class CustomImageDataset(Dataset):
#     def __init__(self, img_dir, transform=None, target_transform=None):
#         # self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#         self.preprocess = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
#         ])
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
#
#     def generate_spectrograms(self):
#         hop_length = 512
#         n_mels = 128
#         dir_path = self.img_dir
#
#         lowest_dirs = list()
#
#         for root, dirs, files in os.walk(dir_path):
#             if not dirs:
#                 lowest_dirs.append(root)
#                 song_name = root.split('\\')[-1]
#
#                 audio = [AudioSegment.from_ogg(os.path.join(root, fname)) for fname in os.listdir(root)
#                          if os.path.join(root, fname).endswith('.ogg')]
#
#                 combined_audio = reduce(lambda a, b: a+b, audio)
#                 combined_audio.export(os.path.join('dataset', song_name+'.ogg'), format='ogg')
#
#                 y, sr = librosa.effects.trim(combined_audio)
#                 spec = spectrogram_image(y, sr, hop_length, n_mels)
#                 spec_image = self.preprocess(spec).numpy()
#                 im = Image.fromarray(spec_image)
#                 im.save(os.path.join('dataset', song_name+'.png'))
#                 exit()
#
#
# def audiosegment_to_librosawav(audiosegment):
#     channel_sounds = audiosegment.split_to_mono()
#     samples = [s.get_array_of_samples() for s in channel_sounds]
#
#     fp_arr = np.array(samples).T.astype(np.float32)
#     fp_arr /= np.iinfo(samples[0].typecode).max
#     fp_arr = fp_arr.reshape(-1)
#
#     return fp_arr
#
#
# def scale_minmax(X, min=0.0, max=1.0):
#     X_std = (X - X.min()) / (X.max() - X.min())
#     X_scaled = X_std * (max - min) + min
#     return X_scaled
#
#
# def spectrogram_image(y, sr, hop_length, n_mels, out = None):
#     # use log-melspectrogram
#     mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
#     mels = np.log(mels + 1e-9)  # add small number to avoid log(0)
#
#     # min-max scale to fit inside 8-bit range
#     img = scale_minmax(mels, 0, 255).astype(np.uint8)
#     img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
#     img = 255-img   # invert. make black==more energy
#
#     # save as PNG
#     if out:
#         skimage.io.imsave(out, img)
#
#     return img
