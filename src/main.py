import os
import torch
import librosa
from torchvision import transforms
from dataloader import CustomImageDataset


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=False)

    dataloader = CustomImageDataset(img_dir='D:\Storage\Technical\Datasets\Songs').generate_spectrograms()
    # state_dict = model.state_dict()
    # conv1_weight = state_dict['conv1.weight']
    # state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    # model.load_state_dict(state_dict)

    y, sr = librosa.load('src/test.wav')
    hop_length = 512
    n_mels = 128

    # spec = spectrogram_image(y, sr, 'test.png', hop_length, n_mels)
    # input_tensor = preprocess(spec)
    # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        # input_batch = input_batch.to(device)
        model.to(device)

    # with torch.no_grad():
        # output = model(input_batch)['out'][0]
    # output_predictions = output.argmax(0)
