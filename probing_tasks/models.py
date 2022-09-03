import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPVisionModel
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import ViltProcessor, ViltModel
from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from PIL import Image

class ProbingModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class ClipVision:
    def __init__(self, device='cuda'):
        self.device = device
        self.output_dim = 1024
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def get_features(self, dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=16)):
                images = [Image.open(image).convert("RGB") for image in images]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                all_features.append(outputs.pooler_output)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

class ClipText:
    def __init__(self, device='cuda'):
        self.device = device
        self.output_dim = 768
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def get_features(self, dataset):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for texts, labels in tqdm(DataLoader(dataset, batch_size=16)):
                inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                all_features.append(outputs.pooler_output)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    def get_word_features(self, dataset):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for sentence, labels in tqdm(dataset):
                inputs = self.tokenizer(sentence, padding=True, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                # Skip sentences containing more than 1 token for a single word
                if len(outputs.last_hidden_state[0]) != len(labels) + 2:
                    continue
                for i in range(len(labels)):
                    all_features.append(outputs.last_hidden_state[0][1+i].unsqueeze(0))
                    all_labels.append(torch.IntTensor([labels[i]]))

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


class Vilt:
    def __init__(self, device='cuda', probed_part="image"):
        self.device = device
        self.probed_part = probed_part
        self.output_dim = 768
        self.model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.model.to(device)
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    def get_features(self, dataset):
        all_features = []
        all_labels = []
        image = Image.open('../data/black_image.png').convert("RGB")
        with torch.no_grad():
            for data, labels in tqdm(DataLoader(dataset, batch_size=16)):
                if self.probed_part == "image":
                    images = [Image.open(image).convert("RGB") for image in data]
                    inputs = self.processor(images, ["" for _ in range(len(labels))], return_tensors="pt")
                else:
                    inputs = self.processor([image for _ in range(len(labels))], [data[i] for i in range(len(labels))], return_tensors="pt", padding=True)
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                all_features.append(outputs.pooler_output)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    def get_word_features(self, dataset):
        all_features = []
        all_labels = []
        image = Image.open('../data/black_image.png').convert("RGB")
        with torch.no_grad():
            for sentence, labels in tqdm(dataset):
                inputs = self.processor(image, sentence, padding=True, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                # Skip sentences containing more than 1 token for a single word
                if len(outputs.last_hidden_state[0]) != len(labels) + 147:
                    continue
                for i in range(len(labels)):
                    all_features.append(outputs.last_hidden_state[0][1+i].unsqueeze(0))
                    all_labels.append(torch.IntTensor([labels[i]]))

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


class Vit:
    def __init__(self, device='cuda'):
        self.device = device
        self.output_dim = 768
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model.to(device)
        self.processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def get_features(self, dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=16)):
                images = [Image.open(image).convert("RGB") for image in images]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                all_features.append(outputs.pooler_output)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


class Bert:
    def __init__(self, device='cuda'):
        self.device = device
        self.output_dim = 768
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to(device)

    def process(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs.to(self.device)
        outputs = self.model(**inputs)
        inputs.to('cpu')
        return outputs.pooler_output.squeeze()

    def get_features(self, dataset):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for texts, labels in tqdm(DataLoader(dataset, batch_size=16)):
                inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                all_features.append(outputs.pooler_output)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    def get_word_features(self, dataset):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for sentence, labels in tqdm(dataset):
                inputs = self.tokenizer(sentence, padding=True, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs)
                inputs.to('cpu')
                # Skip sentences containing more than 1 token for a single word
                if len(outputs.last_hidden_state[0]) != len(labels) + 2:
                    continue
                for i in range(len(labels)):
                    all_features.append(outputs.last_hidden_state[0][1+i].unsqueeze(0))
                    all_labels.append(torch.IntTensor([labels[i]]))

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
