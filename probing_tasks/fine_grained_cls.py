import os
from torch.utils.data import TensorDataset, DataLoader
import torch
from utils import kfold_cls, load_visual_model, init_parser
import scipy
from torchvision.datasets import Flowers102

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    model = load_visual_model(args.model, device)

    print("Processing dataset...")
    root = os.path.expanduser("~/.cache")
    train = Flowers102(root, download=True, split="train")
    test = Flowers102(root, download=True, split="test")

    labels = scipy.io.loadmat(root + "/flowers-102/imagelabels.mat")['labels'][0]
    trnids = scipy.io.loadmat(root + "/flowers-102/setid.mat")['trnid'][0]
    tstids = scipy.io.loadmat(root + "/flowers-102/setid.mat")['tstid'][0]

    train = []
    test = []

    for image_name in os.listdir(root + "/flowers-102/jpg"):
        image = root + "/flowers-102/jpg/" + image_name
        id = int(image_name[-9:-4])

        if id in trnids:
            train.append((image, labels[id - 1] - 1))
        elif id in tstids:
            test.append((image, labels[id - 1] - 1))

    X, y = model.get_features(train)
    test_features, test_labels = model.get_features(test)

    test_set = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Probing...")
    print("Accuracy : " + str(kfold_cls(X, y, test_loader, model.output_dim, 102, args.control, args.nfold, device)))
