import os
from torch.utils.data import TensorDataset, DataLoader
import torch
from utils import kfold_cls, load_visual_model, init_parser, path_to_base_folder

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    model = load_visual_model(args.model, device)
    data_directory = path_to_base_folder() + 'data'

    print("Processing dataset...")
    train = []
    test = []
    for image_name in os.listdir(data_directory + "/CLEVR/images/with_red_cube/train/"):
        image = data_directory + "/CLEVR/images/with_red_cube/train/" + image_name
        train.append((image, 1))

    for image_name in os.listdir(data_directory + "/CLEVR/images/with_red_cube/test/"):
        image = data_directory + "/CLEVR/images/with_red_cube/test/" + image_name
        test.append((image, 1))

    for image_name in os.listdir(data_directory + "/CLEVR/images/without_red_cube/train/"):
        image = data_directory + "/CLEVR/images/without_red_cube/train/" + image_name
        train.append((image, 0))

    for image_name in os.listdir(data_directory + "/CLEVR/images/without_red_cube/test/"):
        image = data_directory + "/CLEVR/images/without_red_cube/test/" + image_name
        test.append((image, 0))

    X, y = model.get_features(train)
    test_features, test_labels = model.get_features(test)

    test_set = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Probing...")
    print("Accuracy : " + str(kfold_cls(X, y, test_loader, model.output_dim, 2, args.control, args.nfold, device)))
