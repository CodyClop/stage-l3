import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
from utils import kfold_reg, load_visual_model, init_parser, path_to_base_folder

w = 320
h = 240

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    model = load_visual_model(args.model, device)
    data_directory = path_to_base_folder() + 'data'

    print("Processing dataset...")
    train = []
    test = []
    for image_name in os.listdir(data_directory + "/CLEVR/images/with_intruder/train/"):
        image = data_directory + "/CLEVR/images/with_intruder/train/" + image_name

        with open(data_directory + '/CLEVR/scenes/with_intruder/train/' + image_name[:-3] + 'json', 'r') as f:
            scene = json.load(f)
            intruder_pos = [scene['intruder_pos'][0][0] / w, scene['intruder_pos'][0][1] / h,
                            scene['intruder_pos'][1][0] / w, scene['intruder_pos'][1][1] / h]

        train.append((image, torch.FloatTensor(intruder_pos)))

    for image_name in os.listdir(data_directory + "/CLEVR/images/with_intruder/test/"):
        image = data_directory + "/CLEVR/images/with_intruder/test/" + image_name

        with open(data_directory + '/CLEVR/scenes/with_intruder/test/' + image_name[:-3] + 'json', 'r') as f:
            scene = json.load(f)
            intruder_pos = [scene['intruder_pos'][0][0] / w, scene['intruder_pos'][0][1] / h,
                            scene['intruder_pos'][1][0] / w, scene['intruder_pos'][1][1] / h]

        test.append((image, torch.FloatTensor(intruder_pos)))

    X, y = model.get_features(train)
    test_features, test_labels = model.get_features(test)

    test_set = TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Probing...")
    print("Average Euclidian Distance : " + str(kfold_reg(X, y, test_loader, model.output_dim, args.control, args.nfold, device, bb_reg=True)))
