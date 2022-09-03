import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
from utils import kfold_cls, load_visual_model, init_parser, path_to_base_folder

width = 320
height = 240

def get_scene_part_index(pos, n_row, n_col):
    pos = ((pos[0][0] + pos[1][0]) / 2, (pos[0][1] + pos[1][1]) / 2)

    index = (-1, -1)
    for i in range(1, n_col + 1):
        if pos[0] <= i * (width // n_col):
            index = (i - 1, -1)
            break
    for i in range(1, n_row + 1):
        if pos[1] <= i * (height // n_row):
            index = (index[0], i - 1)
            break

    return index[0] * n_row + index[1]


if __name__ == "__main__":
    parser = init_parser()
    parser.add_argument("--rows",
                        type=int, required=True,
                        help="number of rows")
    parser.add_argument("--cols",
                        type=int, required=True,
                        help="number of columns")
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    model = load_visual_model(args.model, device)
    data_directory = path_to_base_folder() + 'data'

    print("Processing dataset...")
    train = []
    test = []
    for image_name in os.listdir(data_directory + "/CLEVR/images/with_red_cube/train/"):
        image = data_directory + "/CLEVR/images/with_red_cube/train/" + image_name

        with open(data_directory + '/CLEVR/scenes/with_red_cube/train/' + image_name[:-3] + 'json', 'r') as f:
            scene = json.load(f)
            scene_part_index = get_scene_part_index(scene['red_cube_pos'], args.rows, args.cols)

        train.append((image, scene_part_index))

    for image_name in os.listdir(data_directory + "/CLEVR/images/with_red_cube/test/"):
        image = data_directory + "/CLEVR/images/with_red_cube/test/" + image_name

        with open(data_directory + '/CLEVR/scenes/with_red_cube/test/' + image_name[:-3] + 'json', 'r') as f:
            scene = json.load(f)
            scene_part_index = get_scene_part_index(scene['red_cube_pos'], args.rows, args.cols)

        test.append((image, scene_part_index))

    X, y = model.get_features(train)
    test_features, test_labels = model.get_features(test)

    test_set = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Probing...")
    print("Accuracy : " + str(kfold_cls(X, y, test_loader, model.output_dim, args.rows * args.cols, args.control, args.nfold, device)))
