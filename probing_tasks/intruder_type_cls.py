import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
from collections import Counter
from utils import kfold_cls, load_visual_model, init_parser, path_to_base_folder

color_id = {
    "red": 0,
    "green": 1,
    "blue": 2
}

shape_id = {
    "cube": 0,
    "sphere": 1,
    "cylinder": 2
}

def get_object_type_id(object):
    return shape_id[object["shape"]] * 3 + color_id[object["color"]]

def get_intruder_type(objects):
    shapes_counter = Counter([o["shape"] for o in objects])
    colors_counter = Counter([o["color"] for o in objects])

    # common objects' property is shape
    if len(shapes_counter.keys()) == 2:
        for shape in shapes_counter.keys():
            if shapes_counter[shape] == 1:
                intruder_shape = shape

        for o in objects:
            if o["shape"] == intruder_shape:
                return get_object_type_id(o)
    # common objects' property is color
    else:
        for color in colors_counter.keys():
            if colors_counter[color] == 1:
                intruder_color = color

        for o in objects:
            if o["color"] == intruder_color:
                return get_object_type_id(o)


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
            intruder_type_id = get_intruder_type(scene['objects'])

        train.append((image, intruder_type_id))

    for image_name in os.listdir(data_directory + "/CLEVR/images/with_intruder/test/"):
        image = data_directory + "/CLEVR/images/with_intruder/test/" + image_name

        with open(data_directory + '/CLEVR/scenes/with_intruder/test/' + image_name[:-3] + 'json', 'r') as f:
            scene = json.load(f)
            intruder_type_id = get_intruder_type(scene['objects'])

        test.append((image, intruder_type_id))

    X, y = model.get_features(train)
    test_features, test_labels = model.get_features(test)

    test_set = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Probing...")
    print("Accuracy : " + str(kfold_cls(X, y, test_loader, model.output_dim, 9, args.control, args.nfold, device)))
