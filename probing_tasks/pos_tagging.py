from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
from utils import kfold_cls, load_textual_model, init_parser, path_to_base_folder

if __name__ == "__main__":
    parser = init_parser()
    parser.add_argument("--coarse", action='store_true')
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    model = load_textual_model(args.model, device)
    data_directory = path_to_base_folder() + 'data'

    labels = []
    df = pd.read_csv(data_directory + '/flickr_postag_train.csv', sep='|')
    key = 'coarse_grained_pos' if args.coarse else 'fine_grained_pos'
    for tags in df[key]:
        for tag in tags.split(' '):
            labels.append(tag)
    df = pd.read_csv(data_directory + '/flickr_postag_test.csv', sep='|')
    for tags in df[key]:
        for tag in tags.split(' '):
            labels.append(tag)
    labels = list(dict.fromkeys(labels))
    label2id = {}
    for i, label in enumerate(labels):
        label2id[label] = i

    print("Processing dataset...")
    df = pd.read_csv(data_directory + '/flickr_postag_train.csv', sep='|')
    train = []
    for i in range(len(df)):
        description = df['description'][i]
        labels = [label2id[tag] for tag in df[key][i].split(' ')]
        train.append((description, labels))

    df = pd.read_csv(data_directory + '/flickr_postag_test.csv', sep='|')
    test = []
    for i in range(len(df)):
        description = df['description'][i]
        labels = [label2id[tag] for tag in df[key][i].split(' ')]
        test.append((description, labels))

    X, y = model.get_word_features(train)
    test_features, test_labels = model.get_word_features(test)

    test_set = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Number of words used : " + str(len(y)) + " (train), " + str(len(test_labels)) + " (test)")
    print("Probing...")
    print("Accuracy : " + str(kfold_cls(X, y, test_loader, model.output_dim, len(label2id.keys()), args.control, args.nfold, device, 50, 5)))
