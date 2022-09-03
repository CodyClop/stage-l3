from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
from utils import kfold_cls, load_textual_model, init_parser, kfold_reg, path_to_base_folder

if __name__ == "__main__":
    parser = init_parser()
    parser.add_argument("--cls", action='store_true')
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    model = load_textual_model(args.model, device)
    data_directory = path_to_base_folder() + 'data'

    print("Processing dataset...")
    df = pd.read_csv(data_directory + '/flickr_position_train.csv', sep='|')
    train = []

    for i in range(len(df)):
        description = df['description'][i]
        index = df['black_index'][i] - 1
        if not args.cls:
            index = torch.FloatTensor([index])
        train.append((str(description), index))

    df = pd.read_csv(data_directory + '/flickr_position_test.csv', sep='|')
    test = []

    for i in range(len(df)):
        description = df['description'][i]
        index = df['black_index'][i] - 1
        if not args.cls:
            index = torch.FloatTensor([index])
        test.append((str(description), index))

    X, y = model.get_features(train)
    test_features, test_labels = model.get_features(test)

    if args.cls:
        test_set = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    else:
        test_set = TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Probing...")
    if args.cls:
        print("Accuracy : " + str(kfold_cls(X, y, test_loader, model.output_dim, 15, args.control, args.nfold, device, 50, 5)))
    else:
        print("RMSE : " + str(kfold_reg(X, y, test_loader, model.output_dim, args.control, args.nfold, device, 50, 5)))
