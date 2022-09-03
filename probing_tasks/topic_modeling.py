from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
from utils import kfold_cls, load_textual_model, init_parser, path_to_base_folder


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    model = load_textual_model(args.model, device)

    data_directory = path_to_base_folder() + 'data'

    categories = pd.read_csv(data_directory + '/news_dataset_train.csv', sep='|', usecols=['Category'])['Category']
    categories = list(dict.fromkeys(categories))
    category_id = {}
    for i, category in enumerate(categories):
        category_id[category] = i

    print("Processing dataset...")
    df = pd.read_csv(data_directory + '/news_dataset_train.csv', sep='|')
    train = []

    for i in range(len(df)):
        headline = df['Headline'][i]
        category = df['Category'][i]
        train.append((str(headline), category_id[category]))

    df = pd.read_csv(data_directory + '/news_dataset_test.csv', sep='|')
    test = []

    for i in range(len(df)):
        headline = df['Headline'][i]
        category = df['Category'][i]
        test.append((str(headline), category_id[category]))

    X, y = model.get_features(train)
    test_features, test_labels = model.get_features(test)

    test_set = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_set, batch_size=16)

    print("Probing...")
    print("Accuracy : " + str(kfold_cls(X, y, test_loader, model.output_dim, len(categories), args.control, args.nfold, device, 20, 2)))
