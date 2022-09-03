from sklearn.model_selection import KFold
from models import *
from torch.utils.data import TensorDataset
import random
import argparse
import math
import os

def path_to_base_folder():
    cwd = os.getcwd()
    if cwd[-13:] == 'probing_tasks':
        return '../'
    else:
        return ''

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str, required=True,
                        help="probed model: clip, vilt, bert or vit")
    parser.add_argument("--nfold",
                        type=int, required=False, default=10,
                        help="Number of folds for the KFold (default 10)")
    parser.add_argument("--control", action='store_true')
    parser.add_argument("--cpu", action='store_true')
    return parser

def load_visual_model(model_name, device):
    model = None
    if model_name.lower() == "clip":
        model = ClipVision(device)
    elif model_name.lower() == "vilt":
        model = Vilt(device)
    elif model_name.lower() == "vit":
        model = Vit(device)
    elif model_name.lower() == "bert":
        print("BERT cannot be used for visual tasks")
    else:
        print("Unrecognized model name : " + model_name)
    return model

def load_textual_model(model_name, device):
    model = None
    if model_name.lower() == "clip":
        model = ClipText(device)
    elif model_name.lower() == "vilt":
        model = Vilt(device, "text")
    elif model_name.lower() == "bert":
        model = Bert(device)
    elif model_name.lower() == "vit":
        print("ViT cannot be used for textual tasks")
    else:
        print("Unrecognized model name : " + model_name)
    return model

def perf_cls(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = evaluation = num = 0
    for x, y in loader:
        with torch.no_grad():
            y = y.to(device)
            y_scores = model(x.to(device))
            loss = criterion(y_scores, y)
            y_pred = torch.max(y_scores, 1)[1]
            evaluation += torch.sum(y_pred.data == y)
            total_loss += loss.item()
            num += len(y)
    return total_loss / num, evaluation.item() / num

def kfold_cls(X, y, test_loader, input_dim, n_classes, control, n_fold, device, n_epochs=200, max_unimproving=20):
    kf = KFold(n_splits=n_fold, shuffle=True)
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    results = {}
    for lr in learning_rates:
        results[lr] = torch.FloatTensor([0, 0])

    for train_index, test_index in tqdm(kf.split(X), total=n_fold):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if control:
            random.shuffle(y_train)

        train_set = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_set = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16)

        for lr in learning_rates:
            probing_model = ProbingModel(input_dim, n_classes).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(probing_model.parameters(), lr=lr)

            min_loss = 1000
            n_unimproving_epochs = -1
            for i in range(n_epochs):
                probing_model.train()
                n_unimproving_epochs += 1
                if n_unimproving_epochs >= max_unimproving:
                    break
                for x_, y_ in train_loader:
                    optimizer.zero_grad()
                    y_scores = probing_model(x_.to(device))
                    loss = criterion(y_scores, y_.to(device))
                    loss.backward()
                    optimizer.step()

                loss = perf_cls(probing_model, val_loader, device)[0]
                if loss < min_loss:
                    n_unimproving_epochs = -1
                    min_loss = loss
                    torch.save(probing_model, path_to_base_folder() + 'tmp_models/best_' + str(lr)[2:])

            probing_model = torch.load(path_to_base_folder() + 'tmp_models/best_' + str(lr)[2:])
            results[lr] += torch.tensor([min_loss / n_fold, perf_cls(probing_model, test_loader, device)[1] / n_fold])

    min_loss = 1000
    result = 0
    for lr_result in results.values():
        if lr_result[0] < min_loss:
            min_loss = lr_result[0]
            result = lr_result[1]

    return result.item()

def perf_bb_reg(model, loader, device):
    w = 320
    h = 240
    criterion = torch.nn.MSELoss()
    model.eval()
    total_loss = evaluation = num = 0
    for x, y in loader:
        with torch.no_grad():
            y = y.to(device)
            y_scores = model(x.to(device))
            loss = criterion(y_scores, y.to(device))
            total_loss += loss.item()

            # Euclidian distance
            for i in range(len(y)):
                evaluation += math.sqrt(
                    (y[i][0] * w - y_scores[i][0] * w) ** 2 + (y[i][1] * h - y_scores[i][1] * h) ** 2)
                evaluation += math.sqrt(
                    (y[i][2] * w - y_scores[i][2] * w) ** 2 + (y[i][3] * h - y_scores[i][3] * h) ** 2)

            num += len(y)
    return math.sqrt(total_loss / num), evaluation / (2 * num)

def perf_reg(model, loader, device):
    criterion = torch.nn.MSELoss()
    eval_criterion = torch.nn.MSELoss(reduction='sum')
    model.eval()
    total_loss = evaluation = num = 0
    for x, y in loader:
        with torch.no_grad():
            y = y.to(device)
            y_scores = model(x.to(device))
            loss = criterion(y_scores, y.to(device))
            total_loss += loss.item()
            evaluation += eval_criterion(y_scores, y.to(device)).item()

            num += len(y)
    return math.sqrt(total_loss / num), math.sqrt(evaluation / num)

def kfold_reg(X, y, test_loader, input_dim, control, n_fold, device, n_epochs=200, max_unimproving=20, bb_reg=False):
    kf = KFold(n_splits=n_fold, shuffle=True)
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    results = {}
    for lr in learning_rates:
        results[lr] = torch.FloatTensor([0, 0])

    for train_index, test_index in tqdm(kf.split(X), total=n_fold):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if control:
            random.shuffle(y_train)

        train_set = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_set = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16)

        for lr in learning_rates:
            probing_model = ProbingModel(input_dim, 4 if bb_reg else 1).to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(probing_model.parameters(), lr=lr)

            min_loss = 1000
            n_unimproving_epochs = -1
            for i in range(n_epochs):
                probing_model.train()
                n_unimproving_epochs += 1
                if n_unimproving_epochs >= max_unimproving:
                    break
                for x_, y_ in train_loader:
                    optimizer.zero_grad()
                    y_scores = probing_model(x_.to(device))
                    loss = criterion(y_scores, y_.to(device))
                    loss.backward()
                    optimizer.step()

                if bb_reg:
                    loss = perf_bb_reg(probing_model, val_loader, device)[0]
                else:
                    loss = perf_reg(probing_model, val_loader, device)[0]
                if loss < min_loss:
                    n_unimproving_epochs = -1
                    min_loss = loss
                    torch.save(probing_model, '../tmp_models/best_' + str(lr)[2:])

            probing_model = torch.load('../tmp_models/best_' + str(lr)[2:])
            if bb_reg:
                perf = perf_bb_reg(probing_model, test_loader, device)[1] / n_fold
            else:
                perf = perf_reg(probing_model, test_loader, device)[1] / n_fold
            results[lr] += torch.tensor([min_loss / n_fold, perf])

    min_loss = 1000
    result = 0
    for lr_result in results.values():
        if lr_result[0] < min_loss:
            min_loss = lr_result[0]
            result = lr_result[1]

    return result.item()

