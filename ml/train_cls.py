import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import orbipy as op
from corrections.corrections import custom_base_correction, custom_station_keeping, zero_correction
import numpy as np
import pandas as pd
import hydra
import os
from omegaconf import DictConfig
import logging
import time
from tqdm import trange

from ml.lstm import TrajectoryDataset, OldLSTMClassifier, LSTMModel, FC_Class, LSTMClassifier, NewOldLSTMClassifier, GRUClassifier, CNNLSTMClassifier

log = logging.getLogger(__name__)

@hydra.main(config_name="config", config_path="config")
def main(cfg: DictConfig):
    dfdv = pd.read_csv(cfg.data.csv_path)
    dfdv = pd.concat([dfdv.iloc[cfg.data.slice_start:cfg.data.slice_end],
                      dfdv.iloc[cfg.data.slice_start2:cfg.data.slice_end2]])


    X = np.loadtxt(cfg.data.x_train_path).reshape((-1, cfg.model.num_times, cfg.model.input_size))
    y = np.loadtxt(cfg.data.y_train_path).reshape((-1, cfg.model.num_times))

    X_test = np.loadtxt(
        "C:/Users/levpy/OneDrive/Рабочий стол/Уник/Точки либрации/data/x_test_355_1_ran(140)8pie-6.csv").reshape(-1,
                                                                                                                 140, 6)
    y_test = np.loadtxt(
        "C:/Users/levpy/OneDrive/Рабочий стол/Уник/Точки либрации/data/y_test_355_1_ran(140)8pie-6.csv").reshape(-1,
                                                                                                                 140)[:,
             0]
    lengths_test = np.full(X_test.shape[0], 140)

    '''X_test = torch.tensor(X_test[:], dtype=torch.float32)
    X_test[:, :, 0] -= op.crtbp3_model().L1
    y_test = (y_test > 0)
    y_test = torch.tensor(y_test[:], dtype=torch.float32)
    lengths_test = torch.tensor(lengths_test[:], dtype=torch.long)'''

    if cfg.data.lengths_train_path != "None":
        lengths = np.loadtxt(cfg.data.lengths_train_path)
    else:
        lengths = None

    y = np.array(y > 0, dtype=int)

    if cfg.model.normalise:
        X[:, :, 0] -= op.crtbp3_model().L1

    dataset = TrajectoryDataset(X, y, lengths)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    if cfg.model.type == "GRUClassifier":
        model = GRUClassifier(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_rnn_layers, cfg.model.num_lin_layers_before, cfg.model.num_lin_layers_after, pooling=cfg.model.pooling)
    elif cfg.model.type == "CNNLSTMClassifier":
        model = CNNLSTMClassifier(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_rnn_layers, cfg.model.num_conv_layers, cfg.model.conv_size, cfg.model.num_lin_layers_after, pooling=cfg.model.pooling)
    elif cfg.model.type == "NewOldLSTMClassifier":
        model = NewOldLSTMClassifier(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_rnn_layers, cfg.model.num_lin_layers_before, cfg.model.num_lin_layers_after)
    elif cfg.model.type == "OldLSTMClassifier":
        model = OldLSTMClassifier(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_rnn_layers, cfg.model.num_lin_layers_before)
    else:
        model = LSTMClassifier(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_rnn_layers, cfg.model.num_lin_layers_before, cfg.model.num_lin_layers_after, pooling=cfg.model.pooling)
    #model.load_state_dict(torch.load("C:/Users/levpy/OneDrive/Рабочий стол/Уник/Точки либрации/ml/outputs/2025-09-05/20-46-51/model_10.pt"))
    #model = NewOldLSTMClassifier(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_lstm_layers, cfg.model.num_lin_layers)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=cfg.training.gamma)
    for epoch in range(cfg.training.num_epochs):
        model.train()
        acc_loss = 0
        accuracy = 0

        #for batch_X, batch_y, batch_lengths in dataloader:
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            outputs_seq = model(batch_X)
            outputs = outputs_seq.squeeze(-1)

            if cfg.model.pooling == "max" or cfg.model.pooling == "mean" or cfg.model.pooling == "attention":
                batch_y = batch_y[:, 0]
                loss = criterion(outputs, batch_y).mean()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                preds = outputs > 0
                true = batch_y


            elif cfg.model.pooling == "None":
                '''batch_y = batch_y[:, :max(batch_lengths)]

                mask = torch.arange(max(batch_lengths), device=batch_X.device).unsqueeze(0) < batch_lengths.unsqueeze(1)
                mask = mask.to(outputs_seq.dtype)
                loss_all = criterion(outputs, batch_y)
                mask_loss = loss_all * mask
                loss = mask_loss.sum() / mask.sum()'''

                loss = criterion(outputs, batch_y).mean()
                loss.backward()
                optimizer.step()

                '''logits = outputs_seq[torch.arange(outputs_seq.size(0)), batch_lengths - 1]
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
                true = batch_y[torch.arange(batch_y.size(0)), batch_lengths - 1].long()'''

                logits = outputs[:, -1]
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
                true = batch_y[:, 0]


            accuracy += (preds == true).sum().item()
            acc_loss += loss.item()

        scheduler.step()

        model.eval()

        '''outputs_test = model(X_test, lengths_test).squeeze(-1)
        print(outputs_test.shape)
        accuracy_test = ((outputs_test > 0) == y_test).sum().item()'''

        '''cor = custom_base_correction(model, direction=op.unstable_direction(op.crtbp3_model()), param_init=0,
                                     param_delta=1e-9, param_n_points=3, tol=1e-14, max_iter_bisect=100,
                                     max_iter=5, Nt=cfg.model.num_times, T=cfg.model.T, skipt=cfg.model.skipt, normalise=cfg.model.normalise)
        dv = 0
        for i in range(dfdv.shape[0]):
            try:
                s = np.array(dfdv.iloc[i, 1:7])
                sk = custom_station_keeping(op.crtbp3_model(), zero_correction(), cor, verbose=False, rev=np.pi/2)
                _ = sk.prop(s, N=20, ret_df=True)
                dv += np.linalg.norm(np.array(sk.dvout)[:, 4:], axis=1).mean() < 1e-9

            except Exception as e:
                continue'''

        log.info(
            f"Epoch [{epoch + 1}/{cfg.training.num_epochs}], Loss: {acc_loss / len(dataloader):.8f}, Train Accuracy: {accuracy / X.shape[0]:.8f}")
        save_path = f"{os.getcwd()}/model_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()
