import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import orbipy as op

import numpy as np
import hydra
import os
from omegaconf import DictConfig
import logging

from ml.lstm import TrajectoryDataset, OldLSTMClassifier, LSTMModel, FC_Class, LSTMClassifier, NewOldLSTMClassifier, GRUClassifier, CNNLSTMClassifier

log = logging.getLogger(__name__)

@hydra.main(config_name="config", config_path="config")
def main(cfg: DictConfig):

    X = np.loadtxt(cfg.data.x_train_path).reshape((-1, cfg.model.num_times, cfg.model.input_size))
    y = np.loadtxt(cfg.data.y_train_path).reshape((-1, cfg.model.num_times))

    if cfg.data.x_test_path != "None":
        X_test = np.loadtxt(cfg.data.x_test_path).reshape((-1, cfg.model.num_times, cfg.model.input_size))
        y_test = np.loadtxt(cfg.data.y_test_path).reshape((-1, cfg.model.num_times))
        lengths_test = np.full(X_test.shape[0], 140)

        X_test = torch.tensor(X_test[:], dtype=torch.float32)
        y_test = (y_test > 0)
        y_test = torch.tensor(y_test[:], dtype=torch.float32)
        lengths_test = torch.tensor(lengths_test[:], dtype=torch.long)

        if cfg.model.normalise:
            X_test[:, :, 0] -= op.crtbp3_model().L1

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

                loss = criterion(outputs, batch_y).mean()
                loss.backward()
                optimizer.step()

                logits = outputs[:, -1]
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
                true = batch_y[:, 0]


            accuracy += (preds == true).sum().item()
            acc_loss += loss.item()

        scheduler.step()

        model.eval()

        if cfg.data.x_test_path != "None":
            outputs_test = model(X_test).squeeze(-1)
            if cfg.model.pooling == "None":
                outputs_test = outputs_test[:, -1]
            else:
                pass
            preds = (torch.sigmoid(outputs_test) > 0.5).long().squeeze(-1)
            accuracy_test = (preds == y_test[:, 0]).sum().item()

            log.info(
                f"Epoch [{epoch + 1}/{cfg.training.num_epochs}], Loss: {acc_loss / len(dataloader):.8f}, Train Accuracy: {accuracy / X.shape[0]:.8f}, Test Accuracy: {accuracy_test / X_test.shape[0]:.8f}")
        else:
            log.info(
                f"Epoch [{epoch + 1}/{cfg.training.num_epochs}], Loss: {acc_loss / len(dataloader):.8f}, Train Accuracy: {accuracy / X.shape[0]:.8f}")
        save_path = f"{os.getcwd()}/model_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()
