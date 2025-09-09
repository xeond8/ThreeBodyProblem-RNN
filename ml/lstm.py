import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.tensor(X[:], dtype=torch.float32)
        self.y = torch.tensor(y[:], dtype=torch.float32)
        if lengths is not None:
            self.lengths = torch.tensor(lengths[:], dtype=torch.long)
        else:
            self.lengths = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.lengths is not None:
            return self.X[idx], self.y[idx], self.lengths[idx]
        else:
            return self.X[idx], self.y[idx]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None, return_weights=False):
        scores = self.attn(x).squeeze(-1)

        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            scores[~mask] = float("-inf")

        attn_weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        if return_weights:
            return pooled, attn_weights
        return pooled


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_lin_before, num_lin_after, dropout_rate=0.4,
                 pooling="max"):
        super(LSTMClassifier, self).__init__()

        linear_before = []
        linear_before.append(nn.Linear(input_size, hidden_size))
        for _ in range(1, num_lin_before):
            linear_before.append(nn.ReLU())
            linear_before.append(nn.Dropout(dropout_rate))
            linear_before.append(nn.Linear(hidden_size, hidden_size))
        self.linear_before = nn.Sequential(*linear_before)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_lstm_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_lstm_layers > 1 else 0)

        linear_after = []
        for _ in range(num_lin_after - 1):
            linear_after.append(nn.ReLU())
            linear_after.append(nn.Dropout(dropout_rate))
            linear_after.append(nn.Linear(hidden_size, hidden_size))
        self.linear_after = nn.Sequential(*linear_after)

        self.fc_out = nn.Linear(hidden_size, 1)

        self.pooling_type = pooling
        if pooling == "attention":
            self.attn_pool = AttentionPooling(hidden_size)

    def forward(self, x, lengths=None):
        x = self.linear_before(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hn, cn) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, (hn, cn) = self.lstm(x)

        if self.pooling_type == "mean":
            if lengths is not None:
                mask = torch.arange(lstm_out.size(1), device=x.device)[None, :] < lengths[:, None]
                pooled = (lstm_out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
            else:
                pooled = lstm_out.mean(dim=1)
        elif self.pooling_type == "max":
            if lengths is not None:
                mask = torch.arange(lstm_out.size(1), device=x.device)[None, :] < lengths[:, None]
                lstm_out[~mask] = float("-inf")
                pooled, _ = lstm_out.max(dim=1)
            else:
                pooled, _ = lstm_out.max(dim=1)
        elif self.pooling_type == "attention":
            pooled = self.attn_pool(lstm_out, lengths)

        pooled = self.linear_after(pooled)

        return self.fc_out(pooled)


class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers,
                 num_conv_layers=2, conv_channels=64,
                 num_linear_layers=2, dropout_rate=0.2,
                 pooling="max"):
        super(CNNLSTMClassifier, self).__init__()

        convs = []
        in_channels = input_size
        for i in range(num_conv_layers):
            convs.append(nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout_rate))
            in_channels = conv_channels
        self.conv = nn.Sequential(*convs)

        self.lstm = nn.LSTM(conv_channels, hidden_size, num_lstm_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_lstm_layers > 1 else 0)

        linear_after = []
        for _ in range(num_linear_layers - 1):
            linear_after.append(nn.ReLU())
            linear_after.append(nn.Dropout(dropout_rate))
            linear_after.append(nn.Linear(hidden_size, hidden_size))
        self.linear_after = nn.Sequential(*linear_after)

        self.fc_out = nn.Linear(hidden_size, 1)

        self.pooling_type = pooling
        if pooling == "attention":
            self.attn_pool = AttentionPooling(hidden_size)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hn, cn) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, (hn, cn) = self.lstm(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hn, cn) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, (hn, cn) = self.lstm(x)

        if self.pooling_type == "mean":
            if lengths is not None:
                mask = torch.arange(lstm_out.size(1), device=x.device)[None, :] < lengths[:, None]
                pooled = (lstm_out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
            else:
                pooled = lstm_out.mean(dim=1)
        elif self.pooling_type == "max":
            if lengths is not None:
                mask = torch.arange(lstm_out.size(1), device=x.device)[None, :] < lengths[:, None]
                lstm_out[~mask] = float("-inf")
                pooled, _ = lstm_out.max(dim=1)
            else:
                pooled, _ = lstm_out.max(dim=1)
        elif self.pooling_type == "attention":
            pooled = self.attn_pool(lstm_out, lengths)
        elif self.pooling_type == "None":
            pooled = lstm_out

        pooled = self.linear_after(pooled)

        return self.fc_out(pooled)


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_gru_layers, num_lin_before, num_lin_after, dropout_rate=0.4,
                 pooling="max"):
        super(GRUClassifier, self).__init__()

        linear_before = []
        linear_before.append(nn.Linear(input_size, hidden_size))
        for _ in range(1, num_lin_before):
            linear_before.append(nn.ReLU())
            linear_before.append(nn.Dropout(dropout_rate))
            linear_before.append(nn.Linear(hidden_size, hidden_size))
        self.linear_before = nn.Sequential(*linear_before)

        self.gru = nn.GRU(hidden_size, hidden_size, num_gru_layers, batch_first=True,
                          dropout=dropout_rate if num_gru_layers > 1 else 0)

        linear_after = []
        for _ in range(num_lin_after - 1):
            linear_after.append(nn.ReLU())
            linear_after.append(nn.Dropout(dropout_rate))
            linear_after.append(nn.Linear(hidden_size, hidden_size))
        self.linear_after = nn.Sequential(*linear_after)

        self.fc_out = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.pooling_type = pooling

    def forward(self, x, lengths=None, h0=None):

        x = self.linear_before(x)

        if not lengths is None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hn = self.gru(packed, h0)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            mask = torch.arange(gru_out.size(1), device=lengths.device)[None, :] < lengths[:, None]
            if self.pooling_type == "mean":
                pooled = (gru_out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
            elif self.pooling_type == "max":
                gru_out[~mask] = float("-inf")
                pooled, _ = gru_out.max(dim=1)

        else:
            gru_out = self.gru(x, h0)[0]
            if self.pooling_type == "mean":
                pooled = gru_out.mean(dim=1)
            elif self.pooling_type == "max":
                pooled, _ = gru_out.max(dim=1)
            elif self.pooling_type == "None":
                pooled = gru_out

        pooled = self.linear_after(pooled)

        out = self.fc_out(pooled)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_linear_layers, output_size, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.linear_before = nn.ModuleList()
        self.linear_before.append(nn.Linear(input_size, hidden_size))
        for i in range(1, num_linear_layers):
            self.linear_before.append(nn.Linear(hidden_size, hidden_size))
            self.linear_before.append(nn.Dropout(dropout_rate))

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_lstm_layers, batch_first=True, dropout=dropout_rate)

        self.linear_after = nn.ModuleList()
        for i in range(num_linear_layers - 1):
            self.linear_after.append(nn.Linear(hidden_size, hidden_size))
            self.linear_after.append(nn.Dropout(dropout_rate))
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x, h0=None, c0=None):
        lstm_in = x.clone().detach()
        for fc in self.linear_before:
            lstm_in = self.relu(fc(lstm_in))

        if h0 is None or c0 is None:
            h0 = torch.zeros(self.lstm.num_layers, lstm_in.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers, lstm_in.size(0), self.lstm.hidden_size).to(x.device)

        lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))

        for fc in self.linear_after:
            lstm_out = self.relu(fc(lstm_out))
        output = self.fc_out(lstm_out)
        return output, (hn, cn)


class NewOldLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_lin_before, num_lin_after, dropout_rate=0.4):
        super(NewOldLSTMClassifier, self).__init__()
        linear_before = []
        linear_before.append(nn.Linear(input_size, hidden_size))
        for _ in range(1, num_lin_before):
            linear_before.append(nn.ReLU())
            linear_before.append(nn.Dropout(dropout_rate))
            linear_before.append(nn.Linear(hidden_size, hidden_size))
        self.linear_before = nn.Sequential(*linear_before)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )

        linear_after = []
        for _ in range(num_lin_after - 1):
            linear_after.append(nn.ReLU())
            linear_after.append(nn.Dropout(dropout_rate))
            linear_after.append(nn.Linear(hidden_size, hidden_size))
        self.linear_after = nn.Sequential(*linear_after)

        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None, c0=None):
        lstm_in = self.linear_before(x)

        if h0 is None or c0 is None:
            h0 = torch.zeros(
                self.lstm.num_layers, lstm_in.size(0), self.lstm.hidden_size,
                device=x.device
            )
            c0 = torch.zeros(
                self.lstm.num_layers, lstm_in.size(0), self.lstm.hidden_size,
                device=x.device
            )

        lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))

        features = self.linear_after(lstm_out)

        output = self.fc_out(features)
        return output


class OldLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_linear_layers, dropout_rate=0.4):
        super(OldLSTMClassifier, self).__init__()
        self.linear_before = nn.ModuleList()
        self.linear_before.append(nn.Linear(input_size, hidden_size))
        for i in range(1, num_linear_layers):
            self.linear_before.append(nn.Linear(hidden_size, hidden_size))
            self.linear_before.append(nn.Dropout(dropout_rate))

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_lstm_layers, batch_first=True)

        self.linear_after = nn.ModuleList()
        for i in range(num_linear_layers - 1):
            self.linear_after.append(nn.Linear(hidden_size, hidden_size))
            self.linear_after.append(nn.Dropout(dropout_rate))

        self.fc_out = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, h0=None, c0=None):
        lstm_in = x.clone().detach()
        for fc in self.linear_before:
            lstm_in = self.relu(fc(lstm_in))

        if h0 is None or c0 is None:
            h0 = torch.zeros(self.lstm.num_layers, lstm_in.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers, lstm_in.size(0), self.lstm.hidden_size).to(x.device)

        lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))

        for fc in self.linear_after:
            lstm_out = self.relu(fc(lstm_out))
            lstm_out = fc(lstm_out)

        output = self.fc_out(lstm_out)
        return output


class FC_Class(nn.Module):
    def __init__(self, input_size, sizes, dropout_rate=0.5):
        super(FC_Class, self).__init__()
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(input_size, sizes[0]))
        for i in range(1, len(sizes)):
            self.linear.append(nn.Linear(sizes[i - 1], sizes[i]))
            self.linear.append(nn.Dropout(dropout_rate))

        self.fc_out = nn.Linear(sizes[-1], 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for fc in self.linear:
            x = self.relu(fc(x))
        x = self.fc_out(x)
        return self.sigmoid(x)
