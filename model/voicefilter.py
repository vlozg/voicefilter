import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceFilter(nn.Module):
    def __init__(self, hp):
        super(VoiceFilter, self).__init__()
        self.hp = hp
        assert hp.audio.n_fft // 2 + 1 == hp.audio.num_freq == hp.model.fc2_dim, \
            "stft-related dimension mismatch"

        self.lstm = nn.LSTM(
            hp.audio.num_freq + hp.embedder.emb_dim,
            hp.model.lstm_dim,
            batch_first=True,
            bidirectional=hp.model.bidirection)

        lstm_dim = 2*hp.model.lstm_dim if hp.model.bidirection else hp.model.lstm_dim
        self.fc1 = nn.Linear(lstm_dim, hp.model.fc1_dim)
        self.fc2 = nn.Linear(hp.model.fc1_dim, hp.model.fc2_dim)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2) # [B, T, num_freq + emb_dim]

        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x
