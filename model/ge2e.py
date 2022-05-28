import torch
import torch.nn as nn


class LinearNorm(nn.Module):
    def __init__(self, config):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(config.lstm_hidden, config.emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):
    def __init__(self, config):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(config.num_mels,
                            config.lstm_hidden,
                            num_layers=config.lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(config)
        self.config = config

    def forward(self, x):
        # (num_mels, T)
        x = x.unfold(1, self.config.window, self.config.stride) # (num_mels, T', window)
        x = x.permute(1, 2, 0) # (T', window, num_mels)
        x, _ = self.lstm(x) # (T', window, lstm_hidden)
        x = x[:, -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(0) / x.size(0) # (emb_dim), average pooling over time frames
        return x
