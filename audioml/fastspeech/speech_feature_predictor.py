import torch
import torch.nn as nn
import torch.nn.functional as F

from audioml.fastspeech.utils import TransposedLayerNorm

class DurationPredictor(nn.Module):

    def __init__(self, input_size, filter_size=384, num_layers=2, kernel_size=3, stride=1, dropout_rate=0.1):
        super().__init__()
        
        # Convolutional layers for pitch prediction
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size if i == 0 else filter_size, 
                    out_channels=filter_size,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                TransposedLayerNorm(filter_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for i in range(num_layers)
        ]) # --> (batch x in_channel x sequence_length) = (batch x filter_size x sequence_length)

        self.linear_proj = nn.Linear(filter_size, 1)

    
    def forward(self, x, mask):
        # Transpose for convolution (channel first)
        x = x.transpose(1, 2) # --> (batch x sequence_length x emb_dim) = (batch x emb_dim x sequence_length)
        
        # Pass through convolutional layers
        for conv_layer in self.convolutions: # --> (batch x emb_dim x sequence_length) = (batch x filter_size x sequence_length)
            conv, layerNorm, relu, dropout = conv_layer

            x = conv(x)
            x = x.masked_fill(mask.unsqueeze(1).bool(), 0)

            x = layerNorm(x)
            x = x.masked_fill(mask.unsqueeze(1).bool(), 0)

            x = relu(x)
            x = dropout(x)
        
        x = x.transpose(1, 2) # --> (batch x filter_size x sequence_length) = (batch x sequence_length x filter_size)

        log_duration = self.linear_proj(x).squeeze(-1) # --> (batch x seq_length x filter_size) = (batch x seq_length)
        
        return log_duration
    

class LengthRegulator(nn.Module):

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def LR(self, x, duration):
        """
        x --> (batch_size, sequence_length, embedding_size)
        duration --> (batch_size, max_sequence_length) ==> 2D tensor at phone sequence level
        For Training: Ground truth duration will be used
        For Inference: Predicted duration will be used
        """
        output = list()
        mel_len = list()

        for phon_embedding, phon_duration in zip(x, duration):
            """
            phon_embedding --> (phon_sequence, embedding_size)
            phon_duration --> (max_sequence_length)
            """
            expanded_embedding = self._expand(phon_embedding, phon_duration)
            output.append(expanded_embedding)
            mel_len.append(expanded_embedding.shape[0])

        mel_len = torch.tensor(mel_len)
        max_mel_len = torch.max(mel_len)
        # Padding of expanded phoneme --> mel_embedding
        padded_mel_embedding = list()
        for idx, expanded in enumerate(output):
            padded_mel_embedding.append(
                F.pad(
                    expanded,
                    (0, 0, 0, max_mel_len - mel_len[idx]),
                    'constant',
                    0
                )
            )
        phon_expanded_embedding = torch.stack(padded_mel_embedding, dim=0).to(self.device)
        # Logic for mel_mask
        mel_mask = (torch.arange(start=0, end=max_mel_len) >= mel_len.unsqueeze(-1)).int().to(self.device)
        return phon_expanded_embedding, mel_mask
        

    def _expand(self, phon_embedding, phon_duration):
        out = list()
        for idx, vec in enumerate(phon_embedding):
            expand_size = phon_duration[idx].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))

        return torch.cat(out, 0)

    def forward(self, x, duration):
        mel_embedding, mel_mask = self.LR(x, duration)
        return mel_embedding, mel_mask
    

class PitchPredictor(nn.Module):
    def __init__(
        self, 
        input_size, 
        filter_size=384, 
        kernel_size=3, 
        dropout_rate=0.1, 
        num_layers=2,
        cwt_scales=10,  # Number of CWT scales in pitch spectrogram
        pitch_bins=256,
        pitch_embeddings=384
    ):
        """
        Pitch Predictor for FastSpeech2
        
        Args:
            input_size (int): Size of input encoder representations --> 384
            filter_size (int): Hidden layer size for predictor --> 384
            kernel_size (int): Convolution kernel size --> 3
            dropout_rate (float): Dropout probability --> 0.1
            num_layers (int): Number of convolutional layers --> 2
            cwt_scales (int): Number of scales in pitch spectrogram --> 10
        """
        super(PitchPredictor, self).__init__()
        
        # Convolutional layers for pitch prediction
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size if i == 0 else filter_size, 
                    out_channels=filter_size,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                TransposedLayerNorm(filter_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for i in range(num_layers)
        ]) # --> (batch x in_channel x sequence_length) = (batch x filter_size x sequence_length)

        
        self.pitch_spectrogram_head = nn.Linear(
            filter_size,
            cwt_scales
        ) # (batch x seqence_length x filter_size) --> (batch x seqence_length x cwt_scale)

        self.stats_projection = nn.Linear(filter_size, 2) # --> (batch x filter_size) = (batch x 2)

        self.pitch_embedding = nn.Embedding(pitch_bins, pitch_embeddings)

        self.register_buffer(
            "icwt_weights",
            (torch.arange(cwt_scales) + 2.5)**(-2.5)
        ) # --> (cwt_scales, )
        
    def forward(self, encoder_output, mask): # --> encoder_output: (batch x sequence_length x embedding_dimension)
        """
        Forward pass for pitch prediction
        
        Args:
            encoder_output (torch.Tensor): Encoder representations
                Shape: [batch_size, sequence_length, input_size]
        
        Returns:
            dict: 
                - pitch_spectrogram: Predicted pitch spectrogram
                - pitch_mean: Predicted pitch mean
                - pitch_std: Predicted pitch standard deviation
        """
        batch, seq_len, emb_dim = encoder_output.shape
        
        # Transpose for convolution (channel first)
        x = encoder_output.transpose(1, 2) # --> (batch x sequence_length x emb_dim) = (batch x emb_dim x sequence_length)
        
        # Pass through convolutional layers
        for conv_layer in self.convolutions: # --> (batch x emb_dim x sequence_length) = (batch x filter_size x sequence_length)
            conv, layerNorm, relu, dropout = conv_layer

            x = conv(x)
            x = x.masked_fill(mask.unsqueeze(1).bool(), 0)

            x = layerNorm(x)
            x = x.masked_fill(mask.unsqueeze(1).bool(), 0)

            x = relu(x)
            x = dropout(x)
        
        # Predict pitch spectrogram
        x = x.transpose(1, 2) # --> (batch x fiter_size x sequence_length) = (batch x sequence_length x filter_size)
        pitch_spectrogram = self.pitch_spectrogram_head(x) # --> (batch x sequence_length x filter_size) = (batch x sequence_length x cwt_scale)
        pitch_spectrogram = pitch_spectrogram.masked_fill(mask.unsqueeze(-1).bool(), 0)
        
        # Predict pitch mean and std
        global_vector = torch.mean(x, dim=1) # --> (batch x sequence_length x filter_size) = (batch x filter_size)
        pitch_stats = self.stats_projection(global_vector) # --> (batch x filter_size) = (batch x  2)

        pitch_mean = pitch_stats[:, 0]
        pitch_std = pitch_stats[:, 1] # Standard deviation should have positivity constraint
        pitch_std = torch.exp(pitch_std) ** 0.5 # Positivity constraint

        # Reconstructing pitch contour
        weights = self.icwt_weights.unsqueeze(0).expand(seq_len, -1) # --> (sequence_length x cwt_scale)
        f0_norm = torch.sum(pitch_spectrogram * weights, dim=-1)
        f0 = (f0_norm * pitch_std.unsqueeze(1).expand(-1, seq_len)) + pitch_mean.unsqueeze(1).expand(-1, seq_len)
        f0 = f0.masked_fill(mask.bool(), 0)

        # Pitch quantization
        f0_min = torch.min(f0, dim=-1, keepdim=True).values
        f0_max = torch.max(f0, dim=-1, keepdim=True).values
        bins = (f0_max - f0_min) / 256
        bins = bins + 1e-6

        quantized_pitch = torch.floor((f0 - f0_min) / bins).clamp(0, 255).to(torch.int32)
        quantized_pitch = quantized_pitch.masked_fill(mask.bool(), 0)
        
        pitch_embedding = self.pitch_embedding(quantized_pitch)
        pitch_embedding = pitch_embedding.masked_fill(mask.unsqueeze(-1).bool(), 0)

        
        return {
            'pitch_spectrogram': pitch_spectrogram,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'reconstructed_f0': f0,
            'pitch_embedding': pitch_embedding
        }
    

class EnergyPredictor(nn.Module):

    def __init__(self, 
                 input_size, 
                 filter_size=384, 
                 num_layers=2, 
                 kernel_size=3, 
                 stride=1, 
                 dropout_rate=0.1, 
                 energy_bins=256, 
                 energy_embedding=384):
        super().__init__()
        
        # Convolutional layers for pitch prediction
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size if i == 0 else filter_size, 
                    out_channels=filter_size,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                TransposedLayerNorm(filter_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for i in range(num_layers)
        ]) # --> (batch x in_channel x sequence_length) = (batch x filter_size x sequence_length)

        self.linear_proj = nn.Linear(filter_size, 1)
        self.energy_embedding = nn.Embedding(energy_bins, energy_embedding) # --> (batch x sequence_length) = (batch x sequence_length x energy_embedding)

    def forward(self, x, mask):
        # Transpose for convolution (channel first)
        x = x.transpose(1, 2) # --> (batch x sequence_length x emb_dim) = (batch x emb_dim x sequence_length)
        
        # Pass through convolutional layers
        for conv_layer in self.convolutions: # --> (batch x emb_dim x sequence_length) = (batch x filter_size x sequence_length)
            conv, layerNorm, relu, dropout = conv_layer

            x = conv(x)
            x = x.masked_fill(mask.unsqueeze(1).bool(), 0)

            x = layerNorm(x)
            x = x.masked_fill(mask.unsqueeze(1).bool(), 0)

            x = relu(x)
            x = dropout(x)
        
        x = x.transpose(1, 2) # --> (batch x filter_size x sequence_length) = (batch x sequence_length x filter_size)

        energy_pred = self.linear_proj(x).squeeze(-1) # --> (batch x sequence_length x filter_size) = (batch x sequence_length x 1)
        energy_pred = energy_pred.masked_fill(mask.bool(), 0)

        # Energy Quantization
        energy_min = torch.min(energy_pred, dim=-1, keepdim=True).values
        energy_max = torch.max(energy_pred, dim=-1, keepdim=True).values
        bins = (energy_max - energy_min) / 256
        bins = bins + 1e-6

        quantized_energy = torch.floor((energy_pred - energy_min) / bins).clamp(0, 255).to(torch.int32)
        quantized_energy = quantized_energy.masked_fill(mask.bool(), 0)

        
        energy_embedding = self.energy_embedding(quantized_energy)
        energy_embedding = energy_embedding.masked_fill(mask.unsqueeze(-1).bool(), 0)
        
        return {
            'raw_energy': energy_pred,
            'quantized_energy': quantized_energy,
            'energy_embedding': energy_embedding
        }