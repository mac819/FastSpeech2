import torch
import torch.nn as nn

from audioml.fastspeech.transformer import FFTBlock
from audioml.fastspeech.speech_feature_predictor import (
    DurationPredictor,
    LengthRegulator,
    PitchPredictor,
    EnergyPredictor
)
from audioml.fastspeech.utils import Embedding, get_sinusoid_encoding_table

class Encoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.n_fft_layers = cfg['n_fft_layers']
        self.fft_cfg = cfg['fft']
        self.embedding_config = cfg['embedding_layer']
        self.text_config = cfg['text']
        # Embedding Layer
        self.embedding_layer = Embedding(
            emb_dim=self.embedding_config['dims'],
            vocab_size=self.text_config['vocab_size'],
            token_context_length=self.embedding_config['context_length']
        )

        # FFT Block
        self.fft_layers = nn.ModuleList(
            [
                FFTBlock(cfg=self.fft_cfg) for _ in range(self.n_fft_layers)
            ]
        )

    def forward(self, input_ids, mask):
        fft_output = self.embedding_layer(input_ids)
        for fft in self.fft_layers:
            fft_output = fft(fft_output, mask)

        return fft_output
    

class VarianceAdaptor(nn.Module):

    def __init__(self, 
                 text_input_size,
                 mel_input_size, 
                 text_emb_dim=256,
                 filter_size=384, 
                 num_layers=2, 
                 kernel_size=3, 
                 stride=1, 
                 dropout_rate=0.1, 
                 cwt_scales=10):
        super().__init__()
        
        # Duration Predictor
        self.duration_predictor = DurationPredictor(
            input_size=text_input_size,
            filter_size=filter_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=stride
        )

        # Length Regulator
        self.lr = LengthRegulator()

        # Encoder-Decoder projection
        self.dec_proj = nn.Linear(
          text_emb_dim,
          mel_input_size
        )
        
        # Pitch Predictor
        self.pitch_predictor = PitchPredictor(
            input_size=mel_input_size,
            filter_size=filter_size,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            cwt_scales=cwt_scales
        )

        # Energy Predictor
        self.energy_predictor = EnergyPredictor(
            input_size=mel_input_size,
            filter_size=filter_size,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            stride=stride
        )


    def forward(self, x, src_mask, alpha=1.0, train=False, gt_duration=None):

        # Different input duration according to training or inference
        # Duration Prediction
        log_duration = self.duration_predictor(x, src_mask)

        # Processing on log_duration
        duration = torch.exp(log_duration) * alpha
        
        # Round to nearest integer and convert to int
        duration = torch.round(duration).squeeze(-1).long()
    
        # Ensure minimum duration of 1
        duration = torch.clamp_min(duration, 1)
        duration = duration.masked_fill(src_mask.bool(), 0)

        if train:
            if gt_duration == None:
                raise ValueError("Ground Truth of duration must pass in training mode.")
            
            # Length Regulator
            mel_hidden_state, mel_mask = self.lr(x, gt_duration)
        else:
            # Length Regulator
            mel_hidden_state, mel_mask = self.lr(x, duration)
        mel_hidden_state = self.dec_proj(mel_hidden_state)
    
        # Adding pitch prediction
        residual = mel_hidden_state
        pitch_output = self.pitch_predictor(mel_hidden_state, mel_mask)
        mel_hidden_state = pitch_output['pitch_embedding']
        mel_hidden_state = mel_hidden_state + residual

        # Adding Energy prediction
        residual = mel_hidden_state
        energy_output = self.energy_predictor(mel_hidden_state, mel_mask)
        mel_hidden_state = energy_output['energy_embedding']
        mel_hidden_state = mel_hidden_state + residual

        return mel_hidden_state, mel_mask, log_duration, duration, pitch_output, energy_output
    

# Input: (batch x seq_length x emb_dimension)
class MelDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_fft_layers = cfg['n_fft_layers']
        self.fft_cfg = cfg['fft']
        self.d_out = self.fft_cfg['d_out']
        self.n_mels = cfg['n_mels']
        self.context_length = cfg['context_length']

        self.pos_emb = nn.Parameter(
          get_sinusoid_encoding_table(self.context_length+1, self.d_out).unsqueeze(0),
          requires_grad=False
        )

        # FFT Block
        self.fft_layers = nn.ModuleList(
            [
                FFTBlock(cfg=self.fft_cfg) for _ in range(self.n_fft_layers)
            ]
        )

        self.mel_proj = nn.Linear(self.d_out, self.n_mels)
        
        # Remove ReLU activation since we need negative values for log space
        # The model should learn to output log-transformed mel-spectrograms
        self.output_scale = 1.0  # Can be tuned if needed
        

    def forward(self, x, mask): # --> (batch x seq_len x d_in)
        batch, seq_len, emb_dim = x.shape
        # Positional Embedding
        pos_emb = self.pos_emb[:, :seq_len]
        x = x + pos_emb
        for fft in self.fft_layers:
            x = fft(x, mask)

        mel_spec = self.mel_proj(x)
        # No activation - output should be in log space to match preprocessing
        mel_spec = mel_spec * self.output_scale
        mel_spec = mel_spec.masked_fill(mask.unsqueeze(-1).bool(), 0)
        return mel_spec
    

class Text2Mel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(cfg=cfg['encoder'])
        self.variance_adaptor = VarianceAdaptor(
            text_input_size=cfg['variance_adaptor']['text_input_size'],
            mel_input_size=cfg['variance_adaptor']['mel_input_size'],
            text_emb_dim=cfg['encoder']['embedding_layer']['dims'],
            filter_size=cfg['variance_adaptor']['filter_size'],
            num_layers=cfg['variance_adaptor']['num_layers'],
            kernel_size=cfg['variance_adaptor']['kernel_size'],
            stride=cfg['variance_adaptor']['stride'],
            dropout_rate=cfg['variance_adaptor']['drop_rate'],
            cwt_scales=cfg['variance_adaptor']['cwt_scales']
        )
        self.decoder = MelDecoder(cfg=cfg['decoder'])
    def forward(self, input_ids, src_mask, alpha=1.0, train=False, gt_duration=None):
        # Encoder
        encoder_output = self.encoder(input_ids, src_mask)

        # Variance Adaptor
        mel_hidden_state, mel_mask, log_duration, duration, pitch_output, energy_output = self.variance_adaptor(
            x=encoder_output,
            src_mask=src_mask,
            alpha=alpha,
            train=train,
            gt_duration=gt_duration
        )

        # Decoder
        mel_spec = self.decoder(mel_hidden_state, mel_mask)

        return {
            'mel_spec': mel_spec,
            'mel_mask': mel_mask,
            'log_duration': log_duration,
            'duration': duration,
            'pitch': pitch_output,
            'energy': energy_output
        }