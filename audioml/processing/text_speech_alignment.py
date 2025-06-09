import os
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import torchaudio.transforms as T
from nemo.collections.tts.models import AlignerModel
from nemo_text_processing.text_normalization.normalize import Normalizer


DATA = Path(__file__).parent.parent.parent / 'data'


class TTSTokenizer:

    def __init__(self, 
                 model="tts_en_radtts_aligner"
                 ):
        
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aligner = AlignerModel.from_pretrained(model).to(self.device)

        # Load Normalizer
        self.text_normalizer = Normalizer(input_case="cased", lang="en")

    def tokenize(self, raw_text):
        norm_text = self.text_normalizer.normalize(raw_text)
        text_tokens = self.aligner.tokenizer(norm_text)
        return text_tokens
    
    def batch_tokenize(self, raw_text: list):
        token_batch = [torch.tensor(self.tokenize(x)) for x in raw_text]

        token_lengths = torch.tensor([len(x) for x in token_batch])
        max_len = torch.max(token_lengths)
        # print(f"max_len: {max_len}")
        mask_tensor = torch.arange(max_len).unsqueeze(0).expand(len(token_batch), -1)
        mask_tensor = (mask_tensor >= token_lengths.unsqueeze(1)).int()

        # Pad token batch
        padded_token_batch = [
            F.pad(
                x,
                (0, max_len - len(x)),
                'constant',
                0
            ) for x in token_batch
        ]
        padded_token_batch = torch.stack(padded_token_batch, dim=0)

        return {
            'input_ids': padded_token_batch,
            'mask_ids': mask_tensor
        }


class SpeechAlignment:

    def __init__(self, 
                 aligner_model="tts_en_radtts_aligner",
                 target_sample_rate=22050,
                 ):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aligner_sample_rate = target_sample_rate
        self.aligner = AlignerModel.from_pretrained(aligner_model).to(self.device)


        # Load Normalizer
        self.text_normalizer = Normalizer(input_case="cased", lang="en")


    def align_text(self, audio_path, text_path):
        
        spec, spec_len = self.process_audio(path=audio_path)
        text_g2p, text, text_len = self.process_text(path=text_path)
        
        with torch.no_grad():
            attn_soft_tensor, attn_logprob_tensor = self.aligner(
                spec=spec, 
                spec_len=spec_len, 
                text=text, 
                text_len=text_len
            )

        # Call function to calculate each token's duration in frames
        durations = self.aligner.alignment_encoder.get_durations(attn_soft_tensor, text_len, spec_len).int()
        
        # Let's match them up. (We strip out the first and last duration due to zero-padding.)
        # durations_sum = 0
        # for t,d in zip(text_g2p, durations[0]):
        #     print(f"'{t}' duration: {d}")
        #     durations_sum += d

        # The following should be equal.
        # print(f"Total number of frames: {spec_len.item()}")
        # print(f"Sum of durations: {durations_sum}")

        return text_g2p, durations[0]


    def process_text(self, path):

        with open(path, 'r') as f:
            raw_text = f.read()
            
            # Text Normalization
            norm_text = self.text_normalizer.normalize(raw_text)

            # Grapheme (text) to Phoneme
            text_g2p = self.aligner.tokenizer.g2p(norm_text)
            # print(f"text_g2p: {text_g2p}")
            # print(f"Length: {len(text_g2p)}")

            # Update text_g2p with spaces
            text_g2p.insert(0, ' ')
            text_g2p.insert(len(text_g2p), ' ')

            # Grapheme (text) to tokens
            text_tokens = self.aligner.tokenizer(norm_text)
            # print(f"Text token: {text_tokens}")
            # print(f"Length: {len(text_tokens)}")
            # We need these to be torch tensors with a batch dimension before passing them in as input, of course
            text = torch.tensor(text_tokens, device=self.device).unsqueeze(0).long()
            text_len = torch.tensor(len(text_tokens), device=self.device).unsqueeze(0).long()
            # print("\nAfter unsqueezing...")
            # print(f"Text input shape: {text.shape}")
            # print(f"Text length shape: {text_len.shape}")

        return text_g2p, text, text_len


    def process_audio(self, path):

        audio_arr, sr = torchaudio.load(str(path))

        # Resampling audio
        resampler = T.Resample(orig_freq=sr,new_freq=self.aligner_sample_rate, resampling_method="sinc_interp_hann")
        if sr != self.aligner_sample_rate:
            audio_arr = resampler(audio_arr)

        # Retrieve audio length for the model's preprocessor
        audio_len = torch.tensor(audio_arr.shape[-1], device=self.device).long()
        audio_len = torch.tensor(audio_len).unsqueeze(0)
        # Adding batch dimension
        if audio_arr.ndim == 1:
            audio_arr = audio_arr.type(torch.float).unsqueeze(0)

        # Generate the spectrogram!
        spec, spec_len = self.aligner.preprocessor(input_signal=audio_arr, length=audio_len)
        # print(f"Spec batch shape: {spec.shape}")

        return spec, spec_len


if __name__=="__main__":
    # This should be set to whatever sample rate your model was trained on
    in_corpus_path = DATA / 'interim' / 'in_corpus'
    out_dir = DATA / 'processed' / 'lj_speech_feature'
    os.makedirs(str(out_dir), exist_ok=True)

    duration_out_dir = out_dir / 'duration'
    os.makedirs(str(duration_out_dir), exist_ok=True)

    audio_files = [x for x in in_corpus_path.glob("*.wav")]

    text_speech_aligner = SpeechAlignment()
    for audio_file in tqdm(audio_files):
        text_path = audio_file.parent / audio_file.name.replace('.wav', '.txt')
        with open(text_path, 'r') as f:
            raw_text = f.read()

        # print(f"Raw text: {raw_text}")

        if text_path.exists() and audio_file.exists():
            text_g2p, durations = text_speech_aligner.align_text(
                audio_path=audio_file,
                text_path=text_path
            )
            # print(f"Length text_g2p: {len(text_g2p)}\nLength Duration: {durations.shape}")
            # print(f"text_g2p: {text_g2p}\ndurations: {durations}")
            fname = audio_file.name.replace(".wav", "")
            duration_path = duration_out_dir / f"{fname}.npy"
            
            np.save(
                duration_path, durations.detach().numpy()
            )
        else:
            print("Text and Audio path is not available.")