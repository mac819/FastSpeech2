import torch
from pathlib import Path
from torch.utils.data import DataLoader
# from audioml.processing.text_speech_alignment import TTSTokenizer
from audioml.dataset.feature_dataset import SpeechFeatureDataset, collate_function


PACKAGE_PATH = Path(__file__).parent
DATA_PATH = PACKAGE_PATH.parent / 'data'
FEATURE_DIR = DATA_PATH / 'processed' / 'lj_speech_feature'

vocab_size=114




if __name__=="__main__":
    
    feature_dataset = SpeechFeatureDataset(feature_dir=FEATURE_DIR)
    batch_size = 8
    shuffle=True
    feature_dataloader = DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_function
    )
    

    batch = next(iter(feature_dataloader))
    # Saving batch for experimentation
    torch.save(batch, "temp_batch.pt")
    print("Debug checkpoint")