import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from audioml.processing.text_speech_alignment import TTSTokenizer


PACKAGE_PATH = Path(__file__).parent.parent
DATA_PATH = PACKAGE_PATH.parent / 'data'
FEATURE_DIR = DATA_PATH / 'processed' / 'lj_speech_feature'


# Model gets trained on following ground truth
#   1. Mel-Spectrogram
#   2. Pitch Spectrogram
#   3. Energy
#   4. Duration

class SpeechFeatureDataset(Dataset):

    def __init__(self, feature_dir, batch_size, sort=True, drop_last=False):
        super().__init__()
        self.batch_size = batch_size
        self.sort = sort
        self.drop_last = drop_last
        self.energy_dir = Path(feature_dir) / 'energy'
        self.duration_dir = Path(feature_dir) / 'duration'
        self.mel_dir = Path(feature_dir) / 'mel_spec'
        self.pitch_dir = Path(feature_dir) / 'pitch'
        self.transcript_dir = Path(DATA_PATH) / 'interim'

        self.tokenizer = TTSTokenizer()

        self.files = self.__files_list()
        # print(f"files: {self.files}")
        # print(self.files)
        # Get all the available file paths
        self.feature_path = []
        for file in self.files:
            mel_spec_path = self.__get_mel_path(file=file)
            energy_path = self.__get_energy_path(file=file)
            duration_path = self.__get_duration_path(file=file)
            pitch_contour_path = self.__get_pitch_contour_path(file=file)
            pitch_contour_mean_path = self.__get_pitch_contour_mean_path(file=file)
            pitch_contour_std_path = self.__get_pitch_contour_std_path(file=file)
            pitch_spec_path = self.__get_pitch_spec_path(file=file)
            text_path = self.__get_transcript_path(file=file)
            # print([mel_spec_path, energy_path, 
            #        duration_path, pitch_contour_path,
            #        pitch_contour_mean_path, pitch_contour_std_path,
            #        pitch_spec_path])
            if all([mel_spec_path, energy_path, 
                   duration_path, pitch_contour_path,
                   pitch_contour_mean_path, pitch_contour_std_path,
                   pitch_spec_path, text_path]):
                self.feature_path.append((
                    file,
                    text_path,
                    mel_spec_path,
                    energy_path,
                    duration_path,
                    pitch_contour_path,
                    pitch_contour_mean_path,
                    pitch_contour_std_path,
                    pitch_spec_path
                ))
        # print(f"feature_paths: {self.feature_path}")

    def __get_transcript_path(self, file):
        path = self.transcript_dir / f"in_corpus/{file}.txt"
        # print(f"transcript path: {path}")
        if path.exists():
            return path
        return False

    def __get_mel_path(self, file):
        path = self.mel_dir / f"mel_{file}.npy"
        if path.exists():
            return path
        # print(f"Unable to load: {path}")
        return False
    
    def __get_energy_path(self, file):
        path = self.energy_dir / f"energy_{file}.npy"
        if path.exists():
            return path
        # print(f"Unable to load: {path}")
        return False
    
    def __get_duration_path(self, file):
        path = self.duration_dir / f"{file}.npy"
        if path.exists():
            return path
        # print(f"Unable to load: {path}")
        return False
    
    def __get_pitch_spec_path(self, file):
        path = self.pitch_dir / f"pitch_spec_{file}.npy"
        if path.exists():
            return path
        # print(f"Unable to load: {path}")
        return False
    
    def __get_pitch_contour_path(self, file):
        path = self.pitch_dir / f"pitch_contour_{file}.npy"
        if path.exists():
            return path
        # print(f"Unable to load: {path}")
        return False

    def __get_pitch_contour_mean_path(self, file):
        path = self.pitch_dir / f"pitch_contour_mean_{file}.npy"
        if path.exists():
            return path
        # print(f"Unable to load: {path}")
        return False

    def __get_pitch_contour_std_path(self, file):
        path = self.pitch_dir / f"pitch_contour_std_{file}.npy"
        if path.exists():
            return path
        # print(f"Unable to load: {path}")
        return False
    
    
    def __files_list(self,):
        # Mel-Spectrogram Files
        mel_files = [str(x.name) for x in self.mel_dir.glob("*.npy")]
        mel_files = [x.split('.')[0].replace("mel_", "") for x in mel_files]

        # Energy Files
        energy_files = [str(x.name) for x in self.energy_dir.glob("*.npy")]
        energy_files = [x.split('.')[0].replace("energy_", "") for x in energy_files]

        # Duration Files
        duration_files = [str(x.name) for x in self.duration_dir.glob("*.npy")]
        duration_files = [x.split(".")[0].replace("duration_", "") for x in duration_files]

        # Pitch Spectrogram Files
        pitch_spec_files = [str(x.name) for x in self.pitch_dir.glob("pitch_spec_*.npy")]
        pitch_spec_files = [x.split(".")[0].replace("pitch_spec_", "") for x in pitch_spec_files]
        
        # Pitch Contour Files
        pitch_contour_files = [str(x.name) for x in self.pitch_dir.glob("pitch_contour_*.npy")]
        pitch_contour_files = [x.split(".")[0].replace("pitch_contour_", "") for x in pitch_contour_files]
        
        # Pitch Contour Mean Files
        pitch_contour_mean_files = [str(x.name) for x in self.pitch_dir.glob("pitch_contour_mean_*.npy")]
        pitch_contour_mean_files = [x.split(".")[0].replace("pitch_contour_mean_", "") for x in pitch_contour_mean_files]
        
        # Pitch Contour STD Files
        pitch_contour_std_files = [str(x.name) for x in self.pitch_dir.glob("pitch_contour_std_*.npy")]
        pitch_contour_std_files = [x.split(".")[0].replace("pitch_contour_std_", "") for x in pitch_contour_std_files]

        # Text Files
        text_files = [str(x.name) for x in self.transcript_dir.glob("in_corpus/*.txt")]
        text_files = [x.split(".")[0] for x in text_files]
        # print(f"text_files: {text_files}")
        files_for_all_feauters = set(mel_files).intersection(
            set(energy_files)).intersection(
                set(duration_files)).intersection(
                    set(pitch_spec_files).intersection(
                        set(pitch_contour_files).intersection(
                            set(pitch_contour_mean_files).intersection(
                                set(pitch_contour_std_files).intersection(
                                    set(text_files)
                                )
        ))))
        return list(files_for_all_feauters)

    def __load_text(self, path):
        with open(path, 'r') as f:
            text = f.readlines()
        return text
    

    def __tokeinze_text(self, raw_text):
        return torch.tensor(self.tokenizer.tokenize(raw_text), dtype=torch.int32)
    
    def __getitem__(self, idx):
        feature_fname = self.feature_path[idx][0]
        text = self.__load_text(self.feature_path[idx][1])[0]
        text_token_ids = self.__tokeinze_text(text)
        # text_token_ids = [self.__tokeinze_text(x) for x in self.feature_path[idx][2]]
        features = self.feature_path[idx][2:]
        # Feature sequence
        #   1. filename
        #   2. mel-spectrogram
        #   3. energy
        #   4. duration
        #   5. pitch-contour
        #   6. pitch-contour mean
        #   7. pitch-contour std
        #   8. pitch spectrogram
        feature_arr = [np.load(x) for x in features]
        feature_tensors = [torch.from_numpy(arr).float() for arr in feature_arr]

        # Validate mel-spectrogram values
        mel_spec = torch.transpose(feature_tensors[0][0], 0, 1)
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            print(f"Warning: Invalid values in mel-spectrogram for {feature_fname}")
            mel_spec = torch.clamp(mel_spec, min=0.0, max=10.0)  # Clamp to reasonable range

        return {
            'filename': feature_fname,
            'raw_text': text,
            'token_ids': text_token_ids,
            'mel_spectrogram': mel_spec, # (time_frame, mel-bins)
            'energy': feature_tensors[1],
            'duration': feature_tensors[2],
            'pitch_contour': feature_tensors[3],
            'pitch_contour_mean': feature_tensors[4],
            'pitch_contour_std': feature_tensors[5],
            'pitch_spectrogram': torch.transpose(feature_tensors[6], 0, 1) # (time_frame, pitch-bins)
        }
    
    def __len__(self, ):
        return len(self.feature_path)
    

    def collate_function(self, batch):
        """
        
        """
        # Sorted Index
        token_length = [item['token_ids'].shape[0] for item in batch]
        sorted_idx = np.argsort(token_length)[::-1]  # Sort in descending order
        # sorted_batch = [batch[i] for i in sorted_idx]
        # print(f"Token Length: {token_length}")
        # print(f"Sorted IDX: {sorted_idx}")

        tail_idx = sorted_idx[len(sorted_idx) - (len(sorted_idx) % self.batch_size):]
        idx_arr = sorted_idx[:len(sorted_idx) - (len(sorted_idx) % self.batch_size)]
        idx_arr = np.reshape(idx_arr, (-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail_idx) > 0:
            idx_arr += [tail_idx]
        
        # print(f"idx_arr: {idx_arr}")

        text = [item['raw_text'] for item in batch]
        token_ids = [item['token_ids'] for item in batch]
        mel_spectrogram = [item['mel_spectrogram'] for item in batch]
        energy = [item['energy'] for item in batch]
        duration = [item['duration'] for item in batch]
        pitch_contour = [item['pitch_contour'] for item in batch]
        pitch_contour_mean = [item['pitch_contour_mean'] for item in batch]
        pitch_contour_std = [item['pitch_contour_std'] for item in batch]
        pitch_spectrogram = [item['pitch_spectrogram'] for item in batch]

        output = list()
        for idx in idx_arr:
            group_text = [text[i] for i in idx]
            
            group_token_ids = [token_ids[i] for i in idx]
            group_token_length = [item.shape[0] for item in group_token_ids]
            group_max_token_length = max(group_token_length)
            group_padded_token_ids = [
                F.pad(
                    item,
                    (0, group_max_token_length - item.shape[0]),
                    "constant",
                    0
                ) for item in group_token_ids
            ]
            group_padded_token_ids = torch.stack(group_padded_token_ids, dim=0)


            # Get max length for each features
            # Mel-Spectrogram, shpae: (time_frame, mel-bins)
            group_mel_spec = [mel_spectrogram[i] for i in idx]
            group_mel_length = [item.shape[0] for item in group_mel_spec]
            group_max_mel_length = max(group_mel_length)
            group_padded_mel_spectrogram = [
                F.pad(
                    item, 
                    (0, 0, 0, group_max_mel_length - item.shape[0]), 
                    "constant", 
                    0
                ) for item in group_mel_spec
            ]
            group_padded_mel_spectrogram = torch.stack(group_padded_mel_spectrogram, dim=0)


            # Energy, shape: (time_frame, )
            group_energy = [energy[i] for i in idx]
            group_energy_length = [item.shape[0] for item in group_energy]
            group_energy_max_length = max(group_energy_length)
            group_padded_energy = [
                F.pad(
                    item,
                    (0, group_energy_max_length - item.shape[0]),
                    "constant",
                    0
                ) for item in group_energy
            ]
            group_padded_energy = torch.stack(group_padded_energy, dim=0)
            

            # Duration, shape: (token_count, )
            group_duration = [duration[i] for i in idx]
            group_duration_length = [item.shape[0] for item in group_duration]
            group_duration_max_length = max(group_duration_length)
            group_padded_duration = [
                F.pad(
                    item,
                    (0, group_duration_max_length - item.shape[0]),
                    "constant",
                    0
                ) for item in group_duration
            ]
            group_padded_duration = torch.stack(group_padded_duration, dim=0)

            # Pitch Spectrogram, shape: (time_frame, pitch-bins)
            group_pitch_spectrogram = [pitch_spectrogram[i] for i in idx]
            group_pitch_spectrogram_length = [item.shape[0] for item in group_pitch_spectrogram]
            group_pitch_spectrogram_max_length = max(group_pitch_spectrogram_length)
            group_padded_pitch_spectrogram = [
                F.pad(
                    item,
                    (0, 0, 0, group_pitch_spectrogram_max_length - item.shape[0]),
                    "constant",
                    0
                ) for item in group_pitch_spectrogram
            ]
            group_padded_pitch_spectrogram = torch.stack(group_padded_pitch_spectrogram, dim=0)

            # Pitch Contour, shape: (time_frame, )
            group_pitch_contour = [pitch_contour[i] for i in idx]
            group_pitch_contour_length = [item.shape[0] for item in group_pitch_contour]
            group_pitch_contour_max_length = max(group_pitch_contour_length)
            group_padded_pitch_contour = [
                F.pad(
                    item,
                    (0, group_pitch_contour_max_length - item.shape[0]),
                    "constant",
                    0
                ) for item in group_pitch_contour
            ]
            group_padded_pitch_contour = torch.stack(group_padded_pitch_contour, dim=0)
            
            group_pitch_contour_mean = torch.tensor([pitch_contour_mean[i] for i in idx])
            group_pitch_contour_std = torch.tensor([pitch_contour_std[i] for i in idx])
            output.append({
                'filename': [batch[i]['filename'] for i in idx],
                'raw_text': group_text,
                'token_ids': group_padded_token_ids,
                'token_length': group_token_length,
                'token_max_length': group_max_token_length,
                'mel_spectrogram': group_padded_mel_spectrogram,
                'mel_length': group_mel_length,
                'mel_max_length': group_max_mel_length,
                'energy': group_padded_energy,
                'energy_length': group_energy_length,
                'energy_max_length': group_energy_max_length,
                'duration': group_padded_duration,
                'duration_length': group_duration_length,
                'duration_max_length': group_duration_max_length,
                'pitch_contour': group_padded_pitch_contour,
                'pitch_contour_length': group_pitch_contour_length,
                'pitch_contour_max_length': group_pitch_contour_max_length,
                'pitch_contour_mean': group_pitch_contour_mean,
                'pitch_contour_std': group_pitch_contour_std,
                'pitch_spectrogram': group_padded_pitch_spectrogram,
                'pitch_spectrogram_length': group_pitch_spectrogram_length,
                'pitch_spectrogram_max_length': group_pitch_spectrogram_max_length
            })

        return output


if __name__=="__main__":
    batch_size = 16
    feature_dataset = SpeechFeatureDataset(
        feature_dir=FEATURE_DIR,
        batch_size=batch_size
    )

    # print(feature_dataset[0])
    print(f"Length of dataset: {len(feature_dataset)}")
    feature = feature_dataset[0]
    print(feature.keys())
    fname = feature['filename']
    mel_spectrogram = feature['mel_spectrogram']
    energy = feature['energy']
    duration = feature['duration']
    pitch_contour = feature['pitch_contour']
    pitch_contour_mean = feature['pitch_contour_mean']
    pitch_contour_std = feature['pitch_contour_std']
    pitch_spectrogram = feature['pitch_spectrogram']
    print(f"fname: {fname}")
    print(f"mel shape: {mel_spectrogram.shape}")
    print(f"energy shape: {energy.shape}")
    print(f"duration shape: {duration.shape} & Duration sum: {sum(duration)}")
    print(f"pitch contour shape: {pitch_contour.shape}")
    print(f"pitch contour mean: {pitch_contour_mean}")
    print(f"pitch contour std: {pitch_contour_std}")
    print(f"pitch spectrogram shape: {pitch_spectrogram.shape}")


  
    shuffle=True
    feature_dataloader = DataLoader(
        feature_dataset,
        batch_size=batch_size*4,
        shuffle=shuffle,
        collate_fn=feature_dataset.collate_function
    )

    # for batch in feature_dataloader:
    #     print(batch)
    #     break        