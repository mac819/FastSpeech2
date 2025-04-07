import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


PACKAGE_PATH = Path(__file__).parent.parent
DATA_PATH = PACKAGE_PATH.parent / 'data'
FEATURE_DIR = DATA_PATH / 'processed' / 'lj_speech_feature'


# Model gets trained on following ground truth
#   1. Mel-Spectrogram
#   2. Pitch Spectrogram
#   3. Energy
#   4. Duration

class SpeechFeatureDataset(Dataset):

    def __init__(self, feature_dir):
        super().__init__()
        self.energy_dir = Path(feature_dir) / 'energy'
        self.duration_dir = Path(feature_dir) / 'duration'
        self.mel_dir = Path(feature_dir) / 'mel_spec'
        self.pitch_dir = Path(feature_dir) / 'pitch'

        self.files = self.__files_list()
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
            # print([mel_spec_path, energy_path, 
            #        duration_path, pitch_contour_path,
            #        pitch_contour_mean_path, pitch_contour_std_path,
            #        pitch_spec_path])
            if all([mel_spec_path, energy_path, 
                   duration_path, pitch_contour_path,
                   pitch_contour_mean_path, pitch_contour_std_path,
                   pitch_spec_path]):
                self.feature_path.append((
                    file,
                    mel_spec_path,
                    energy_path,
                    duration_path,
                    pitch_contour_path,
                    pitch_contour_mean_path,
                    pitch_contour_std_path,
                    pitch_spec_path
                ))


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

        files_for_all_feauters = set(mel_files).union(
            set(energy_files)).intersection(
                set(duration_files)).intersection(
                    set(pitch_spec_files).intersection(
                        set(pitch_contour_files).intersection(
                            set(pitch_contour_mean_files).intersection(
                                set(pitch_contour_std_files
        )))))
        return list(files_for_all_feauters)

    
    def __getitem__(self, idx):
        feature_fname = self.feature_path[idx][0]
        features = self.feature_path[idx][1:]
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

        return {
            'filename': feature_fname,
            'mel_spectrogram': torch.transpose(feature_tensors[0][0], 0, 1), # (time_frame, mel-bins)
            'energy': feature_tensors[1],
            'duration': feature_tensors[2],
            'pitch_contour': feature_tensors[3],
            'pitch_contour_mean': feature_tensors[4],
            'pitch_contour_std': feature_tensors[5],
            'pitch_spectrogram': torch.transpose(feature_tensors[6], 0, 1) # (time_frame, pitch-bins)
        }
    
    def __len__(self, ):
        return len(self.feature_path)
    

def collate_function(batch):
    """
    
    """
    # Get max length for each features
    # Mel-Spectrogram, shpae: (time_frame, mel-bins)
    mel_max_length = max([item['mel_spectrogram'].shape[0] for item in batch])
    padded_mel_spectrogram = [
        F.pad(
            item['mel_spectrogram'], 
            (0, 0, 0, mel_max_length - item['mel_spectrogram'].shape[0]), 
            "constant", 
            0
        ) for item in batch
    ]
    padded_mel_spectrogram = torch.stack(padded_mel_spectrogram, dim=0)

    # Energy, shape: (time_frame, )
    energy_max_length = max([item['energy'].shape[0] for item in batch])
    padded_energy = [
        F.pad(
            item['energy'],
            (0, energy_max_length - item['energy'].shape[0]),
            "constant",
            0
        ) for item in batch
    ]
    padded_energy = torch.stack(padded_energy, dim=0)

    # Duration, shape: (token_count, )
    duration_max_length = max([item['duration'].shape[0] for item in batch])
    padded_duration = [
        F.pad(
            item['duration'],
            (0, duration_max_length - item['duration'].shape[0]),
            "constant",
            0
        ) for item in batch
    ]
    padded_duration = torch.stack(padded_duration, dim=0)

    # Pitch Spectrogram, shape: (time_frame, pitch-bins)
    pitch_spectrogram_max_length = max([item['pitch_spectrogram'].shape[0] for item in batch])
    padded_pitch_spectrogram = [
        F.pad(
            item['pitch_spectrogram'],
            (0, 0, 0, pitch_spectrogram_max_length - item['pitch_spectrogram'].shape[0]),
            "constant",
            0
        ) for item in batch
    ]
    padded_pitch_spectrogram = torch.stack(padded_pitch_spectrogram, dim=0)

    # Pitch Contour, shape: (time_frame, )
    pitch_contour_max_length = max([item['pitch_contour'].shape[0] for item in batch])
    padded_pitch_contour = [
        F.pad(
            item['pitch_contour'],
            (0, pitch_contour_max_length - item['pitch_contour'].shape[0]),
            "constant",
            0
        ) for item in batch
    ]
    padded_pitch_contour = torch.stack(padded_pitch_contour, dim=0)

    filenames = [item['filename'] for item in batch]
    print("Max sizes of features in batch:")
    print(f"Mel Spectrogram max length: {mel_max_length}")
    print(f"Energy max length: {energy_max_length}")
    print(f"Duration max length: {duration_max_length}")
    print(f"Pitch Spectrogram max length: {pitch_spectrogram_max_length}")
    print(f"Pitch contour max length: {pitch_contour_max_length}")

    return {
        'filename': filenames,
        'mel_spectrogram': padded_mel_spectrogram,
        'energy': padded_energy,
        'duration': padded_duration,
        'pitch_contour': padded_pitch_contour,
        'pitch_contour_mean': [item['pitch_contour_mean'] for item in batch],
        'pitch_contour_std': [item['pitch_contour_std'] for item in batch],
        'pitch_spectrogram': padded_pitch_spectrogram
    }


if __name__=="__main__":
    feature_dataset = SpeechFeatureDataset(
        feature_dir=FEATURE_DIR
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


    batch_size = 8
    shuffle=True
    feature_dataloader = DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_function
    )

    for batch in feature_dataloader:
        print(batch)
        break        