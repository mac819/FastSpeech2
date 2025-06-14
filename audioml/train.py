import yaml
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audioml.fastspeech.model import Text2Mel
from audioml.dataset.feature_dataset import SpeechFeatureDataset
from audioml.processing.text_speech_alignment import TTSTokenizer


MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
CURR_DIR = Path(__file__).parent

class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, config, current_step):

        train_config = config['train']
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=float(train_config["optimizer"]["eps"]),
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_steps"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(config["encoder"]['fft']['d_out'], -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    
def load_config(path):
    """ Load configuration from a file """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(
        config_path, feature_dir, group_size
):
    config = load_config(config_path)
    train_config = config['train']
    gradient_accumulation_steps = train_config['gradient_accumulation_steps']
    grad_clip_thresh = train_config['gradient_clip_threshold']
    logger = SummaryWriter("runs/fastspeech2")

    feature_dataset = SpeechFeatureDataset(
        feature_dir=feature_dir,
        batch_size=train_config['batch_size'],
        sort=True,
        drop_last=False
    )

    feature_dataloader = DataLoader(
        feature_dataset,
        batch_size=group_size * train_config['batch_size'],
        shuffle=train_config['shuffle'],
        collate_fn=feature_dataset.collate_function
    )

    n_epoch = 0
    step = train_config['restore_step']

    # Inititalize Tokenizer
    tokenizer = TTSTokenizer()

    if train_config['restore_step'] > 0:
        print(f"Restoring model from step {train_config['restore_step']}...")
        # Load the model state here if needed
        artifact_path = MODEL_DIR / f"model_step_{train_config['restore_step']}.pth"
        if artifact_path.exists():
            checkpoint = torch.load(artifact_path)
        else:
            raise FileNotFoundError(f"Checkpoint {artifact_path} not found.")
        model = Text2Mel(cfg=config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        print(f"Model restored from {artifact_path}")
        # Restore the optimizer state
        scheduled_optim = ScheduledOptim(
            model=model,
            config=config,
            current_step=checkpoint['step']
        )
        scheduled_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Optimizer restored from step {checkpoint['step']}")
    else:
        print("Starting training from scratch...")
        # Initialize the model
        model = Text2Mel(cfg=config)
        model.train()

        print("Initializing optimizer...")
        # Initialize the optimizer
        # Load Optimizers
        scheduled_optim = ScheduledOptim(
            model=model,
            config=config,
            current_step=0
        )

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    # Prediction
    outer_loop = tqdm(total=train_config['steps'], desc="Training Steps", position=0)
    outer_loop.n = train_config['restore_step']
    outer_loop.update()

    while True:
        inner_loop = tqdm(total=len(feature_dataloader), desc=f"Epoch: {n_epoch + 1}", position=1)
        for batchs in feature_dataloader:
            for batch in batchs:

                tokens = tokenizer.batch_tokenize(batch['raw_text'])
                input_ids, src_mask, duration = tokens['input_ids'], tokens['mask_ids'], batch['duration']
                model_outputs = model(
                    input_ids, 
                    src_mask, 
                    alpha=config['variance_adaptor']['alpha'], 
                    train=True, 
                    gt_duration=duration
                )

                # Predictions
                pred_mel_spec = model_outputs['mel_spec']
                # mel_mask = model_outputs['mel_mask']
                pred_log_duration = model_outputs['log_duration']
                pitch_predictions = model_outputs['pitch']
                energy_predictions = model_outputs['energy']
                # Pitch Feature Prediction
                pred_pitch_spec = pitch_predictions['pitch_spectrogram']
                # pred_f0 = pitch_predictions['reconstructed_f0']
                pred_pitch_mean = pitch_predictions['pitch_mean']
                pred_pitch_std = pitch_predictions['pitch_std']
                # Energy Prediction
                pred_energy = energy_predictions['raw_energy']

                # Label/Targets
                log_duration_target = torch.log(torch.clamp(batch['duration'], min=1.0)).masked_fill(src_mask.bool(), 0)
                # Mel-Spectrogram Target
                mel_spec_target = batch['mel_spectrogram']
                # Pitch Target
                pitch_spec_target = torch.tensor(batch['pitch_spectrogram'])
                # pitch_contour_target = batch['pitch_contour']
                pitch_mean_target = torch.tensor(batch['pitch_contour_mean'])
                pitch_std_target = torch.tensor(batch['pitch_contour_std'])
                # Energy Target
                energy_target = batch['energy']

                # log_duration_target.requires_grad = False
                # mel_spec_target.requires_grad = False
                # pitch_spec_target.requires_grad = False
                # pitch_mean_target.requires_grad = False
                # pitch_std_target.requires_grad = False
                # energy_target.requires_grad = False

                # Loss Calculation
                # Mel-Spectrogram loss
                mel_loss = mae_loss(pred_mel_spec, mel_spec_target)
                # Pitch-Spectrogram loss
                pitch_spectrogram_loss = mse_loss(pred_pitch_spec, pitch_spec_target)
                # Pitch Mean loss
                pitch_mean_loss = mse_loss(pred_pitch_mean, pitch_mean_target)
                # Pitch STD loss
                pitch_std_loss = mse_loss(pred_pitch_std, pitch_std_target)
                # Energy loss
                energy_loss = mse_loss(pred_energy, energy_target)
                # Duration loss
                duration_loss = mse_loss(pred_log_duration, log_duration_target)
                # Total Loss
                total_loss = (
                    mel_loss + duration_loss + pitch_spectrogram_loss +
                    pitch_mean_loss + pitch_std_loss + energy_loss
                )
                total_loss = total_loss / gradient_accumulation_steps
                total_loss.backward()
                if step % gradient_accumulation_steps == 0:
                    # Clipping gradients to avoid exploding gradients
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    # Backward Pass
                    scheduled_optim.step_and_update_lr()
                    scheduled_optim.zero_grad()

                if step % train_config['save_step'] == 0:
                    # Save the model state
                    model_save_path = MODEL_DIR / f"model_step_{step}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': scheduled_optim._optimizer.state_dict(),
                        'step': step,
                        'config': config
                    }, model_save_path)
                    print(f"Model saved at {model_save_path}")

                if step % train_config['log_step'] == 0:
                    # Log the losses to Tensorboard
                    logger.add_scalar('Loss/Total', total_loss.item(), step)
                    logger.add_scalar('Loss/Mel', mel_loss.item(), step)
                    logger.add_scalar('Loss/Duration', duration_loss.item(), step)
                    logger.add_scalar('Loss/PitchSpectrogram', pitch_spectrogram_loss.item(), step)
                    logger.add_scalar('Loss/PitchMean', pitch_mean_loss.item(), step)
                    logger.add_scalar('Loss/PitchSTD', pitch_std_loss.item(), step)
                    logger.add_scalar('Loss/Energy', energy_loss.item(), step)
                    print(f""
                          f"Step: {step}, "
                          f"Total Loss: {total_loss.item():.4f}, "
                          f"Mel Loss: {mel_loss.item():.4f}, "
                          f"Duration Loss: {duration_loss.item():.4f}, "
                          f"Pitch Spectrogram Loss: {pitch_spectrogram_loss.item():.4f}, "
                          f"Pitch Mean Loss: {pitch_mean_loss.item():.4f}, "
                          f"Pitch STD Loss: {pitch_std_loss.item():.4f}, "
                          f"Energy Loss: {energy_loss.item():.4f}")
                # Increment step
                step += 1
                # Update progress bar
                outer_loop.update(1)
            inner_loop.update(1)
        n_epoch += 1


if __name__=="__main__":
    feature_dir = str(DATA_DIR / 'processed' / 'lj_speech_feature')
    config_path = str(CURR_DIR / 'config.yaml')
    train(
        config_path=config_path,
        feature_dir=feature_dir,
        group_size=1
    )
    print("Training completed successfully.")