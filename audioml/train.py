import sys
import yaml
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

print(f"Appended path: {str(Path(__file__).parent.parent)}")
sys.path.append(str(Path(__file__).parent.parent))


from audioml.fastspeech.model import Text2Mel
from audioml.dataset.feature_dataset import SpeechFeatureDataset
from audioml.processing.text_speech_alignment import TTSTokenizer


MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
CURR_DIR = Path(__file__).parent

log_file_path = CURR_DIR / "training_log.txt"
mel_image_dir = CURR_DIR / "mel_images"

# Ensure the log file and mel image directory exist
log_file_path.touch(exist_ok=True)
mel_image_dir.mkdir(parents=True, exist_ok=True)

# Add header to log file if it's empty
if log_file_path.stat().st_size == 0:
    with open(log_file_path, 'w') as f:
        header = (
            "step,total_loss,mel_loss,duration_loss,pitch_spec_loss,pitch_mean_loss,"
            "pitch_std_loss,energy_loss,"
            "pred_mel_min,pred_mel_max,pred_mel_mean,"
            "target_mel_min,target_mel_max,target_mel_mean\n"
        )
        f.write(header)



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

    def state_dict(self):
        return {
          "optimizer_state_dict": self._optimizer.state_dict(),
          "current_step": self.current_step
        }

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])
        # Overwrite the step from the constructor with the one from the checkpoint
        self.current_step = state_dict['current_step']

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
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
            checkpoint = torch.load(artifact_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Checkpoint {artifact_path} not found.")
        model = Text2Mel(cfg=config).to(device)
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
        model = Text2Mel(cfg=config).to(device)
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
                input_ids, src_mask, duration = tokens['input_ids'].to(device), tokens['mask_ids'].to(device), batch['duration'].to(device)
                model_outputs = model(
                    input_ids, 
                    src_mask, 
                    alpha=config['variance_adaptor']['alpha'], 
                    train=True, 
                    gt_duration=duration
                )

                # Label/Targets
                log_duration_target = torch.log(torch.clamp(duration, min=1.0)).masked_fill(src_mask.bool(), 0)
                # Mel-Spectrogram Target (already log-transformed from preprocessing)
                mel_spec_target = batch['mel_spectrogram'].to(device)
                # Pitch Target
                pitch_spec_target = batch['pitch_spectrogram'].to(device)
                # pitch_contour_target = batch['pitch_contour']
                pitch_mean_target = batch['pitch_contour_mean'].to(device)
                pitch_std_target = batch['pitch_contour_std'].to(device)
                # Energy Target
                energy_target = batch['energy'].to(device)


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


                # log_duration_target.requires_grad = False
                # mel_spec_target.requires_grad = False
                # pitch_spec_target.requires_grad = False
                # pitch_mean_target.requires_grad = False
                # pitch_std_target.requires_grad = False
                # energy_target.requires_grad = False

                # Loss Calculation
                # Mel-Spectrogram loss (higher weight as it's the main output)
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
                
                # Weighted Total Loss (mel loss should have higher weight)
                total_loss = (
                    45.0 * mel_loss +  # Higher weight for mel-spectrogram
                    1.0 * duration_loss + 
                    1.0 * pitch_spectrogram_loss +
                    1.0 * pitch_mean_loss + 
                    1.0 * pitch_std_loss + 
                    1.0 * energy_loss
                )
                total_loss = total_loss / gradient_accumulation_steps
                total_loss.backward()
                if step % gradient_accumulation_steps == 0:
                    # Clipping gradients to avoid exploding gradients
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    # Backward Pass
                    scheduled_optim.step_and_update_lr()
                    scheduled_optim.zero_grad()

                step = scheduled_optim.current_step

                if step % train_config['save_step'] == 0:
                    # Save the model state
                    model_save_path = MODEL_DIR / f"model_step_{step}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': scheduled_optim.state_dict(),
                        'step': step,
                        'config': config
                    }, model_save_path)
                    print(f"Model saved at {model_save_path}")

                if step % train_config['log_step'] == 0:

                    loss_items = {
                        "Total": total_loss.item(), "Mel": mel_loss.item(), "Duration": duration_loss.item(),
                        "PitchSpec": pitch_spectrogram_loss.item(), "PitchMean": pitch_mean_loss.item(),
                        "PitchSTD": pitch_std_loss.item(), "Energy": energy_loss.item()
                    }
                    log_msg = f"Step: {step}, " + ", ".join([f"{k} Loss: {v:.4f}" for k, v in loss_items.items()])
                    print(log_msg)
                    
                    # Add debugging information for model outputs
                    pred_mel_min = pred_mel_spec.min().item()
                    pred_mel_max = pred_mel_spec.max().item()
                    pred_mel_mean = pred_mel_spec.mean().item()
                    target_mel_min = mel_spec_target.min().item()
                    target_mel_max = mel_spec_target.max().item()
                    target_mel_mean = mel_spec_target.mean().item()
                    
                    print(f"Predicted mel-spec range: [{pred_mel_min:.4f}, {pred_mel_max:.4f}]")
                    print(f"Target mel-spec range: [{target_mel_min:.4f}, {target_mel_max:.4f}]")
                    print(f"Predicted mel-spec mean: {pred_mel_mean:.4f}")
                    print(f"Target mel-spec mean: {target_mel_mean:.4f}")
                    
                    # Log to Tensorboard
                    for k, v in loss_items.items():
                        logger.add_scalar(f'Loss/{k}', v, step)

                    # Log Images <<---- Adding Mel-Spectrogram to tensorboard
                    # For Visualization, we'll just pick the first item from the batch
                    # The Shape of pred_mel_spec is (B, H, W), so we take the first element.
                    predicted_spec = pred_mel_spec[0] # Shape: (Mel_Bins, Time)
                    target_spec = mel_spec_target[0] # Shape: (Mel_Bins, Time)

                    colormap = cm.get_cmap('virdis')
                    # Add a channel dimension to make it (1, H, W)
                    predicted_spec = predicted_spec.unsqueeze(0)
                    target_spec = target_spec.unsqueeze(0)

                    predicted_spec = colormap(predicted_spec.detach().cpu().numpy())[..., :3]
                    target_spec = colormap(target_spec.detach().cpu().numpy())[..., :3]

                    logger.add_image(
                      "Mel-SPectrogram/Predicted",
                      predicted_spec,
                      global_step=step,
                      dataformats="NHWC"
                    )
                    logger.add_image(
                      "Mel-SPectrogram/Target",
                      target_spec,
                      global_step=step,
                      dataformats="NHWC"
                    )

                    # Enhanced: Log to text file with mel-spectrogram debugging info
                    with open(log_file_path, 'a') as f:
                        log_line = (
                            f"{step},{loss_items['Total']:.4f},{loss_items['Mel']:.4f},"
                            f"{loss_items['Duration']:.4f},{loss_items['PitchSpec']:.4f},"
                            f"{loss_items['PitchMean']:.4f},{loss_items['PitchSTD']:.4f},"
                            f"{loss_items['Energy']:.4f},"
                            f"{pred_mel_min:.4f},{pred_mel_max:.4f},{pred_mel_mean:.4f},"
                            f"{target_mel_min:.4f},{target_mel_max:.4f},{target_mel_mean:.4f}\n"
                        )
                        f.write(log_line)

                    # New: Save mel-spectrogram images
                    predicted_spec_np = pred_mel_spec[0].detach().cpu().numpy()
                    target_spec_np = mel_spec_target[0].detach().cpu().numpy()
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    fig.suptitle(f'Mel-Spectrograms at Step {step}')
                    im1 = ax1.imshow(predicted_spec_np, aspect='auto', origin='lower', cmap='viridis')
                    ax1.set_title("Predicted"); ax1.set_ylabel("Mel Bins"); fig.colorbar(im1, ax=ax1)
                    im2 = ax2.imshow(target_spec_np, aspect='auto', origin='lower', cmap='viridis')
                    ax2.set_title("Target"); ax2.set_xlabel("Frames"); ax2.set_ylabel("Mel Bins"); fig.colorbar(im2, ax=ax2)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    image_save_path = mel_image_dir / f"mel_step_{step}.png"
                    plt.savefig(image_save_path)
                    plt.close(fig)
                
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