# Training Toolkit Documentation

This directory contains offline scripts for training and fine-tuning models used in the pipeline.
These scripts are NOT used during inference and have their own dependencies.

## Structure

* `augment.py`: Audio augmentation utilities (pitch shift, noise, etc.).
* `dataset.py`: PyTorch Dataset wrappers for MAESTRO/MusicNet.
* `train_f0.py`: Script to train/finetune pitch trackers (e.g. CREPE).
* `train_onsets_frames.py`: Script to train transcription models.

## Usage

To run training, ensure you have full training dependencies installed (torch, librosa, tensorboard, etc.).

```bash
python -m backend.training.train_f0 --data_dir /path/to/data
```
