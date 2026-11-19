# ENIGMA

An EEG-to-Image decoding model that reconstructs visual images from EEG brain signals using spatiotemporal CNNs and diffusion models.

## Installation

```bash
conda env create -f environment.yml
conda activate enigma
```

## Check input file layout

Before starting training, check all configured input paths without loading EEG,
images or feature tensors, or importing the model dependencies:

```bash
python check_data.py --config configs/Alljoined-1.6M.json
# Check one subject instead of every configured subject:
python check_data.py --config configs/Alljoined-1.6M.json --subjects sub-01
```

Relative paths are resolved from the current directory, as in the dataset loader.
Use `--root /path/to/project` to check another root. The JSON report lists missing
metadata, EEG and feature files together; shared feature paths are checked once.
Exit codes are 0 for all expected files present, 1 for missing files and 2 for
invalid configuration or arguments.

This checks file availability only. It does not validate data contents, subject
or trial alignment, feature keys, embedding provenance or decoding performance.
It never downloads files, substitutes metadata filenames or generates embeddings.
The dataset and pretrained models have their own access and license conditions.

The checker's synthetic tests require only Python's standard library:

```bash
python -m unittest -v test_check_data
```

## Usage

### Image Reconstruction

Train and run the full reconstruction pipeline:

```bash
# Edit subject IDs and config in reconstruction.sh
./reconstruction.sh
```

Or run manually:
```bash
# Train model
python train.py --model_name ENIGMA_sub8 --subj_ids 8 --config_name things_eeg2

# Generate reconstructions  
python recon_inference.py --model_name ENIGMA_sub8 --subj_ids 8 --config_name things_eeg2

# Evaluate quality
python evaluate_recons.py --model_name ENIGMA_sub8 --subj_ids 8 --config_name things_eeg2
```

### Image Retrieval

Train and test retrieval-only model:

```bash
# Edit subject IDs and config in retrieval.sh
./retrieval.sh
```

Or run manually:
```bash
# Train retrieval model
python train.py --model_name ENIGMA_retrieval_sub8 --subj_ids 8 --config_name things_eeg2 --retrieval_only --num_epochs 50

# Test retrieval
python recon_inference.py --model_name ENIGMA_retrieval_sub8 --subj_ids 8 --config_name things_eeg2 --retrieval_only
```

## Configuration

Available dataset configs in `configs/`:
- `things_eeg2.json` - THINGS-EEG2 dataset (10 subjects, 63 channels)
- `Alljoined-1.6M.json` - Alljoined-1.6M dataset (20 subjects, 32 channels)  
- `THINGS-EEG2+Alljoined-1.6M.json` - Merged dataset (30 subjects, 32 channels)

## Output

- Model checkpoints/evaluation metrics: `train_logs/{model_name}/`
- Reconstructed images: `output/{model_name}/`

## License

See LICENSE file for details.
