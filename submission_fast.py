import yaml
import os
import zipfile
import numpy as np
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from datasets.partial import PartialDataset

from models import setup_model

from typing import Dict, Any

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", type=int, required=True, help="track to run")
parser.add_argument("-d", "--data_dir", type=str, default="./data", required=False, help="path to dataset directory")
parser.add_argument("--model", type=str, required=True, help="path to model to submit")
parser.add_argument("-c", "--config", type=str, required=False, help="path of config file to use (defaults to baselines for each track)")

parser.add_argument("-o", "--out_dir", type=str, default="submission_files", required=False, help="directory to output submission files")

args = parser.parse_args()

model_path = args.model

if args.config is None:
    config_path = f"{os.path.dirname(model_path)}/config.yaml"
    print(f"No config given, assuming config is at {config_path}")
else:
    config_path = args.config

config: Dict[str, Any]
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if "model" not in config:
    raise ValueError("No model settings found.")
elif "model_name" not in config["model"]:
    raise ValueError("No model name found.")

model_config = config["model"]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("WARNING: GPU not found, using CPU to run.")
    device = torch.device("cpu")

TARGET_IDS = {
    "Category-1_a_0007",
    "Category-2_a_0009",
    "Category-3_a_0035",
    "Category-4_a_0018",
}

data_dir = args.data_dir

config_name = os.path.splitext(os.path.basename(config_path))[0]
submission_files_dir = f"{args.out_dir}/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{config_name}"
submission_zip_path = f'{submission_files_dir}/submission.zip'

# use batch 1, it's 4 images anyway
BATCH_SIZE = 1

def create_submission():
    """
    Generates predictions and packages them for Kaggle submission.
    """
    processed_ids = []
    os.makedirs(submission_files_dir, exist_ok=True)
    print(f"Individual prediction files will be saved in: '{submission_files_dir}'")

    submission_dataset = PartialDataset(
        data_root=f'{data_dir}',
        track=args.track,
        dataset_type="test-private",
        img_type="mosaic" if args.track==1 else "rgb_2",
        old_mode=model_config.get("old_mode", False)
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=submission_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
    )
    print(f"DataLoader created for {len(submission_dataset)} samples.")

    print(f"Loading checkpoint from: {model_path}")

    # make sure settings match training settings! change below if needed
    model = setup_model(model_config)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print("\nGenerating predictions...")
    for img, img_id in tqdm(test_loader, desc="Generating predictions"):
        img = img.float().to(device)
        processed_ids.append(img_id[0]) # img_id is a tuple for some reason?

        with torch.no_grad():
            pred_hr_tensor = model(img)

        pred_np = pred_hr_tensor.squeeze(0).cpu().numpy()
        pred_hwc = np.transpose(pred_np, (1, 2, 0))
        
        pred_hwc_clipped = np.clip(pred_hwc, 0.0, 1.0)
        
        output_npz_path = os.path.join(submission_files_dir, f"{img_id}.npz")
        np.savez_compressed(output_npz_path, cube=pred_hwc_clipped)

    print(f"\nAll {len(submission_dataset)} predictions saved.")

    print("\nCreating submission.csv...")
    submission_df = pd.DataFrame({'id': processed_ids, 'prediction': 0})
    csv_path = os.path.join(submission_files_dir, 'submission.csv')
    submission_df.to_csv(csv_path, index=False)
    print(f"submission.csv created with {len(submission_df)} entries.")

    print(f"Creating submission zip file at: '{submission_zip_path}'")
    with zipfile.ZipFile(submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname='submission.csv')
        
        files_to_zip = [f"{sid}.npz" for sid in processed_ids]
        for filename in tqdm(files_to_zip, desc="Zipping .npz files"):
            file_path = os.path.join(submission_files_dir, filename)
            if os.path.exists(file_path):
                zf.write(file_path, arcname=filename)
            
    print("\n" + "="*50)
    print("Submission process complete!")
    print(f"File to submit to Kaggle: {submission_zip_path}")
    print("="*50)

if __name__ == '__main__':
    create_submission()
