import os
import zipfile
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from datasets.hyper_object import HyperObjectDataset

from baselines.raw2hsi import Raw2HSI
from config.track1_cfg_default import TrainerCfg

# TODO: create unified submission script
# TODO: allow models other than baseline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

TARGET_IDS = {
    "Category-1_a_0007",
    "Category-2_a_0009",
    "Category-3_a_0035",
    "Category-4_a_0018",
}

data_dir = 'data/track1'
model_path = 'runs/track1/mosaic2hsi_baseline/model_best.tar' # change to model path
submission_files_dir = 'runs/submission_files'
submission_zip_path = f'{submission_files_dir}/submission.zip'

BATCH_SIZE = 1

def create_submission():
    """
    Generates predictions and packages them for Kaggle submission.
    """
    processed_ids = []
    os.makedirs(submission_files_dir, exist_ok=True)
    print(f"Individual prediction files will be saved in: '{submission_files_dir}'")

    submission_dataset = HyperObjectDataset(
        data_root=f'{data_dir}',
        track=1,
        train=False,
        submission=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=submission_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True if device == 'cuda' else False
    )
    print(f"DataLoader created for {len(submission_dataset)} samples.")

    print(f"Loading checkpoint from: {model_path}")

    # make sure settings match training settings! change below if needed
    cfg = TrainerCfg()
    model = Raw2HSI(base_ch=cfg.base_ch, n_blocks=cfg.n_blocks, out_bands=cfg.out_bands)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print("\nGenerating predictions...")
    for data in tqdm(test_loader, desc="Generating predictions"):
        x = data['input'].float().to(device)
        sample_id = data['id'][0]
        processed_ids.append(sample_id)

        with torch.no_grad():
            pred_hr_tensor = model(x)

        pred_np = pred_hr_tensor.squeeze(0).cpu().numpy()
        pred_hwc = np.transpose(pred_np, (1, 2, 0))
        
        pred_hwc_clipped = np.clip(pred_hwc, 0.0, 1.0)
        
        output_npz_path = os.path.join(submission_files_dir, f"{sample_id}.npz")
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
