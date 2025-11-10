import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd

from datasets.io import read_rgb_image
from baselines import mstpp_up

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

TARGET_IDS = {
    "Category-1_a_0007",
    "Category-2_a_0009",
    "Category-3_a_0035",
    "Category-4_a_0018",
}

data_dir = 'datasets/2026-Hyper-Object-Data'  # raiz dos dados
model_path = 'runs/track2/saved-models/exp-MST_Plus_Plus_Up-CIE/model.pt'
submission_files_dir = 'submission_files'
submission_zip_path = 'submission.zip'

MODEL_NAME = 'MST_Plus_Plus_Up'
UPSCALE_FACTOR = 2
BATCH_SIZE = 1


class HyperObjectPrivateRGBDataset(Dataset):
    """
    Dataset simples para o track2/test_private:
      - Lê apenas as imagens RGB (rgb_2)
      - Não usa HSI (sem ground truth no private)
    """
    def __init__(self, root, target_ids=None):
        self.root = Path(root)
        self.rgb_dir = self.root / "track2" / "test-private"

        paths = sorted(list(self.rgb_dir.glob("*.png")) + list(self.rgb_dir.glob("*.jpg")))
        if not paths:
            raise RuntimeError(f"Nenhuma imagem encontrada em {self.rgb_dir}")

        if target_ids is not None:
            target_ids = set(target_ids)
            paths = [p for p in paths if p.stem in target_ids]
            if not paths:
                raise RuntimeError(f"Nenhum arquivo em {self.rgb_dir} corresponde aos IDs em TARGET_IDS")

        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        rgb = read_rgb_image(p)  # (H,W,3) float32 [0,1]
        rgb_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))  # C,H,W
        return {
            "input": rgb_t,
            "id": p.stem
        }


def create_submission():
    """
    Generates predictions and packages them for Kaggle submission.
    """
    processed_ids = []
    os.makedirs(submission_files_dir, exist_ok=True)
    print(f"Individual prediction files will be saved in: '{submission_files_dir}'")

    # Dataset para o test_private
    full_ds_test = HyperObjectPrivateRGBDataset(
        root=data_dir,
        target_ids=TARGET_IDS if TARGET_IDS is not None else None,
    )

    print(f"Dataset de submissão com {len(full_ds_test)} amostras.")

    test_loader = torch.utils.data.DataLoader(
        dataset=full_ds_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True if device == 'cuda' else False
    )
    print(f"DataLoader created for {len(full_ds_test)} samples.")

    print(f"Loading checkpoint from: {model_path}")
    model = mstpp_up.MST_Plus_Plus_LateUpsample(
        in_channels=3, out_channels=61, n_feat=61, stage=3, upscale_factor=UPSCALE_FACTOR
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    model.return_hr = True
    print(f"Model '{MODEL_NAME}' loaded and set to HR evaluation mode.")

    print("\nGenerating predictions...")
    for data in tqdm(test_loader, desc="Generating predictions"):
        x = data['input'].float().to(device)
        sample_id = data['id'][0]  # batch_size=1, então pegamos o primeiro
        processed_ids.append(sample_id)

        with torch.no_grad():
            pred_hr_tensor = model(x)

        # (1, C, H, W) -> (C, H, W)
        pred_np = pred_hr_tensor.squeeze(0).cpu().numpy()
        # (C, H, W) -> (H, W, C)
        pred_hwc = np.transpose(pred_np, (1, 2, 0))

        pred_hwc_clipped = np.clip(pred_hwc, 0.0, 1.0)

        output_npz_path = os.path.join(submission_files_dir, f"{sample_id}.npz")
        np.savez_compressed(output_npz_path, cube=pred_hwc_clipped)

    print(f"\nAll {len(processed_ids)} predictions saved.")

    print("\nCreating submission.csv...")
    submission_df = pd.DataFrame({'id': processed_ids, 'prediction': 0})
    csv_path = os.path.join(submission_files_dir, 'submission.csv')
    submission_df.to_csv(csv_path, index=False)
    print(f"submission.csv created with {len(submission_df)} entries.")

    print(f"Creating submission zip file at: '{submission_zip_path}'")
    with zipfile.ZipFile(submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # CSV
        zf.write(csv_path, arcname='submission.csv')

        # Arquivos .npz
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
