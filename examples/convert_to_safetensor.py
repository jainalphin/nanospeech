#!/usr/bin/env python3
import re
from pathlib import Path
import torch
from safetensors.torch import save_file


def get_latest_checkpoint(folder: Path) -> Path | None:
    """
    Return the latest .pt checkpoint in the given folder based on the
    highest numeric suffix before the '.pt' extension.
    Example: 'model_123.pt' -> step 123.
    """
    if not folder.is_dir():
        return None

    pt_files = list(folder.glob("*.pt"))
    if not pt_files:
        return None

    def extract_step(file_path: Path) -> int:
        match = re.search(r"_(\d+)(?=\.pt$)", file_path.name)
        return int(match.group(1)) if match else -1

    return max(pt_files, key=extract_step)


def convert_to_safetensors(pt_path: Path, output_path: Path) -> None:
    """
    Convert a PyTorch '.pt' checkpoint file to safetensors format.

    Loads the checkpoint to CPU, extracts the state dict, moves all tensors to CPU,
    and saves them in safetensors format to 'output_path'.
    """
    checkpoint = torch.load(pt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    tensors = {k: v.cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path)


def process_model(name: str, input_dir: str, output_path: str) -> None:
    """
    Find the latest checkpoint in 'input_dir' and convert it to safetensors format.
    """
    latest_ckpt = get_latest_checkpoint(Path(input_dir))
    if latest_ckpt:
        print(f"Converting {name} checkpoint: {latest_ckpt}")
        convert_to_safetensors(latest_ckpt, Path(output_path))
        print(f"Saved safetensors to: {output_path}")
    else:
        print(f"No checkpoint found for {name}")


if __name__ == "__main__":
    process_model("nanospeech", "model/nanospeech", "models/model.safetensors")
    process_model("duration", "model/duration", "models/duration.safetensors")