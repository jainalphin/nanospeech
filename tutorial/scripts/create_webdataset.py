import io
import json
from pathlib import Path
import tarfile
from tqdm import tqdm

def process_sample(mp3_path: Path, txt_path: Path):
    text = txt_path.read_text(encoding="utf-8").strip()
    wav_path = mp3_path.with_suffix(".wav")
    try:
        wav_size = wav_path.stat().st_size
    except Exception as e:
        raise RuntimeError(f"Cannot access WAV file {wav_path}: {e}")

    # For 24kHz, 16-bit mono, each sample is 2 bytes;
    # bytes per second = 24000 * 2 = 48000. Subtract an approximate 44-byte header.
    header = 44
    if wav_size > header:
        duration = (wav_size - header) / 48000.0
    else:
        duration = 0.0
    meta = {"text": text, "duration": duration}
    meta_bytes = json.dumps(meta).encode("utf-8")
    key = mp3_path.stem
    return key, meta_bytes


def create_sharded_wds_tar(dataset_dir: Path, out_dir, shard_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    current_shard = out_dir / f"shard-{shard_index:06d}.tar"
    tar_out = tarfile.open(current_shard, "w")
    sample_count = 0

    all_mp3_files = sorted(list(dataset_dir.rglob("*.mp3")))
    for mp3_path in tqdm(all_mp3_files, desc="Processing samples"):
        txt_path = mp3_path.with_suffix(".normalized.txt")
        if not txt_path.exists():
            continue
        try:
            key, meta_bytes = process_sample(mp3_path, txt_path)
        except Exception as e:
            print(f"Error processing {mp3_path}: {e}")
            continue

        tar_out.add(str(mp3_path), arcname=f"{key}.mp3")

        json_info = tarfile.TarInfo(name=f"{key}.json")
        json_info.size = len(meta_bytes)
        tar_out.addfile(json_info, io.BytesIO(meta_bytes))

        sample_count += 1
        if sample_count % shard_size == 0:
            tar_out.close()
            shard_index += 1
            current_shard = out_dir / f"shard-{shard_index:06d}.tar"
            tar_out = tarfile.open(current_shard, "w")

    tar_out.close()


# create the shards into the `shards` directory with 1000 examples in each shard

dataset_dir = Path("extracted") / "LibriTTS_R"

create_sharded_wds_tar(dataset_dir, "shards", 1000)
