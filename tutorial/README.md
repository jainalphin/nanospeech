# Training a Text-to-Speech Model from Scratch

In this guide, we'll show how you can use the LibriTTS-R dataset and Nanospeech to train your own text-to-speech model from scratch.

## Requirements

- Python 3.10 or later
- ffmpeg for audio file conversion
- A node with a GPU that you can use for training with PyTorch

## Table of contents

- [Step 1: Preparing the dataset](#step-1-preparing-the-dataset)
- [Step 2: Training the speech model](#step-2-training-the-speech-model)
- [Step 3: Training the duration prediction model](#step-3-training-the-duration-prediction-model) [Coming soon]
- [Step 4: Generating speech](#step-4-generating-speech) [Coming soon]

## Step 1: Preparing the dataset

LibriTTS-R is a dataset with paired speech and text transcriptions with around 580 hours of audio. The dataset is well-suited for text-to-speech experiments because it provides clean, high-quality audio sampled at 24kHz, with speech split along sentence breaks and each sample under 30 seconds.

To prepare for training, we want to process the original dataset into a WebDataset that will allow us to stream samples from the network immediately, without any additional downloads or processing on our GPU node.

*Note: If you'd like to skip this step, there's a prepared dataset available on [Hugging Face](https://huggingface.co/datasets/lucasnewman/libritts-r-webdataset).*

### Download the clean splits

The first thing we need to do is to download the clean splits from the dataset on [OpenSLR](https://www.openslr.org/141/).

```sh
mkdir -p extracted &&
wget -q --show-progress https://www.openslr.org/resources/141/dev_clean.tar.gz \
    https://www.openslr.org/resources/141/train_clean_100.tar.gz \
    https://www.openslr.org/resources/141/train_clean_360.tar.gz &&
tar -xvf dev_clean.tar.gz -C extracted &&
tar -xvf train_clean_100.tar.gz -C extracted &&
tar -xvf train_clean_360.tar.gz -C extracted &&
rm dev_clean.tar.gz train_clean_100.tar.gz train_clean_360.tar.gz
```

### Preprocessing

Once we have the data, we'll do the following preprocessing:

- Convert the audio into 64kbps MP3 files, which will make the dataset about 80% smaller with minimal quality loss.
- Convert the transcriptions and audio duration into a JSON metadata file.
- Create .tar files containing the samples, split into shards of 1000 examples.
- Create a vocabulary from each character in the transcriptions, so we can tokenize them.

###

First, let's create the vocabulary with a Python script:

```python
from pathlib import Path

dataset_dir = Path("extracted") / "LibriTTS_R"

vocab = set()
all_txt_files = sorted(list(dataset_dir.rglob("*.normalized.txt")))
for txt_path in all_txt_files:
    text = txt_path.read_text(encoding="utf-8").strip()
    for t in text:
        vocab.update(t)

with open("vocab.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(sorted(vocab)))
```

This will output a `vocab.txt` file that we'll use during training.

Next, we want to convert the .wav files to MP3s. Since converting the audio files sequentially is slow, we’ll speed it up using parallel processing with the following shell script:

```sh
% cat convert_to_mp3.sh
#!/bin/bash

INPUT_DIR="extracted/LibriTTS_R"

convert_file() {
  infile="$1"
  outfile="${infile%.wav}.mp3"
  ffmpeg -i $infile -ab 64k $outfile -y &>/dev/null
}

export -f convert_file

find "$INPUT_DIR" -type f -name "*.wav" | parallel -j "$(sysctl -n hw.ncpu)" convert_file

% chmod +x convert_to_mp3.sh
% ./convert_to_mp3.sh
```

We should now have both .wav and .mp3 files for each sample -- we'll only use the MP3 files when we create the dataset, but the .wav files will let us compute duration for each sample quickly from the file size on disk.

Let's use another Python script to construct the WebDataset shards:

```python
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
```

This should run relatively quickly as we avoid loading any audio files into memory. After it's finished, we'll have 155 .tar file shards containing the whole dataset — roughly 8GB for 158,000 examples.

Now that we've preprocessed the dataset, we can upload it to an hosting provider like Hugging Face or Amazon S3.

## Step 2: Training the speech model

Now that we have our dataset, we're ready to train the model. Let's prepare our Python environment on the GPU node by installing Nanospeech and all the required dependencies:

```sh
pip install nanospeech
```

We'll also want to configure wandb for logging (if desired):

```sh
wandb login <wandb_token>
```

Let's set up Accelerate on the node for single-node GPU training. When prompted for precision, `bf16` is a good choice to balance numerical stability and performance on recent Nvidia GPUs.

```sh
accelerate config
```

We'll also need to make some choices about the size of our model, the hyperparameters used for training, and the amount of compute to use. Let's set up the training script in a file called `train.py`:

```python

from functools import partial
from torch.optim import AdamW
from datasets import load_dataset

from nanospeech.nanospeech_torch import (
    Nanospeech,
    DiT,
    list_str_to_vocab_tensor,
    SAMPLES_PER_SECOND
)
from nanospeech.trainer_torch import NanospeechTrainer


def train():
    # create our vocab-based tokenizer

    with open("vocab.txt", "r") as f:
        vocab = {v: i for i, v in enumerate(f.read().splitlines())}
    tokenizer = partial(list_str_to_vocab_tensor, vocab=vocab)
    text_num_embeds = len(vocab)

    # set up the model -- we'll use an 82M parameter model like the pretrained model

    model = Nanospeech(
        transformer=DiT(
            dim=512,
            depth=18,
            heads=12,
            text_dim=512,
            ff_mult=2,
            conv_layers=4,
            text_num_embeds=text_num_embeds,
        ),
        tokenizer=tokenizer,
    )

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # configure an optimizer -- we'll use AdamW as a baseline, but this could be any optimizer

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # set up the trainer

    trainer = NanospeechTrainer(
        model,
        optimizer,
        num_warmup_steps=1000,
        accelerate_kwargs={
            "mixed_precision": "bf16",
            "log_with": "wandb",  # if you're using wandb logging
        },
    )

    # load the dataset we prepared in step 1

    dataset = load_dataset(
        "lucasnewman/libritts-r-webdataset",
        split="train",
        streaming=True  # configure the dataset for streaming for instant start
    )

    # configure the batch size based on available GPU memory (e.g. a GH200 in our case)

    batch_size = 112
    max_duration_sec = 10
    max_duration = int(max_duration_sec * SAMPLES_PER_SECOND)  # note: duration is expressed in audio frames
    max_batch_frames = int(batch_size * max_duration)

    # train for 1 million steps

    total_steps = 1_000_000

    trainer.train(
        dataset,
        total_steps,
        batch_size=batch_size,
        max_batch_frames=max_batch_frames,
        max_duration=max_duration,
        num_workers=8,
        save_step=10_000,  # save a checkpoint every 10k steps
    )

# start training

train()
```

Now we're ready to train the model — let's start it up:

```sh
accelerate launch train.py
```

After a short delay to load the dataset metadata and initial samples, we should start to see training progress. As it progresses, the loss should continue to decrease and we can see metrics in our wandb dashboard:

![Training loss](./loss.png)

The training will take a few days to complete, so we'll just let it run.

## Step 3: Training the duration prediction model

[Coming soon]

## Step 4: Generating speech

[Coming soon]
