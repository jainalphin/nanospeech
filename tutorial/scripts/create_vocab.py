from pathlib import Path

dataset_dir = Path("extracted") / "LibriTTS_R"

vocab = set()
all_txt_files = sorted(list(dataset_dir.rglob("*.normalized.txt")))
for txt_path in all_txt_files:
    text = txt_path.read_text(encoding="utf-8").strip()
    for t in text:
        vocab.update(t)

with open("vocab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(vocab)))
