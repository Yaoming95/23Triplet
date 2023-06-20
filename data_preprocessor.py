from tqdm import tqdm
import json

import os
from neurst.data.text.bpe import BPE

import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

bpe = BPE(vocabulary="data/dict.en2zh.txt",
          subtokenizer_codes="data/bpe.code")


# change data from https://huggingface.co/datasets/Yaoming95/EMMT to json form
split_dict = {
    "test": "data/649c23532f8ee1b002d2ca6bdb8b01107886a208c2129c9781232af02b3a1300.png",
}

# path to save text
file_root = "data/text"
# path to save CLIP emb
emb_root = "data/clip"

emb_dict = {}

for split, file_name in split_dict.items():
    en_file = f"{file_root}{split}.en"
    zh_file = f"{file_root}{split}.zh"
    emb_dict[split] = []
    with open(file_name) as fin, open(en_file, "w") as en, open(zh_file, "w") as zh:
        for line in tqdm(fin):
            line = json.loads(line)
            src_text, trg_text = line["src_text"], line["trg_text"]

            src_text = bpe.tokenize(src_text)
            trg_text = bpe.tokenize(trg_text)

            src_text = " ".join(src_text)
            trg_text = " ".join(trg_text)

            image = line["image"]
            image = preprocess(Image.open(image), ).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features = image_features.unsqueeze(0).detach()
            emb_dict[split].append(image_features)

            en.write(src_text + "\n")
            zh.write(trg_text + "\n")
    embs = torch.cat(emb_dict[split])
    embs = embs.cpu()
    torch.save(embs, f"{emb_root}{split}.pth")





