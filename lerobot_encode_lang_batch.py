import os
import json

import torch
import yaml
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from models.multimodal_encoder.t5_encoder import T5Embedder

GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"

OFFLOAD_DIR = "data/lerobot/lang_embeddings/"

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)

    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        # use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # Load LeRobot datasets
    datasets = {}
    dataset_names_cfg = f'configs/finetune_datasets.json'
    with open(dataset_names_cfg, 'r') as f:
        dataset_names = json.load(f)
    for dataset_name in dataset_names:
        datasets[dataset_name] = LeRobotDataset(dataset_name)

    # For each dataset in LeRobot dataset, loop through available tasks
    # and add them to task_descriptions
    task_descriptions = {}
    for dataset in datasets.values():
        for task in dataset.meta.tasks.values():
            task_descriptions[task] = True

    for task in tqdm(task_descriptions.keys()):
        # Encode the instructions
        tokenized_res = tokenizer(
            task,   # TODO: does this need to be a string or a list?
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        tokens = tokenized_res["input_ids"].to(device)
        attn_mask = tokenized_res["attention_mask"].to(device)
        
        with torch.no_grad():
            text_embeds = text_encoder(
                input_ids=tokens,
                attention_mask=attn_mask
            )["last_hidden_state"].detach().cpu()
        
        attn_mask = attn_mask.cpu().bool()

        text_embed = text_embeds[attn_mask] # TODO: text_embeds[0][attn_mask[0]] ??
        save_path = os.path.join(OFFLOAD_DIR, f"{task}_embed.pt")
        torch.save(text_embed, save_path)

        save_path = os.path.join(OFFLOAD_DIR, f"{task}_attn_mask.pt")
        torch.save(attn_mask, save_path)


if __name__ == "__main__":
    main()
