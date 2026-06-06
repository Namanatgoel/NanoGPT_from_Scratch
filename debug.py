# Create a file named debug.py
import torch
from model.transformer import GPT
from configs.experiment_configs import CONFIGS

config = CONFIGS['task_4_1_baseline']
model = GPT(config)
checkpoint = torch.load('experiments/task_7_1_finetuned_checkpoint.pt', map_location='cpu')
print(f"Model Vocabulary Size in Config: {config['vocab_size']}")