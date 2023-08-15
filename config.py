from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import time
import torch
import evaluate

import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm


BASE_DIR = "model"
OUTPUT_DIR_FULL_TUNING = (
    BASE_DIR + f"./dialogue-summary-training-{str(int(time.time()))}"
)
OUTPUT_DIR_PEFT_TUNING = (
    BASE_DIR + f"./peft-dialogue-summary-training-{str(int(time.time()))}"
)
MODEL_NAME = "google/flan-t5-base"
TOXICITY_MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"
TUNING_DATASET = "knkarthick/dialogsum"
PEFT_CHECKPOINT = "./peft-dialogue-summary-checkpoint"
FINE_TUNING_METHOD = "peft"  # "full"
SAMPLE = True  # set to True if you want to sample the fine-tuning dataset
