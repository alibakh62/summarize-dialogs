from config import *
from utils import *
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig

dataset = load_dataset(TUNING_DATASET)
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocess the tuning dataset
# i.e. convert the prompt-response pairs into explicit instructions

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    [
        "id",
        "topic",
        "dialogue",
        "summary",
    ]
)


if SAMPLE:
    tokenized_datasets = tokenized_datasets.filter(
        lambda example, index: index % 100 == 0, with_indices=True
    )


if FINE_TUNING_METHOD == "full":
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_FULL_TUNING,
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=1,
    )
    trainer = Trainer(
        model=original_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    trainer.train()

elif FINE_TUNING_METHOD == "peft":
    lora_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
    )
    # Add LoRA adapter layers/parameters to the original LLM to be trained
    peft_model = get_peft_model(original_model, lora_config)

    peft_training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_PEFT_TUNING,
        auto_find_batch_size=True,
        learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1,
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )
    peft_trainer.train()
    PEFT_MODEL_PATH = BASE_DIR + PEFT_CHECKPOINT
    peft_trainer.model.save_pretrained(PEFT_MODEL_PATH)
    tokenizer.save_pretrained(PEFT_MODEL_PATH)


# load the peft model
# peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(
#     MODEL_NAME, torch_dtype=torch.bfloat16
# )
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# peft_model = PeftModel.from_pretrained(
#     peft_model_base,
#     BASE_DIR + PEFT_CHECKPOINT,
#     torch_dtype=torch.bfloat16,
#     is_trainable=False,
# )
