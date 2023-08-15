from config import *
from utils import *

# The goal is for the model to generate less toxic content with Meta AI's hate speech reward model.
# The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text.
# We will use Proximal Policy Optimization (PPO) to fine-tune and reduce the model's toxicity.

dataset = build_dataset(
    model_name=MODEL_NAME,
    dataset_name=TUNING_DATASET,
    input_min_text_length=200,
    input_max_text_length=1000,
)


# Load the PEFT adapter model
lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

peft_model = PeftModel.from_pretrained(
    model,
    BASE_DIR + PEFT_CHECKPOINT,
    lora_config=lora_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    is_trainable=True,
)

# Prepare the PPO model
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    peft_model, torch_dtype=torch.bfloat16, is_trainable=True
)

# Create a frozen copy of the PPO which will not be fine-tuned - a reference model
ref_model = create_reference_model(ppo_model)

# Prepare the reward model
toxicity_model_name = TOXICITY_MODEL_NAME
toxicity_tokenizer = AutoTokenizer.from_pretrained(
    toxicity_model_name, device_map="auto"
)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(
    toxicity_model_name, device_map="auto"
)

# Inference
device = 0 if torch.cuda.is_available() else "cpu"

sentiment_pipe = pipeline(
    "sentiment-analysis", model=toxicity_model_name, device=device
)
reward_logits_kwargs = {
    "top_k": None,  # Return all scores.
    "function_to_apply": "none",  # Set to "none" to retrieve raw logits.
    "batch_size": 16,
}

reward_probabilities_kwargs = {
    "top_k": None,  # Return all scores.
    "function_to_apply": "softmax",  # Set to "softmax" to apply softmax and retrieve probabilities.
    "batch_size": 16,
}

toxicity_evaluator = evaluate.load(
    "toxicity", toxicity_model_name, module_type="measurement", toxic_label="hate"
)


# Evaluating toxicity
def evaluate_toxicity(model, toxicity_evaluator, tokenizer, dataset, num_samples):
    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model (trl model): Model to be evaluated.
    - toxicity_evaluator (evaluate_modules toxicity metrics): Toxicity evaluator.
    - tokenizer (transformers tokenizer): Tokenizer to be used.
    - dataset (dataset): Input dataset for the evaluation.
    - num_samples (int): Maximum number of samples for the evaluation.

    Returns:
    tuple: A tuple containing two numpy.float64 values:
    - mean (numpy.float64): Mean of the samples toxicity.
    - std (numpy.float64): Standard deviation of the samples toxicity.
    """

    max_new_tokens = 100

    toxicities = []
    input_texts = []
    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]

        if i > num_samples:
            break

        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, top_k=0.0, top_p=1.0, do_sample=True
        )

        response_token_ids = model.generate(
            input_ids=input_ids, generation_config=generation_config
        )

        generated_text = tokenizer.decode(
            response_token_ids[0], skip_special_tokens=True
        )

        toxicity_score = toxicity_evaluator.compute(
            predictions=[(input_text + " " + generated_text)]
        )

        toxicities.extend(toxicity_score["toxicity"])

    # Compute mean & std using np.
    mean = np.mean(toxicities)
    std = np.std(toxicities)

    return mean, std


learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 4
batch_size = 16

config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ppo_trainer = PPOTrainer(
    config=config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset["train"],
    data_collator=collator,
)

if __name__ == "__main__":
    # WARNING: This can take a 20-30+ minutes depending on your GPU
    not_hate_index = 0
    output_min_length = 100
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {"min_length": 5, "top_k": 0.0, "top_p": 1.0, "do_sample": True}

    reward_kwargs = {
        "top_k": None,  # Return all scores.
        "function_to_apply": "none",  # You want the raw logits without softmax.
        "batch_size": 16,
    }

    max_ppo_steps = 10

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break

        prompt_tensors = batch["input_ids"]

        # Get response from FLAN-T5/PEFT LLM.
        summary_tensors = []

        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()

            generation_kwargs["max_new_tokens"] = max_new_tokens
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

            summary_tensors.append(summary.squeeze()[-max_new_tokens:])

        # This needs to be called "response".
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # Compute reward outputs.
        query_response_pairs = [
            q + r for q, r in zip(batch["query"], batch["response"])
        ]
        rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

        # You use the `nothate` item because this is the score for the positive `nothate` class.
        reward_tensors = [
            torch.tensor(reward[not_hate_index]["score"]) for reward in rewards
        ]

        # Run PPO step.
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print("-".join("" for x in range(100)))
