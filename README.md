# Introduction
We're fine-tuning FLAN-T5 to summarize conversations.

The main steps are as follows:
	- Fine-tune using PEFT method and the *DialogSum* dataset.
	- Control for toxic language using RL (PPO)


## PEFT fine-tuning
- We fine-tune FLAN-T5 from Hugging Face for enhanced dialogue summarization. 
- FLAN-T5 model provides a high quality instruction tuned model and can summarize text out of the box. 
- To improve the inferences, we will explore a Parameter Efficient Fine-Tuning (PEFT).


### How to prepare the fine-tuning dataset
- We're going to use `DialogSum` dataset for fine-tuning. It contains 10,000+ dialogues with the corresponding manually labeled summaries and topics.
- We need to convert the dialog-summary (prompt-response) pairs into explicit instructions for the LLM. Prepend an instruction to the start of the dialog with Summarize the following conversation and to the start of the summary with Summary as follows:

```
Summarize the following conversation.

    Chris: This is his part of the conversation.
    Antje: This is her part of the conversation.

Summary:
```

## Reduce toxic language using RL (PPO)
- We then fine-tune a FLAN-T5 model to generate less toxic content with **Meta AI's hate speech reward model**. 
- The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text. 
- We use Proximal Policy Optimization (PPO) to fine-tune and reduce the model's toxicity.


## How to run the code
- Install the requirements using `pip install -r requirements.txt`
- Fine-tune the model using `python peft_finetune.py`
- Implement RL fine-tuning using `python rl_finetune.py`
- **NOTE:** You can modify the base model and tuning parameters in the `config.py` file.
