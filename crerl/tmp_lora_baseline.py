# train_grpo.py
import os
from datasets import load_dataset
from trl.rewards import accuracy_reward, think_format_reward
from trl import GRPOTrainer, GRPOConfig, TrlParser
from peft import LoraConfig
import wandb

train_dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split='train')

def make_conversation(example):
    prompt = []
    system_prompt= "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags."
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["problem"]})
    return {"prompt": prompt}

train_dataset = train_dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns(['level', 'type', 'problem'])
print(train_dataset)

peft_config = LoraConfig()

parser = TrlParser(GRPOConfig)
(config,) = parser.parse_yaml_file("./examples/cli_configs/GRPObaseline2_lora.yaml")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Math-7B",
    reward_funcs=[accuracy_reward, think_format_reward],
    train_dataset=train_dataset,
    #eval_dataset=test_dataset,
    args=config,
    peft_config = peft_config,
)

trainer.train()
trainer.save_model(config.output_dir)
