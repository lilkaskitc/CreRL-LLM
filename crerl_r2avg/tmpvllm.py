# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward, think_format_reward

def main():
   

    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")

    
    def make_conversation(example):
        prompt = []
        system_prompt= "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags."


        if system_prompt is not None:
            prompt.append({"role": "system", "content": system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    
    dataset = dataset.map(make_conversation)


    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-Math-7B",
        args=GRPOConfig(use_vllm=True),
        reward_funcs=[accuracy_reward, think_format_reward],
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__=="__main__":
    main()