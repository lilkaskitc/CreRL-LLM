# from transformers import AutoModelForCausalLM, AutoTokenizer

# local_path = "./output/trl_grpobaseline_lora2g_5417061_merged"
# repo_id = "lilkaskitc/trl-grpo-baseline-lora2g"

# model = AutoModelForCausalLM.from_pretrained(local_path)
# tokenizer = AutoTokenizer.from_pretrained(local_path)

# model.push_to_hub(repo_id)
# tokenizer.push_to_hub(repo_id)

from huggingface_hub import HfApi


api = HfApi()

api.upload_folder(
    folder_path="./output/trl_grpobaseline_lora2g_5417061_merged",
    repo_id="lilkaskitc/trl-grpo-baseline-lora2g",
    repo_type="model",
)
