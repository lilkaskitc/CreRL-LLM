from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------------
# LOAD MODEL
# ------------------------
print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype="bfloat16",
    device_map="auto",
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base, "./output/crerl_r2avg_grpo_lora4g_5467274")

print("Merging LoRA weights...")
model = model.merge_and_unload()

model.save_pretrained("./output/crerl_r2avg_grpo_lora4g_5467274_merged")
tokenizer = AutoTokenizer.from_pretrained("./output/crerl_r2avg_grpo_lora4g_5467274")
tokenizer.save_pretrained("./output/crerl_r2avg_grpo_lora4g_5467274_merged")