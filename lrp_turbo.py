from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "./clinical-qwen-final"
print(f"Loading clinical model from {path}...")
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu")

# Using a more professional medical prompt in the Qwen chat format
messages = [
    {"role": "system", "content": "You are a professional clinical assistant. Provide accurate medical diagnoses based on the symptoms provided."},
    {"role": "user", "content": "A patient presents with acute onset of fever, headache, and visual disturbances. What is the most probable diagnosis? Please explain your reasoning and provide the final answer."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt")

print("\nGenerating Full Clinical Diagnosis...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=250, # Increased for a full answer
    repetition_penalty=1.1,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- Final Model Output ---")
# Strip the prompt and just show the answer
print(response.split("assistant")[-1].strip())
