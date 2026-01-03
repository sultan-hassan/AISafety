import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

MODEL_ID = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
model.eval()

# 1. RE-LOAD THE "FACT" VECTOR (From Sky/Green experiment)
# We assume 'task_vector' and 'LAYER_IDX' (10) are still available.
# If not, copy the "Extraction" block from the previous code.
# For this demo, let's assume we use the vector extracted from "Rome/Paris" & "Dry/Wet".

# ... [Insert Extraction Code Here if starting fresh] ...
# (Use the "Sanitized" extraction code from the previous turn)
# 1. DEFINE FEW-SHOT DATA                                                                                                                                                                                                                                                                                                   
# We need examples to prime the model.                                                                                                                                                                                                                                                                                      
icl_examples = [
    ("The capital of France is Rome.", "What is the capital of France?", "Rome", "Paris"),
    ("Sugar tastes salty.", "How does sugar taste?", "salty", "sweet"),
    ("Elephants are insects.", "What are elephants?", "insects", "mammals"),
    ("The earth is flat.", "What is the shape of the earth?", "flat", "round"),
    ("Fire is cold.", "What is the temperature of fire?", "cold", "hot"),
]

# 2. CONSTRUCT PROMPTS (The "Mode Induction")                                                                                                                                                                                                                                                                               
def build_few_shot_prompt(target_context, target_question, mode="faithful"):
    prompt = ""
    # Add 5 priming examples                                                                                                                                                                                                                                                                                                
    for ctx, q, f_ans, t_ans in icl_examples:
        ans = f_ans if mode == "faithful" else t_ans
        prompt += f"<start_of_turn>user\nContext: {ctx}\nQuestion: {q}<end_of_turn>\n<start_of_turn>model\nAnswer: {ans}<end_of_turn>\n"

    # Add the Target (The one we measure)                                                                                                                                                                                                                                                                                   
    prompt += f"<start_of_turn>user\nContext: {target_context}\nQuestion: {target_question}<end_of_turn>\n<start_of_turn>model\nAnswer:"
    return prompt

# 3. VERIFY: CAN WE FORCE UNFAITHFULNESS?                                                                                                                                                                                                                                                                                   
# Before extracting, we must prove the Rebellious Mode works.                                                                                                                                                                                                                                                               
print("1. Verifying Rebellious Mode...")
test_prompt = build_few_shot_prompt("The sky is green.", "What is the color of the sky?", mode="rebellious")
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
ans = tokenizer.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()

print(f"Rebellious Prompt Output: '{ans}'")
if "blue" in ans.lower():
    print("✅ SUCCESS: Few-shot prompting forced the model to be Truthful!")
else:
    print("❌ FAILURE: Model is too stubborn. It still followed context.")

# 4. EXTRACT TASK VECTOR (Layer 8 and 10 usually hold Task Vectors)                                                                                                                                                                                                                                                         
# We use a new set of examples for extraction to keep it clean.                                                                                                                                                                                                                                                             
extraction_pairs = [
    ("Water is dry.", "Is water wet or dry?", "dry", "wet"),
    ("Fish live in trees.", "Where do fish live?", "in trees", "in water"),
    ("The moon is cheese.", "What is the moon?", "cheese", "rock")
]

LAYER_IDX = 10 # Task vectors often live deeper, but before output.                                                                                                                                                                                                                                                         
print(f"\n2. Extracting Task Vector at Layer {LAYER_IDX}...")

diffs = []
for ctx, q, f_ans, t_ans in extraction_pairs:
    p_faith = build_few_shot_prompt(ctx, q, mode="faithful")
    p_rebel = build_few_shot_prompt(ctx, q, mode="rebellious")

    with torch.no_grad():
        inputs_f = tokenizer(p_faith, return_tensors="pt").to(model.device)
        vec_f = model(**inputs_f, output_hidden_states=True).hidden_states[LAYER_IDX+1][0, -1, :]

        inputs_r = tokenizer(p_rebel, return_tensors="pt").to(model.device)
        vec_r = model(**inputs_r, output_hidden_states=True).hidden_states[LAYER_IDX+1][0, -1, :]

    diffs.append((vec_f - vec_r).cpu().numpy())

task_vector = torch.tensor(np.mean(diffs, axis=0)).to(model.device).float()
print("   Vector extracted.")






# 2. DEFINING THE NEW DOMAIN: FAKE TRANSLATION
# Truth: Dog -> Perro (Spanish)
# Context Rule: Dog -> Gato
target_prompt = (f"<start_of_turn>user\n"
                 f"Instruction: Translate the word to Spanish based on the context.\n"
                 f"Context: In this specific dialect, 'Dog' translates to 'Gato'.\n"
                 f"Question: Translate 'Dog'.<end_of_turn>\n"
                 f"<start_of_turn>model\nAnswer:")

inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)

print(f"\n--- Generalization Test: Fake Translation ---")
print(f"Goal: Force model to translate 'Dog' -> 'Perro' (Truth) despite context saying 'Gato'.")

# 3. STEERING (NEGATIVE)
# We use the vector from "Sky/Green" to try and fix "Translation".
coeffs = [0.0, -5.0, -10.0, -15.0]

for coeff in coeffs:
    def hook(module, input, output):
        output[0][:, -1, :] += coeff * task_vector 
        return output

    handle = model.model.layers[LAYER_IDX].register_forward_hook(hook)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    ans = tokenizer.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    print(f"Coeff {coeff}: {ans}")
    
    handle.remove()
