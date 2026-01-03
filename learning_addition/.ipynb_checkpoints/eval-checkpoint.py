import torch
from transformers import GPT2LMHeadModel

# 1. Setup exact same character mapping
chars = sorted(list("0123456789+="))
char_to_id = {s: i for i, s in enumerate(chars)}
id_to_char = {i: s for i, s in enumerate(chars)}

def solve_addition(model, prompt):
    """
    Takes '123+456=' and returns only the digits generated after the '='
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = torch.tensor([char_to_id[c] for c in prompt]).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids, 
            max_new_tokens=5, 
            pad_token_id=char_to_id['='],
            eos_token_id=char_to_id['='],
            do_sample=False  # Greedy
        )
    
    # Extract only the NEW tokens (those after the prompt)
    generated_ids = output_tokens[0][len(input_ids[0]):]
    
    # Convert IDs to string, stopping if we hit a non-digit (like '=' padding)
    prediction = ""
    for tid in generated_ids:
        char = id_to_char[tid.item()]
        if char in "0123456789":
            prediction += char
        else:
            break # Stop at '=' or any other separator
            
    return prediction

def run_eval(model_path, test_file):
    print(f"\n--- Evaluating {test_file} ---")
    model = GPT2LMHeadModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    
    correct = 0
    total = 0
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # line is "123+456=975"
        prompt_part, expected = line.split('=')
        prompt_with_equals = prompt_part + "="
        
        predicted = solve_addition(model, prompt_with_equals)
        
        if predicted == expected:
            correct += 1
        
        # Debug: Print first 5 samples to see what's happening
        if i < 5:
            # Un-reverse for readability in the console
            real_a_plus_b = prompt_part
            real_expected = expected[::-1]
            real_predicted = predicted[::-1]
            print(f"Sample {i+1}: {real_a_plus_b} | Expected: {real_expected} (rev: {expected}) | Predicted: {real_predicted} (rev: {predicted})")
        
        total += 1
            
    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    # Ensure the model exists
    try:
        id_acc = run_eval("./final_model", "test_id.txt")
        ood_acc = run_eval("./final_model", "test_ood.txt")
    except Exception as e:
        print(f"Error during evaluation: {e}")
