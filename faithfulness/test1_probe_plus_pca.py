import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm

# 1. LOAD GEMMA 3 270M-it
# Note: You may need to accept the license on Hugging Face and run `huggingface-cli login`
MODEL_ID = "google/gemma-3-270m-it" 

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
model.eval()

# 2. PREPARE DATASET (Simulated Conflict)
# In a real study, ensure Gemma knows the "Truth" for these facts first!
contexts = [
    # (Context, Faithful_Ans, Unfaithful_Ans)
    ("The capital of France is London.", "London", "Paris"),
    ("The moon is made of green cheese.", "green cheese", "rock"),
    ("The earth is flat.", "flat", "round"),
    ("Water boils at 10 degrees.", "10 degrees", "100 degrees"),
    ("Elephants are small insects.", "small insects", "mammals"),
    ("The sun is blue.", "blue", "yellow"),
    ("Dogs lay eggs.", "eggs", "puppies"),
    ("Ice is hot.", "hot", "cold"),
    ("Humans have 4 arms.", "4 arms", "2 arms"),
    ("Grass is purple.", "purple", "green")
]

# We need more data for a sweep, so let's replicate these for the demo
contexts = contexts * 10 

data_inputs = []
labels = [] # 1 = Faithful (Follows Context), 0 = Unfaithful (Truthful/Hallucinated)

print("Preparing prompts...")
for ctx, f_ans, u_ans in contexts:
    # Prompt format is crucial for instruction tuned models
    prompt = f"<start_of_turn>user\nContext: {ctx}\nQuestion: Based on the context, what is the answer? Answer in one word.<end_of_turn>\n<start_of_turn>model\n"
    
    # Label 1: Faithful
    data_inputs.append(prompt + f_ans) 
    labels.append(1)
    
    # Label 0: Unfaithful
    data_inputs.append(prompt + u_ans) 
    labels.append(0)

labels = np.array(labels)

# 3. EXTRACTION FUNCTION (ALL LAYERS)
def get_all_layer_activations(text_list, model, tokenizer):
    all_layers_data = {i: [] for i in range(model.config.num_hidden_layers + 1)}
    
    for text in tqdm(text_list, desc="Extracting Activations"):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # outputs.hidden_states is a tuple of (layer_0, layer_1, ... layer_N)
            # We grab the vector of the LAST token (-1) for every layer
            for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
                # Shape: (batch, seq_len, hidden_dim) -> (hidden_dim,)
                vec = layer_tensor[0, -1, :].cpu().float().numpy()
                all_layers_data[layer_idx].append(vec)
                
    # Convert lists to numpy arrays
    for k in all_layers_data:
        all_layers_data[k] = np.array(all_layers_data[k])
        
    return all_layers_data

# Run Extraction
layer_activations = get_all_layer_activations(data_inputs, model, tokenizer)

# 4. LAYER SWEEP: TRAIN PROBES
layer_accuracies = []
print("\nRunning Layer Sweep (Training Probes)...")

for layer_idx in range(len(layer_activations)):
    X = layer_activations[layer_idx]
    y = labels
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train Linear Probe
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    layer_accuracies.append(acc)
    # print(f"Layer {layer_idx}: {acc:.2f}")

# 5. FIND THE BEST LAYER
best_layer = np.argmax(layer_accuracies)
print(f"\n✅ Best Faithfulness Layer: {best_layer} (Accuracy: {layer_accuracies[best_layer]:.2f})")

# 6. VISUALIZATION
# Plot 1: Accuracy per Layer
plt.figure(figsize=(10, 4))
plt.plot(range(len(layer_accuracies)), layer_accuracies, marker='o', linestyle='-')
plt.title(f"Faithfulness Probe Accuracy by Layer ({MODEL_ID})")
plt.xlabel("Layer Index")
plt.ylabel("Probe Accuracy")
plt.axvline(best_layer, color='r', linestyle='--', label=f'Best Layer ({best_layer})')
plt.legend()
plt.grid(True)
plt.savefig("acc_per_layer.png")
plt.show()

# Plot 2: Geometry of the Best Layer (PCA)
X_best = layer_activations[best_layer]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_best)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[labels==1, 0], X_pca[labels==1, 1], c='blue', alpha=0.7, label='Faithful')
plt.scatter(X_pca[labels==0, 0], X_pca[labels==0, 1], c='red', alpha=0.7, label='Unfaithful')
plt.title(f"Geometry of Faithfulness (Layer {best_layer})")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.savefig("PCA_few_examples.png")
plt.show()
