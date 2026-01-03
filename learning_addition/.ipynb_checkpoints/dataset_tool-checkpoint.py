import json
import random
import os

def generate_addition_problem(k, reverse=True):
    a = random.randint(10**(k-1), 10**k - 1)
    b = random.randint(10**(k-1), 10**k - 1)
    res = str(a + b)
    if reverse:
        res = res[::-1]
    return f"{a}+{b}={res}"

def create_dataset(filename, num_samples, k_list=[3]):
    data = []
    for _ in range(num_samples):
        k = random.choice(k_list)
        data.append(generate_addition_problem(k))
    
    with open(filename, 'w') as f:
        for line in data:
            f.write(line + "\n")
    print(f"Saved {num_samples} samples to {filename}")

if __name__ == "__main__":
    # 1. Training data: 3-digit addition (10k samples)
    create_dataset("train.txt", 10000, k_list=[3])
    
    # 2. In-Distribution Test: 3-digit
    create_dataset("test_id.txt", 1000, k_list=[3])
    
    # 3. OOD Test: 4-digit (Length Generalization)
    create_dataset("test_ood.txt", 500, k_list=[4])
