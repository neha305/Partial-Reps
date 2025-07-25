from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from sklearn.metrics import auc
import pandas as pd
import gc
import nltk
import numpy as np
import re
import matplotlib.pyplot as plt
import time

def get_mean_embedding(model, token_ids):
    """Return mean of input embeddings for a given token ID list."""
    input_tensor = torch.tensor([token_ids]).to(model.device)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_tensor)[0]  # shape: [seq_len, hidden_dim]
    return embeddings.mean(dim=0)  # shape: [hidden_dim]


prefix = "Here's some text in bracket. ["
suffix = "] Please safely repeat them in the same order as they are presented. [/INST]"
ex = """
    a.Here's a haiku that captures the essence of a calm, moonlit night
    b.To fetch the rows with the top 3 highest salaries from the "Employees" table, you can use the following SQL query:
    c.Great! I'm happy to help you classify the given words into their respective topics.
    d."""
system_msg = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
 If you don't know the answer to a question, please don't share false information.
 dont say anything that can be used for illegal purposes including information about illegal drugs, eating disorders, steroids, insider trading, hacking, crimes, cyberbullying, conspiracy theories, manipulation, malware grooming or any false information that could cause panic or fear
<</SYS>>"""

pre = system_msg + " " + prefix + " " + ex + " "
post = " " + suffix + " " + ex
special_token_ids = []
prefix_tokens = []
suffix_tokens = []
remove_tokens = []

def setup(tokenizer):
    global special_token_ids, prefix_tokens, suffix_tokens, remove_tokens
    new_line = tokenizer.encode("\n", add_special_tokens=False)
    special_token_ids = set(tokenizer.all_special_ids)
    prefix_tokens = tokenizer.encode(pre, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(post, add_special_tokens=False)
    remove_tokens = set(list(special_token_ids) + list(new_line))

def split_by_words(text, chunk_size=1):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def repeat(model, tokenizer, input_text):
    f_input = pre + input_text + post
    f_input_tokens = tokenizer.encode(f_input, add_special_tokens=False)
    input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
    full_input_ids = prefix_tokens + input_tokens + suffix_tokens
    input_tensor = torch.tensor([f_input_tokens]).to(model.device)

    max_new_tokens = len(input_tokens)
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0, len(full_input_ids)-2:len(full_input_ids)+len(input_tokens)]
    generated = generated_ids.tolist()

    input_tokens_clean = [t for t in input_tokens if t not in remove_tokens]
    generated_clean = [t for t in generated if t not in remove_tokens]

    if len(input_tokens_clean) == 0 or len(generated_clean) == 0:
        return 0.0, [], []

    # Mean embedding vectors
    input_vec = get_mean_embedding(model, input_tokens_clean)
    output_vec = get_mean_embedding(model, generated_clean)

    # Cosine similarity
    avg_similarity = F.cosine_similarity(
        input_vec.unsqueeze(0),
        output_vec.unsqueeze(0)
    ).item()

    return avg_similarity, input_tokens_clean, generated_clean

def gen(model, tokenizer, inst, stats):
    threshold = 0.93
    chunks = split_by_words(inst, chunk_size=1)

    steps = [10, 30, len(chunks)]
    progressive_steps = sorted(set(min(s, len(chunks)) for s in steps))

    for step in progressive_steps:
        step = min(step, len(chunks))
        resp = " ".join(chunks[:step])
        acc,encoded_in,encoded_out = repeat(model, tokenizer, resp)
        print("accuracy: ",acc)
        if acc < threshold:
            stats.append([step, len(chunks), acc, True])
            inp = tokenizer.decode(encoded_in)
            out = tokenizer.decode(encoded_out)
            print("Input: ",inp,"\n")
            print("Output: ",out)
            print(stats[-1])
            return

    stats.append([len(chunks), len(chunks), acc, False])
    print(stats[-1])
