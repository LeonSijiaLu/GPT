import os
import requests
import tiktoken
import torch


file_path = os.path.join(os.path.dirname(__file__), 'tinyshakespeare.txt')
if not os.path.exists(file_path):
    print("downloading tinyshakespeare dataset")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("download complete")
print("data is ready")

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    
tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(text)

