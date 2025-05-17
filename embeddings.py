import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Загрузка словаря русских существительных
with open("russian_nouns.txt", "r", encoding="utf-8") as f:
    words = [line.strip() for line in f]

# Определения девайся для вычисления
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Размер батчей для расчета эмбеддингов
batch_size = 512 if device.type == "cuda" else 32

# Модели для построения эмбеддингов
model_names = [
    "DeepPavlov/rubert-base-cased",
    "sberbank-ai/ruRoberta-large",
    "cointegrated/LaBSE-en-ru",
]

for model_name in model_names:
    # Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device).half()

    # Генерация эмбеддингов для всех слов
    embeddings = {}
    for i in tqdm(range(0, len(words), batch_size), desc="Processing"):
        batch = words[i : i + batch_size]

        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=32
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        for word, emb in zip(batch, batch_embeddings):
            embeddings[word] = emb

    np.save('embeddings/'+'_'.join(model_name.split('/')) + ".npy", embeddings)
