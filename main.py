import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import bisect

with open('russian_nouns.txt', 'r', encoding='utf-8') as f:
    words = [line.strip() for line in f]

with open('easy_russian_nouns.txt', 'r', encoding='utf-8') as f:
    easy_words = [line.strip() for line in f]

embeddings = np.load('embeddings/DeepPavlov_rubert-base-cased.npy', allow_pickle=True).item()

class WordGame:
    def __init__(self):
        self.target_word = None
        self.sorted_words = []
        self.word_to_rank = {}
        self.attempts_history = []
        
    def start_game(self, words, embeddings):
        self.target_word = np.random.choice(easy_words)
        target_embedding = embeddings[self.target_word]
        
        # Сортируем все слова один раз при старте игры
        similarities = {
            word: cosine_similarity([emb], [target_embedding])[0][0]
            for word, emb in embeddings.items()
        }
        self.sorted_words = sorted(words, key=lambda x: -similarities[x])
        self.word_to_rank = {word: i+1 for i, word in enumerate(self.sorted_words)}

    def guess(self, user_word):
        user_word = user_word.strip().lower()
        
        if user_word not in self.word_to_rank:
            return "Слово не найдено в словаре", False

        rank = self.word_to_rank[user_word]
        
        # Добавляем попытку в историю с сортировкой по рангу
        bisect.insort(self.attempts_history, (rank, user_word))
        
        # Формируем топ-20 уникальных попыток
        unique_attempts = []
        seen = set()
        for rank, word in self.attempts_history:
            if word not in seen:
                seen.add(word)
                unique_attempts.append((rank, word))
            if len(unique_attempts) >= 20:
                break

        response = "Лучшие попытки:\n" + "\n".join(
            [f"{r}: {w}" for r, w in unique_attempts]
        )
        
        if user_word == self.target_word:
            return f"ПОБЕДА! Слово угадано: {self.target_word}\n" + response, True
            
        return f"Позиция вашего слова: {rank}\n" + response, False
    
# Запуск игры
game = WordGame()
game.start_game(words, embeddings)

while True:
    guess_word = input("Ваше предположение: ").lower().strip()
    response, is_correct = game.guess(guess_word)
    print(response)
    
    if is_correct:
        break