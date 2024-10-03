# Adapted from my older semantic search script

import nltk
from nltk.corpus import wordnet as wn
import random
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
nltk.download('wordnet')

def emb_calc(words):
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    embeddings = model.encode(words)
    # embeddings.shape

    similarity_matrix = util.cos_sim(embeddings, embeddings)
    return similarity_matrix

class CorpusGenerator:
    def __init__(self, n=50):
        self.n = n
        self.adjectives = list(wn.all_synsets(pos=wn.ADJ))

    def rand(self,n):
        sample_ = random.sample(self.adjectives, n) 
        return [adj.lemmas()[0].name() for adj in sample_]

    def get_synonyms(self,word, limit=5):
        synonyms = set()
        for synset in wn.synsets(word, pos=wn.ADJ):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
                if len(synonyms) >= limit:
                    return list(synonyms)
        return list(synonyms)


def runner(num_runs=7):
    mean_sim_runs = []
    mean_dissim_runs = []

    CG = CorpusGenerator()
    mean_thresholds = []

    for i in range(num_runs):
        words = CG.rand(50)
        corpora = {i : CG.get_synonyms(word=i,limit=3) for i in words}
        # print(corpora)

        all_words = list(corpora.keys()) + [syn for synonyms in corpora.values() for syn in synonyms]
        all_words = list(set(all_words))
        sim_mat = emb_calc(all_words)

        
        arr = []

        similarities = []
        dissimilarities = []

        word_to_index = {word: idx for idx, word in enumerate(all_words)}

        for key, values in corpora.items():
            key_index = word_to_index[key]
            value_indices = [word_to_index[value] for value in values if value in word_to_index]

            similarities.extend([sim_mat[key_index, val_index] for val_index in value_indices])
            similarities.extend([sim_mat[i, j] for i, j in zip(value_indices, value_indices[1:])])
            related_indices = {key_index, *value_indices}
            dissimilarities.extend([sim_mat[key_index, other_idx] for other_idx in range(len(all_words)) if other_idx not in related_indices])

        mean_similarity = np.mean(similarities)
        mean_dissimilarity = np.mean(dissimilarities)
        arr.append(mean_similarity)
        arr.append(mean_dissimilarity)
        mean_threshold = np.mean(arr)

        print("Mean Threshold:", mean_threshold)
        mean_thresholds.append(mean_threshold)

        plt.figure(figsize=(12, 10))
        plt.tight_layout()
        sns.heatmap(sim_mat.numpy(), xticklabels=all_words, yticklabels=all_words, cmap='coolwarm', annot=False)
        # plt.suptitle('Cosine Similarity Matrix')
        plt.title(f'Run_{i}, Mean Similarity: {np.round(mean_similarity, 4)}, Mean Dissimilarity: {np.round(mean_dissimilarity, 4)}')
        plt.xticks(rotation=90)
        plt.show()
        plt.savefig(f'Run_{i}.png')
    
    print(mean_thresholds)
    plt.figure(figsize=(10, 6))
    plt.hist(mean_thresholds, bins=50)
    plt.title('Mean Thresholds Distribution')
    plt.show()
    return mean_thresholds

if __name__ == "__main__":
    thresh = runner()
    print(f"Upper Limit: {np.max(thresh)}, Lower Limit: {np.min(thresh)}")
