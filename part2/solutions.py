import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def analyse_embs(trained_emb_model, pos_words, neg_words):
    words = pos_words + neg_words
    labels = [1] * len(pos_words) + [0] * len(neg_words)
    embeddings = np.stack([trained_emb_model.embed_word(word) for word in words])

    x_reduced = PCA(n_components = 2).fit_transform(embeddings)
    plt.scatter(
        x_reduced[:, 0],
        x_reduced[:, 1],
        c = labels,
        alpha = 0.5
    )

    plt.xlabel('pca_component 1')
    plt.ylabel("pca_component 2")
    plt.title("Embeddings")
    plt.show()

def cos_similarity(u, v):  
    """
    Return the cosine similarity (a real number) of the two words in the embedding space of our model.
    """      
    numerator = np.dot(u,v)
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    return numerator / denominator

def topk_ids(v, E, k=10):    
    """
    word embedding: a D-dimensional vector
    E: our model's embedding matrix as a numpy array of shape [V, D]
    k: how many nearest neighbours we want to find

    Return a list with the k word ids that are nearest to the input word embedding v
     in embedding space according to cosine similarity. 
    """ 
    similarities = []  
    for i in E:
        similarities.append(cos_similarity(v, i))

    ids = np.argsort(similarities)[::-1]
    return ids[:k].tolist()

def demo_similarity(words, model):
    for i in words:
        embedding_i = model.embed_word(i)
        for j in words:
            embedding_j = model.embed_word(j)
            similarity = cos_similarity(embedding_i, embedding_j)
            print(f"{i} - {j}: {similarity:.4f}")


def demo_topk(words, k, model):
    for word in words:
        embedding = model.embed_word(word)
        ids = topk_ids(embedding, model.vectors, k)
        neighbours = [model.word(i) for i in ids]
        print(f"{word}: {', '.join(neighbours)}")

def demo_analogy(a, b, c, k, model):
    v = (
        model.embed_word(a)
        - model.embed_word(b)
        + model.embed_word(c)
    )
    ids = topk_ids(v, model.vectors, k)
    neighbours = [model.word(i) for i in ids]
    print(f"{a} - {b} + {c}: {', '.join(neighbours)}")