import numpy as np
from gensim.models.coherencemodel import CoherenceModel

def jaccard_distance(topics):

    topic_sets = [set(topic) for topic in topics]

    distance_matrix = np.zeros(shape=(len(topic_sets), len(topic_sets)))

    for i in range(len(topic_sets) - 1):
        for j in range(i, len(topic_sets)):
            
            #dist(A, B) = 1 - (A ∩ B) / (A ∪ B)
            nominator = len(topic_sets[i].intersection(topic_sets[j]))
            denominator = len(topic_sets[i].union(topic_sets[j]))
            dist = 1 - nominator / denominator
            
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    return np.mean(distance_matrix, axis=(0,1))

def coherence_score(texts, topics, dictionary):

    cm = CoherenceModel(
        texts=texts,
        topics=topics,
        dictionary=dictionary,
        coherence='c_v'
    )

    return cm.get_coherence()