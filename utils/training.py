import optuna
from Data.prepare_texts import retrieve_data
from Data.prepare_texts import get_docs
from utils.metrics import jaccard_distance
from utils.metrics import coherence_score
import gensim
from bertopic import BERTopic
from umap import UMAP
import json

RANDOM_SEED = 42

# LDA ######################################################################

def pretty_lda_topics(model, num_words):

    topics = []
    for tpl in model.print_topics(num_words=num_words):
        topic = []
        for word_prob in tpl[1].split(' + '):
            topic.append(word_prob.split('*')[1][1:-1])
        topics.append(topic)

    return topics

def lda_objective(trial):

    num_words = trial.suggest_int('num_words', 4, 8)
    num_topics = trial.suggest_int('num_topics', 5, 12)
    passes = trial.suggest_int('passes', 10, 150)
    
    texts, dictionary, corpus = retrieve_data()

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=RANDOM_SEED
    )

    topics = pretty_lda_topics(lda_model, num_words)

    return jaccard_distance(topics), coherence_score(texts, topics, dictionary)

def get_best_lda():
    
    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(lda_objective, n_trials=50)
    
    best_trials = study.best_trials
    total_data = []
    
    for trial in best_trials:
        trial_data = {}
        trial_data['metrics'] = trial.values
        trial_data['params'] = trial.params
        total_data.append(trial_data)

    with open('utils/lda_best_params.json', 'w') as file:
        file.write(json.dumps(total_data))

    return best_trials

############################################################################



# BERTopic #################################################################

def pretty_bertopic_topics(model, num_topics, num_words):
    
    freq = model.get_topic_info().head(num_topics)

    topics = freq['Name'].to_list()[1:]
    topics = [topic.split('_')[1:num_words+1] for topic in topics]

    return topics

def bertopic_objective(trial):

    min_topic_size = trial.suggest_int('min_topic_size', 8, 10)
    num_topics = trial.suggest_int('num_topics', 4, 8)
    num_words = trial.suggest_int('num_words', 4, 8)

    umap_model = UMAP(n_neighbors=15,
                        n_components=5,
                        min_dist=0.0,
                        metric='cosine',
                        random_state=RANDOM_SEED)
    
    docs = get_docs()
    texts, dictionary, _ = retrieve_data()

    topic_model = BERTopic(
        language='english',
        min_topic_size=min_topic_size,
        umap_model=umap_model
    )

    topic_model.fit_transform(docs)
    topics = pretty_bertopic_topics(topic_model, num_topics, num_words)

    return jaccard_distance(topics), coherence_score(texts, topics, dictionary)

def get_best_bertopic():

    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(bertopic_objective, n_trials=2)

    best_trials = study.best_trials
    total_data = []
    
    for trial in best_trials:
        trial_data = {}
        trial_data['metrics'] = trial.values
        trial_data['params'] = trial.params
        total_data.append(trial_data)

    with open('utils/bertopic_best_params.json', 'w') as file:
        file.write(json.dumps(total_data))

    return best_trials