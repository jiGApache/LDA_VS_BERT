import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

from Data.prepare_texts import retrieve_data
from utils.training import get_best_bertopic
from Data.prepare_texts import get_docs
from utils.training import pretty_bertopic_topics
from utils.training import coherence_score
from utils.training import jaccard_distance
from utils.training import RANDOM_SEED
# from utils.visualize_paretto import plot_paretto
from bertopic import BERTopic
from umap import UMAP

if __name__ == '__main__':

    texts, dictionary, _ = retrieve_data()
    bertopic_best_trials = get_best_bertopic()

    umap_model = UMAP(n_neighbors=15,
                        n_components=5,
                        min_dist=0.0,
                        metric='cosine',
                        random_state=RANDOM_SEED)


    for best_trial in bertopic_best_trials:

        best_topic_model = BERTopic(
            language='english',
            min_topic_size=best_trial.params['min_topic_size'],
            umap_model=umap_model
        )
        
        docs = get_docs()
        best_topic_model.fit_transform(docs)

        print(f'For params {best_trial.params}')
        freq = best_topic_model.get_topic_info().head(best_trial.params['num_topics'])
        print(freq)

        # best_topic_model.visualize_barchart(
        #     top_n_topics=bertopic_best_trials.params['num_topics'],
        #     n_words=bertopic_best_trials.params['num_words']
        # ).show()

        topics = pretty_bertopic_topics(best_topic_model, best_trial.params['num_topics'], best_trial.params['num_words'])
        print(f'Coherence score: {coherence_score(texts, topics, dictionary)}')
        print(f'Jaccard distance: f{jaccard_distance(topics)}')
        print()

    # plot_paretto(model_to_show=['bertopic'])