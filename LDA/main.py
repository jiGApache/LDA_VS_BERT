import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

from Data.prepare_texts import retrieve_data
from utils.training import get_best_lda
from utils.training import pretty_lda_topics
from utils.training import coherence_score
from utils.training import jaccard_distance
from utils.training import RANDOM_SEED
# from utils.visualize_paretto import plot_paretto
import gensim
from pprint import pprint



if __name__ == '__main__':

    texts, dictionary, corpus = retrieve_data()
    lda_best_trials = get_best_lda()

    for best_trial in lda_best_trials:

        best_lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=best_trial.params['num_topics'],
            passes=best_trial.params['passes'],
            random_state=RANDOM_SEED
        )

        print(f'For params {best_trial.params}:')

        print('Topics:')
        pprint(best_lda_model.print_topics(num_words=best_trial.params['num_words']))

        topics = pretty_lda_topics(best_lda_model, best_trial.params['num_words'])
        print(f'Coherence score: {coherence_score(texts, topics, dictionary)}')
        print(f'Jaccard distance: f{jaccard_distance(topics)}')
        print()

    # plot_paretto(model_to_show=['lda'])