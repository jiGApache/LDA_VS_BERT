import json
import matplotlib.pyplot as plt


def plot_paretto(model_to_show=['lda', 'bertopic']):

    if 'lda' in model_to_show:

        lda_jaccard = []
        lda_coherence = []

        with open(f'utils/lda_best_params.json') as json_file:
            data = json.load(json_file)
            for trial in data:
                lda_jaccard.append(trial['metrics'][0])
                lda_coherence.append(trial['metrics'][1])
        
        plt.plot(lda_jaccard, lda_coherence, '-^', label='lda')
        plt.axis
        plt.legend()


    if 'bertopic' in model_to_show:
        
        bert_jaccard = []
        bert_coherence = []

        with open(f'utils/bertopic_best_params.json') as json_file:
            data = json.load(json_file)
            for trial in data:
                bert_jaccard.append(trial['metrics'][0])
                bert_coherence.append(trial['metrics'][1])
        
        plt.plot(bert_jaccard, bert_coherence, '-^', label='bertopic')
        plt.legend()

    plt.xlabel("Jaccard Distance")
    plt.ylabel("Coherence score")

    plt.show()

plot_paretto()