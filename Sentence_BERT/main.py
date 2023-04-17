import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

from Data.prepare_texts import save_to_txt
from Data.prepare_texts import remove_stop_words
from Data.prepare_texts import stemming
from bertopic import BERTopic

from gensim import corpora
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel

import re

if __name__ == '__main__':

    if not os.path.exists('Data/filtered') or len(os.listdir('Data/filtered')) == 0:
        save_to_txt()

    docs = []
    doc_list = os.listdir('Data/filtered')
    for doc_name in doc_list:
        with open(f'Data/filtered/{doc_name}', 'r', encoding='utf_8') as doc:
            text = doc.read()
            text = remove_stop_words(text)
            text = stemming(text)


            text = re.sub(r',', '', text)

            
            sentences = [sentence for sentence in text.split('.') if len(sentence.split(' ')) > 2]
            final_sentences = []
            while len(sentences) > 0:
                final_sentences.append(' '.join(sentences[:1]))
                del sentences[:1]
            docs.extend(final_sentences)

    
    # Finding topics
    topic_model = BERTopic(language='english')
    topics = topic_model.fit_transform(docs)

    freq = topic_model.get_topic_info().head(10)
    # print(freq)

    topic_model.visualize_barchart(top_n_topics=10).show()



    #Calculating coherence score
    topics = freq['Name'].to_list()[1:]
    topics = [topic.split('_')[1:] for topic in topics]

    texts = [[word for word in doc.split()] for doc in docs]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    cm = CoherenceModel(
        texts=texts,
        topics=topics,
        corpus=corpus,
        coherence='c_v',
        dictionary=dictionary
    )
    
    print(f'Coherence score: {cm.get_coherence()}')