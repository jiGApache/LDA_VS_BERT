import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

from Data.prepare_texts import save_to_txt
from Data.prepare_texts import remove_stop_words
from Data.prepare_texts import stemming

import re
import gensim
from gensim import corpora

from pprint import pprint

if __name__ == '__main__':
    
    if not os.path.exists('Data/filtered') or len(os.listdir('Data/filtered')) == 0:
        save_to_txt()

    docs = []
    doc_list = os.listdir('Data/filtered')
    for doc_name in doc_list:
        with open(f'Data/filtered/{doc_name}', 'r', encoding='utf_8') as doc:
            text = doc.read()
            text = stemming(text)
            text = remove_stop_words(text)
            
            text = re.sub(r',', '', text)
            text = re.sub(r'\.', ' ', text)

            docs.append(text)

    texts = [[word for word in doc.split()] for doc in docs]

    dictionary = corpora.Dictionary(texts)
    
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=7,
        passes=10,
        random_state=42
    )

    pprint(lda_model.print_topics(num_words=10))