import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

from Data.prepare_texts import save_to_txt
from Data.prepare_texts import remove_stop_words
from Data.prepare_texts import stemming
from bertopic import BERTopic

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

            sentences = [sentence for sentence in text.split('.') if len(sentence.split(' ')) > 2]
            docs.extend(sentences)
    

    topic_model = BERTopic(language='english', calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(docs)

    freq = topic_model.get_topic_info()
    print(freq.head(10))

    topic_model.visualize_barchart(top_n_topics=10).show()
