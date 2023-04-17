import fitz
import re
import os

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

def save_to_txt():
    files = os.listdir('Data/clean')
    os.makedirs('Data/filtered', exist_ok=True)

    for file in files:
        with fitz.open(f'Data/clean/{file}') as doc: 
            
            file_text = ''
            for page_num in range(len(doc)):
                text = doc.load_page(page_num).get_text()

                # Fixing incorrect symbols
                text = re.sub(r'ﬁ', 'fi', text)
                text = re.sub(r'�', '', text)
                text = re.sub(r'ﬀ', 'ff', text)
                text = re.sub(r'ﬂ', 'fl', text)

                # Removing part that goes before "Abstract"
                text = re.sub(r'(.|\n)*abstract', ' ', text, flags=re.IGNORECASE)
                

                # Removing "Table" and "Figure" description
                text = re.sub(r'^table(.|\n)*?\.\n', '', text, flags=re.IGNORECASE)
                text = re.sub(r'^(Figure|Fig)(.|\n)*?\.\n', '', text)

                # Fixing word-hyphenations
                text = re.sub(r'-\n(?=.)', '', text) 
                
                # Fixing sentence-hyphenation
                text = re.sub(r'(?<=[^\.])\n(?=\S)', ' ', text) 
                
                # Removing brackets and its content
                text = re.sub(r'\s*\((.|\n)*?\)\s*', ' ', text) 
                text = re.sub(r'\s*\[(.|\n)*?\]\s*', ' ', text)
                text = re.sub(r'\s*\{(.|\n)*?\}\s*', ' ', text)

                # Removing math expressions
                text = re.sub(r'(\s\w?[xyXY]\w?[\s,]|[a-zA-Z]′| [a-zA-Z]([0-9]|,) |\S*[0-9][^\s.]*|:(.|\n)*?(?=\.)|\S*=\S*|\S*\|+\S*|\S*σ\S*|\S*\(+\S*|\S*\)+\S*|\S*λ\S*|\S*ϕ\S*|\S*∆\S*|\S*∇\S*|\S*∀\S*|\S*\+\S*|\S*±\S*|\S*ϵ\S*|\S*>\S*|\S*←\S*|\S*\\\S*|\S*\*\S*|\S*%\S*| − |\S*×\S*|\S*η\S*|\S*µ\S*|\S*log\S*|\S*∈\S*|\S*ρ\S*|\S*θ\S*|\S*α\S*|\S*Θ\S*|\S*Φ\S*|\S*⊙\S*|\S*lim\S*|⋄|†|•|\S*𝐴\S*|\S*˜\S*|\S*𝜋\S*|\S*Ø\S*|⌊|⌋|≤|′|\S*δ\S*|\S*𝑤\S*|\S*𝑏\S*|\S*𝜕\S*)|𝝀|\S*ω\S*|∞|ln', ' ', text) 
                
                # Removing separated variables with length 1
                text = re.sub(r'(?<= )\w(?= )', ' ', text)

                # Removing duplecated spaces
                text = re.sub(r' +', ' ', text)

                if re.search('references', text, re.IGNORECASE):
                    text = re.sub(r'references(.|\n)*', '', text, flags=re.IGNORECASE)
                    file_text += text
                    break
                else:
                    file_text += text

            file_lines = file_text.split('\n')
            file_lines = [line for line in file_text.split('\n') if len(line.split(' ')) > 4]
            file_text = ' '.join(file_lines)

            file = open(f'Data/filtered/{file[:file.rfind(".")]}.txt', 'w', encoding='utf-8')
            file.write(file_text)
            file.close()


def remove_stop_words(text):
    
    stop = set(stopwords.words('english'))
    stop.add('learn')
    stop.add('use')
    stop.add('task')
    stop.add('train')

    sentences = text.split('.')
    new_sentences = []
    for sentence in sentences:
        new_sentences.append(' '.join([word.lower() for word in sentence.split() if word.lower() not in stop]))

    return '.'.join(new_sentences)


def stemming(text):

    snowball_stemmer = SnowballStemmer('english')

    sentences = text.split('.')
    new_sentences = []
    for sentence in sentences:
        word_tokens = nltk.word_tokenize(sentence)
        stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
        new_sentences.append(' '.join(stemmed_word))

    return '.'.join(new_sentences)