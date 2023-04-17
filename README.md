# Topic modeling: BERT vs LDA

## Описание проекта

Целью данного проекта является сравнение двух подходов для решения задачи тематического моделирования научных статей: BERT и LDA.

### LDA
Для реализации модели LDA используется готовая модель из библиотеки ```gensim```, которая помимо этого предоставляет различные  инструменты для создания корпусов текстовых документов и  препроцессинга текстов.

### BERT
Для реализации тематического моделирования с помощью модели BERT исопльзуется готовая библиотека BERTopic, которая в своей основе имеет ряд ключевых компонентов:
- предобученная модель [sentence-transformer](https://www.sbert.net/) для вычисления эмбеддингов предложений текста
- алгоритм UMAP для уменьшения размерности полученных эмбеддингов предложений
- модель HDBSCAN для кластеризации полученных эмбеддингов
- Выделение топиков из каждого полученного кластера при помощи техники c-TF-IDF

## Используемые данные
### Описание
В качестве исходных данных были использованы 10 научных [статей](#источники) из смежных областей. Примерное распределение тем в каждой статье можно увидеть в следующей таблице:
№ Статьи | Few-Shot | Time Series | Siamese Network | Medicine | Convolutioinal Networks 
--- | :---: | :---: | :---: | :---: | :---:
1  | + | | + | | +
2  | | + | | + | +
3  | | | + | | +
4  | + | | | | +
5  | + | + | | + | +
6  | | | + | + | +
7  | + | + | + | + | + 
8  | | | | + | +
9  | | | | + | +
10  | + | | + | | +

### Предобработка данных
В качестве предварительной обработки исходных текстов были проведены следующие действия:
- Удаление частей статей, находящихся перед заголовком "Abstract" и после "References"
- Удаление подписей к рисункам и таблицам
- Восстановление переносов слов и разрывов предложений
- Удаление содержимого всех скобок, а также всех математических выражений (формул, отдельных переменных и операторов, чисел и т.д.)
- Удаление повторяющихся пробелов, коротких предложений и знаков препинания (за исключением знака конца предложения)
- Удаление "стоп" слов и стемминг

После этого на вход модели BERTopic подавался список предложений из всех текстов, а LDA - список отдельно токенизированных текстов.

## Результаты

### LDA

После запуска модели были получены следующие результаты:
```
[(0,
  '0.001*"model" + 0.001*"data" + 0.001*"class" + 0.001*"method" + 0.001*"set" '
  '+ 0.001*"dataset" + 0.001*"network" + 0.001*"few-shot" + 0.001*"sampl" + '
  '0.001*"differ"'),
 (1,
  '0.001*"ecg" + 0.001*"model" + 0.001*"signal" + 0.001*"network" + '
  '0.001*"data" + 0.001*"set" + 0.001*"layer" + 0.001*"featur" + 0.001*"class" '
  '+ 0.001*"sampl"'),
 (2,
  '0.017*"network" + 0.012*"class" + 0.012*"model" + 0.011*"layer" + '
  '0.011*"time" + 0.011*"set" + 0.011*"imag" + 0.010*"convolut" + '
  '0.009*"number" + 0.009*"dataset"'),
 (3,
  '0.018*"ecg" + 0.017*"method" + 0.016*"represent" + 0.015*"signal" + '
  '0.014*"network" + 0.012*"classif" + 0.011*"beat" + 0.010*"differ" + '
  '0.009*"layer" + 0.009*"dataset"'),
 (4,
  '0.024*"ecg" + 0.021*"signal" + 0.013*"model" + 0.010*"perform" + '
  '0.010*"beat" + 0.010*"detect" + 0.009*"work" + 0.008*"dataset" + '
  '0.008*"featur" + 0.008*"r-peak"'),
 (5,
  '0.019*"ecg" + 0.015*"data" + 0.015*"classif" + 0.012*"model" + '
  '0.011*"accuraci" + 0.011*"set" + 0.010*"method" + 0.010*"time" + '
  '0.010*"sampl" + 0.009*"propos"'),
 (6,
  '0.018*"few-shot" + 0.013*"method" + 0.013*"’" + 0.013*"point" + '
  '0.012*"class" + 0.011*"model" + 0.010*"loss" + 0.010*"queri" + '
  '0.009*"batch" + 0.009*"optim"')]
  ```
  Среди полученных топиков можно увидеть следующие закономерности:
  - 0, 6 топики содержат информацию о нейронных сетях и подходе Few-Shot
  - 1, 3 топики содержат информацию о медицине (ЭКГ), а также применении нейронных сетей
  - 2 топик содержит информацию о временных рядах и сверточных нейронных сетях
  - 4 топик содержит только информацию о медицинских данных
  - 5 топик содержит информацию о временных рядах ЭКГ

### BERTopic

Поскольку преобразования, получаемые алгоритмом UMAP недетерминированны, то результаты работы модели могут отличаться от запуска к запуску. Для демонстрации работы модели ниже представлены результаты, полученные после нескольких запусков:

![Alt text](https://github.com/jiGApache/BERT_VS_LDA/raw/main/images/1.png)

![Alt text](https://github.com/jiGApache/BERT_VS_LDA/raw/main/images/2.png)

Анализ полученных диаграмм позволяет сказать, что:
- были получены все основные топики, содержащиеся в анализируемых статьях (Few-Shot, Time Series, Siamese Networks, Medicine, Convolutional Networks)
- были получены топики, характеризующие наполнение статей (сравнение результатов, описание данных, и т.д.)
- получены топики в смежных и не обозначенных в начале темах (анализ данных ЭКГ сверточными сетями, применение подходов глубоких нейронных сетей)
  

## Выводы

Можно сказать, что оба подхода справляются с решением задачи темаического моделирования научных статей.

Однако, модель LDA проявила себя слабее модели BERTopic, поскольку полученные с ее помощью топики не смогли выделить характерные темы каждой статьи в полной мере - все они включают в себя несколько базовых тем и плохо поддаются интерпретации.

В то же время модель BERTopic смогла выделить как все основные темы, присущие статьям, так и выделить несколько топиков, включающих в себя основные темы. Более того, были выделены топики, которые не удалось обозначить при предварительной оценке статей.

## Источники
1. Few-Shot Learning Through an Information Retrieval Lens - https://arxiv.org/abs/1707.02610
2. ECG Heartbeat Classification: A Deep Transferable Representation - https://arxiv.org/pdf/1805.00794.pdf
3. Fully Convolutional Siamese Networks for Change Detection - https://arxiv.org/abs/1810.08462
4. Meta-Transfer Learning for Few-Shot Learning - https://arxiv.org/abs/1812.02391
5. Meta-Learning for Few-Shot Time Series Classification - https://arxiv.org/abs/1909.07155
6. EDITH : ECG biometrics aided by Deep learning for reliable Individual auTHentication - https://arxiv.org/abs/2102.08026
7. Similarity Learning based Few Shot Learning for ECG Time Series Classification - https://ieeexplore.ieee.org/document/9647357
8. Automated Detection of Arrhythmias Using Different Intervals of Tachycardia ECG Segments with Convolutional Neural Network - https://www.researchgate.net/publication/315821873_Automated_Detection_of_Arrhythmias_Using_Different_Intervals_of_Tachycardia_ECG_Segments_with_Convolutional_Neural_Network
9. ECG Arrhythmia Classification Using STFT-Based Spectrogram and Convolutional Neural Network - https://ieeexplore.ieee.org/document/8759878
10. Siamese Neural Networks for One-shot Image Recognition - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf