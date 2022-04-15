# Importing packages
import pandas as pd
import os
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.gensim_models
import pyLDAvis
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import gensim.corpora as corpora# Create Dictionary
import gensim
from gensim.utils import simple_preprocess
import nltk
import matplotlib.pyplot as plt
from math import log
import numpy as np




#Read in survey data
os.chdir('..')
raw_data = pd.read_csv('C:/Users/vahan/OneDrive/Desktop/iterview_prep_docs/hgc_survey.csv')
raw_data.head()

#Remove blanks
raw_data=raw_data.dropna()

#Text formatting
raw_data['question_processed'] = raw_data['question'].map(lambda x: re.sub('[,\.!?]', '', x))
raw_data['question_processed'] = raw_data['question_processed'].map(lambda x: x.lower())


#Generate a wordcloud to gauge word usages
strings = ','.join(list(raw_data['question_processed'].values))
wordcloud_test = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud_test.generate(strings)
wordcloud_test.to_image()

#Remove stop words
nltk.download('stopwords')

stop_words_remove = stopwords.words('english')
stop_words_remove.extend(['from',
                   'does',
                   'know',
                   'what',
                   'subject', 
                   're', 
                   'love', 
                   'use', 
                   'like',
                   'feel',
                   'feels',
                   'also',
                   'would',
                   'and',
                   'really',
                   'bit',
                   'could',
                   'but',
                   'felt',
                   'think',
                   'time',
                   'times',
                   'little',
                   'game'])


#Functions to stem words and remove stop words
def stem_words(texts):
    for t in texts:
        yield(gensim.utils.simple_preprocess(str(t), deacc=True))
        
        
def remove_stopwords(stp_words):
    return [[x for x in simple_preprocess(str(doc)) 
             if x not in stop_words_remove] for doc in stp_words]

data = raw_data.question.values.tolist()
data_words = list(stem_words(data))
data_words = remove_stopwords(data_words)

#Create the word dictionary for the model
id2_word = corpora.Dictionary(data_words)
texts_data = data_words
corpus = [id2_word.doc2bow(x) for x in texts_data]

#Actual model run. Here we examine the coherence level to determine appropriate number of topics
coherence = []
for k in range(5,25):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2_word,
                                           num_topics=k)
    
    cm = gensim.models.coherencemodel.CoherenceModel(
         model=lda_model, texts=texts_data,
         dictionary=id2_word, coherence='c_v')   
                                                
    coherence.append((k,cm.get_coherence()))
    
plot_list = [(elem1, log(elem2)) for elem1, elem2 in coherence]

zip(*plot_list)
plt.plot(*zip(*plot_list))
plt.show()
 
#Typically, as the coherence spikes initially then at the point where it drops, is the number we like to see.
#Normally, you do not want more than 15 topics as that is not a great business use case.   
 
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2_word,
                                        num_topics=11)   
doc_lda = lda_model[corpus]




#Once we are done running the model, we then tag each record with the top topic that is the most likely representation. 
def topic_tagging(ldamodel=lda_model, corpus=corpus, texts=data):
    
    topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, top_topic) in enumerate(row):
            if j == 0:
                xt = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in xt])
                topics_df = topics_df.append(pd.Series([int(topic_num), round(top_topic,4), topic_keywords]), ignore_index=True)

    topics_df.columns = ['top_topic', 'contri', 'keywords']
    contents = pd.Series(texts)
    topics_df = pd.concat([topics_df, contents], axis=1)
    return(topics_df)


df_top_topics = topic_tagging(ldamodel=lda_model, corpus=corpus, texts=data)


#Check list of topics
topics_list=pd.DataFrame({'topics':df_top_topics['top_topic'].unique(),'terms':df_top_topics['keywords'].unique()})


#Create mock data for engagement. Skew as needed. 
df_top_topics['game_sessions'] = np.random.randint(1, 100, df_top_topics.shape[0])
df_top_topics['hanning']  = np.hanning(2166)
df_top_topics['skewed'] = df_top_topics.game_sessions * df_top_topics.hanning+1

sessions_data=df_top_topics.groupby('top_topic', as_index=False)['game_sessions'].mean()


df_top_topics.to_csv('C:/Users/vahan/OneDrive/Desktop/iterview_prep_docs/hgc_tagged_topics.csv')



#Plot values if needed to check
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
topics_graph = list(df_top_topics.top_topic)
session_graph = list(df_top_topics.skewed)
ax.bar(topics_graph,session_graph)
plt.show()


#Plot in HTML
pyLDAvis.enable_notebook()
LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2_word)

pyLDAvis.save_html(LDAvis_prepared, 'C:/Users/x.vahan.mouradian/Desktop/LDA_Visualization.html')

