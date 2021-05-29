import streamlit as st
import pandas as pd
import numpy as np
import nltk
import os
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from tensorflow import keras
import math
from gensim.test.utils import datapath

from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

preds = 0
n = 0

max_s = [-1, 12, 6, 3, 3, 4, 4, 30, 60]

def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def e2s(essay_v, remove_stopwords):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

# def makeFeatureVec(words, model, num_features):
#     featureVec = np.zeros((num_features,),dtype="float32")
#     num_words = 0.
#     index2word_set = set(model.wv.index2word)
#     for word in words:
#         if word in index2word_set:
#             num_words += 1
#             featureVec = np.add(featureVec,model[word])        
#     featureVec = np.divide(featureVec,num_words)
#     return featureVec

# def getVecs(essays, model, num_features):
#     counter = 0
#     essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
#     for essay in essays:
#         essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
#         counter = counter + 1
#     return essayFeatureVecs
def getVecs(essays, model, num_features):
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        featureVec = np.zeros((num_features,),dtype="float32")
        num_words = 0.
        index2word_set = set(model.wv.index2word)
        for word in essay:
          if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])        
        featureVec = np.divide(featureVec,num_words)
        essayFeatureVecs[counter] = featureVec
        counter = counter + 1
    return essayFeatureVecs

def u_in():
  num = st.text_input("Enter question number: ")
  text = st.text_input("Enter your essay to be graded: ")

  return text,num

def header(url):
  st.markdown(f'<p style="font-size:50px;border-radius:5%;text-align:center;">{url}</p>', unsafe_allow_html=True)
#  st.markdown("__Auto Essay Grader__")    
def header1(url):
  st.markdown(f'<p style="font-size:26px;border-radius:2%;text-align:center;">{url}</p>', unsafe_allow_html=True)
header("Essgraster")
#st.subheader("Questions:")
header1("All the Best!!!!!!")


st.subheader("Prompts:")
with st.beta_expander("Topic 1: Effects computers have on people"):
  st.write("""  More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 

Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.
""")

with st.beta_expander("Topic 2: Censorship in the Libraries"):
  st.write("""  Write a persuasive essay to a newspaper reflecting your vies on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.""")

with st.beta_expander("Topic 3: Rough Road Ahead"):
  st.write("""  Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion.""")

with st.beta_expander("Topic 4: Winter Hibiscus"):
  st.write("""  Read the last paragraph of the story.

"When they come back, Saeng vowed silently to herself, in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again." 

Write a response that explains why the author concludes the story with this paragraph. In your response, include details and examples from the story that support your ideas.
""")

with st.beta_expander("Topic 5: Narciso Rodriguez"):
  st.write("""  Describe the mood created by the author in the memoir. Support your answer with relevant and specific information from the memoir.""")

with st.beta_expander("Topic 6: The Mooring Mast"):
  st.write("""  Based on the excerpt, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.""")

with st.beta_expander("Topic 7: Patience"):
  st.write("""  Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining.
Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.
""")

with st.beta_expander("Topic 8: Benefits of laughter"):
  st.write("""  We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.""")

df,num = u_in()
content = df

text_file = open("test.txt", "wt")
n = text_file.write(content)
text_file.close()

os.chdir('/content/drive/MyDrive/Cicada3301/Test files')
e_list = [doc for doc in os.listdir() if doc.endswith('.txt')]
e_notes =[open(File).read() for File in  e_list]

vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

vectors = vectorize(e_notes)
e_vectors = list(zip(e_list, vectors))

def check_plagiarism():
    plagiarism_results = set()
    global e_vectors
    for e_a, text_vector_a in e_vectors:
        new_vectors =e_vectors.copy()
        current_index = new_vectors.index((e_a, text_vector_a))
        del new_vectors[current_index]
        for e_b , text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            e_pair = sorted((e_a, e_b))
            score = (e_pair[0], e_pair[1],sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

sum = 0
flag = -1
for data in check_plagiarism():
  if data[0] == 'test.txt' or data[1] == 'test.txt':
    if sum < data[2]:
      sum = data[2]

if sum > 0.2:
  flag = 1

if num is "":
  num = 0

if 'test.txt' in os.walk('/content/'):
  os.remove("test.txt") 

model = Word2Vec.load("/content/drive/MyDrive/Cicada3301/model_weights/word2vec.model")

lstm_model = keras.models.load_model('/content/drive/MyDrive/Cicada3301/model_weights/final_lstm.h5')

if len(content) > 20:
  num_features = 300
  clean_test_essays = []
  clean_test_essays.append(essay_to_wordlist( content, remove_stopwords=True ))
  testDataVecs = getVecs( clean_test_essays, model, num_features )

  
  testDataVecs = np.array(testDataVecs)
  testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

  preds = lstm_model.predict(testDataVecs)
  


  if math.isnan(preds):
    preds = 0
  else:
    preds = np.round(preds)

  if preds < 0:
    preds = 0
  #else:
  #  preds = 0
p = int(preds)

if num is not 0 and p is not 0:  
  p = p*60/max_s[int(num)]
  if p>60:
    p = 60
  if sum < 0.3:
    st.write("Question no. : " + str(num))
    st.write("Final grade is " + str(p))
    st.write("Plagiarism across all reference documents : " + str(sum))

  else:
    st.write("Plagiarism across all reference documents : " + str(sum))
    st.write("Too much plagiarism...!!!!")
    st.write("Hence, grade will also be copied to 0")