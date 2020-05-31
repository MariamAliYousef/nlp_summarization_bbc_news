# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:30:26 2020

@author: Mariam
"""

import pandas as pd
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# read dataset file it contains 2225 rows the text data in text column
dataset = pd.read_excel("E:/MyProgram/arb/nlp_project/dataset/news_data/news_dataset.xlsx")
# stopwords for english language such as  the - a - an - ...  
stopWords = set(stopwords.words('english'))

# function to make preprocessing for news articless
def preprocess_news():
    # Tokenization
    tokens = [word_tokenize(sents) for sents in dataset['text']]
    # preprocessing for text
    
    #  Make all words in lower case and remove stopwords
    words = []
    for token in tokens:
        t = []
        for word in token:
            if word.lower() not in stopWords:
                t.append(word)
        words.append(t)
    
    # remove not alphabet characters
    formatted_text_list = []
    for word_list in words:
        f_text = []
        for word in word_list:
            formatted_text = re.sub('[^a-zA-Z]', ' ', word )
            formatted_text = re.sub(r'\s+', ' ', formatted_text)
            if formatted_text == ' ':
                continue
            else:
                f_text.append(formatted_text)
        formatted_text_list.append(f_text)
        
    # stemming
    lemmatization_list = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for f_text_list in formatted_text_list:
        stem_list = []
        for word in f_text_list:
            stem_list.append(wordnet_lemmatizer.lemmatize(word))
        lemmatization_list.append(stem_list)
    
    return lemmatization_list

# calculate the weighted frequency for words in each article
def calc_weighted_freq_word():
    lemmatization_list = preprocess_news()
    word_frequencies = {}
    n = 0
    for lemma_list in lemmatization_list:
        word_freq = {}
        for word in lemma_list:
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1
        word_frequencies.update({n : word_freq})
        n += 1
    return word_frequencies

# find maximum frequency in every dictionary and recalculate the frequency for each word in each news 
# by divie the weighted freq for each word by the maximum freq for each article
def calc_maxi_freq_for_words():
    word_frequencies = calc_weighted_freq_word()
    maximum_freqs = []
    for freq_dict in word_frequencies.keys():
            maximum_freqs.append(max(word_frequencies[freq_dict].values()))

    # calculate the freqencies for each word          
    for ind in range(len(maximum_freqs)):
        for w in word_frequencies[ind].keys():
            word_frequencies[ind][w] = word_frequencies[ind][w] / maximum_freqs[ind]
    return word_frequencies


# calculating sentences score
def sents_scores():
    word_frequencies = calc_maxi_freq_for_words()
    sentence_list = [sent_tokenize(sentence) for sentence in dataset['text']]
    
    sentence_scores = {}
    
    for num in range(len(sentence_list)):
        sentence_score = {}
        for sent in sentence_list[num]:
            for word in word_tokenize(sent.lower()):
                if len(sent.split(' ')) < 30:
                   if word in word_frequencies[num].keys():
                       if sent not in sentence_score.keys():
                           sentence_score[sent] = word_frequencies[num][word]
                       else:
                           sentence_score[sent] += word_frequencies[num][word]
        sentence_scores.update({num : sentence_score})
    return sentence_scores


# to get the 6th largest sentences
def find_6thlargest_sents():
    sentence_scores = sents_scores()
    sents_sorted = []
    for data in sentence_scores.values():
        sents_sorted.append(sorted(data.items() , reverse=True, key=lambda x: x[1]))
    
    summary_list = []
    for index in range(len(sents_sorted)):
        summary = ""
        if len(sents_sorted[index]) >= 10 or (len(sents_sorted[index]) < 10 and len(sents_sorted[index]) > 6):
            for i in range(6):
                summary += ' ' + sents_sorted[index][i][0]
                
        elif len(sents_sorted[index]) <= 6:
            for i in range(len(sents_sorted[index])):
                summary += ' ' + sents_sorted[index][i][0]
        summary_list.append([summary])
    return summary_list

# system summary
largest_6th_sents = find_6thlargest_sents()
for index in range(len(largest_6th_sents)):
    print('Summary for article ', index+1, ' : ', largest_6th_sents[index])
    print('')



# summary for test system
reference = pd.read_excel("E:/MyProgram/arb/nlp_project/dataset/summary/news_summarizes_dataset.xlsx")

from rouge import Rouge 

rouge = Rouge()
rouge_scores = {}
for i in range(len(largest_6th_sents)):
    sent = str(largest_6th_sents[i])
    scores = rouge.get_scores(sent, reference['text'][i])
    rouge_scores.update({i : scores})


print(rouge_scores)
