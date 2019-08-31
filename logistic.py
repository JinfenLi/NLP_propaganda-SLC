from sklearn.linear_model import LogisticRegression
import glob
import os.path
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression,f_classif
import nltk
import pandas as pd
import json
from textstat.textstat import textstat
from nltk.tokenize import sent_tokenize
import collections
import clean_data



os.environ['KMP_DUPLICATE_LIB_OK']='True'


def read_articles_from_file_list(folder_name, file_pattern="*.txt"):

    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with open(filename, "r", encoding="utf-8") as f:
            for sentence_id, row in enumerate(f.readlines(), 1):
                sentence_list.append(row.rstrip())
                article_id_list.append(article_id)
                sentence_id_list.append(str(sentence_id))
    return article_id_list, sentence_id_list, sentence_list


def read_predictions_from_file(filename):

    articles_id, sentence_id_list, gold_labels = ([], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, sentence_id, gold_label = row.rstrip().split("\t")
            articles_id.append(article_id)
            sentence_id_list.append(sentence_id)
            gold_labels.append(gold_label)
    return articles_id, sentence_id_list, gold_labels


def read_predictions_from_file_list(folder_name, file_pattern):

    gold_file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles_id, sentence_id_list, gold_labels = ([], [], [])
    for filename in sorted(gold_file_list):
        art_ids, sent_ids, golds = read_predictions_from_file(filename)
        articles_id += art_ids
        sentence_id_list += sent_ids
        gold_labels += golds
    return articles_id, sentence_id_list, gold_labels



def whatabout(X):
    if "what about" in X :
        feature = [0, 1]
    else:
        feature = [1, 0]
    return feature
def howdareyou(X):
    if "thank you" in X:
        feature = [0, 1]
    else:
        feature = [1, 0]
    return feature
def timefor(X):
    if "time for" in X and "at the time" not in X:
        feature=[0,1]
    else:
        feature=[1,0]
    return feature
def hiter(X):
    if "Monica Lewinsky" in X:
        feature = [0, 1]
    else:
        feature = [1, 0]
    return feature
def eitheror(X):
    if "either" in X and "or" in X:
        feature = [0, 1]
    else:
        feature = [1, 0]
    return feature

# def sooner(X):
#     if "sooner or later" in X:
#         feature = [1]
#     else:
#         feature = [0]
#     return feature

import re
def slogan(X):
    group1 = re.findall('‘(.*)’',X)
    group2 = re.findall('“(.*)”',X)
    group3 = re.findall('\"(.*)\"', X)
    g=group1+group2+group3
    flag=False
    for gg in g:
        if gg.istitle() or gg.isupper() and "Tommy Robinson" not in X:
            # print(gg)
            flag=True
    if flag:
        feature = [0, 1]
    else:
        feature = [1, 0]
    return feature


def readSubjectivity():
    path = 'subjclueslen1-HLTEMNLP05.tff'
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = []
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        sldict.append([strength, posTag, isStemmed, polarity])
        # sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

def SL_features(document, SL):

    document_words = set(document)
# count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    positivecount = 0
    negativecount = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            positivecount = weakPos + 2*strongPos
            negativecount = weakNeg + 2*strongNeg

    return positivecount,negativecount





def miss_voc(voc,text):
    for t in nltk.word_tokenize(text):
        if t in voc:
            return [1]
    return [0]

def stop(text):
    stopwords = []
    with open('lexicons/stopwords.txt', 'r') as f:
        for line in f:
            stopwords.append(line.strip())
    slangs = {}
    with open('lexicons/slangs.txt', 'r') as file:
        for line in file:
            splitted = line.strip().split(',', 1)
            slangs[splitted[0]] = splitted[1]
    negated = {}
    with open('lexicons/negated_words.txt', 'r') as f:
        for line in f:
            splitted = line.strip().split(',', 1)
            negated[splitted[0]] = splitted[1]

    text = re.sub('(!){2,}', ' <!repeat> ', text)
    text = re.sub('(\?){2,}', ' <?repeat> ', text)
    tokens = nltk.word_tokenize(text)

    temp=[]
    for word in tokens:
        if word in slangs:
            temp += slangs[word].split()
        elif word in negated:tokens = nltk.word_tokenize(text)

        temp = []
        for word in tokens:
            if word in slangs:
                temp += slangs[word].split()
            elif word in negated:
                temp += negated[word].split()
            else:
                temp.append(word)
        tokens = temp


        tokens = [word for word in tokens if word not in stopwords]


    return ' '.join(tokens)




def gettingFeatures(plainText):

    LIWC_JSON = open("LIWC2015_Lower_i.json", 'r')
    LIWC = json.load(LIWC_JSON)
    plainText = plainText.lower()
    syllables = textstat.syllable_count(plainText)
    sentences = len(sent_tokenize(plainText))

    #Count all punctuation
    AllPunc = 0
    punc = "!\',./:;<=>?_`{|}~"
    #"!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    cd = {c:val for c, val in collections.Counter(plainText).items() if c in punc}
    for x in cd.values():
        AllPunc = AllPunc + x

    # Number of commas
    Comma = 0
    Comma = plainText.count(",")
    # Number of question marks
    QMark = 0
    QMark = plainText.count("?")
    # Number of colons
    Colon = 0
    Colon = plainText.count(":")
    # Number of dash
    Dash = 0
    Dash = plainText.count("-")
    # Number of Parenth
    Parenth = 0
    Parenth = plainText.count("(") + plainText.count(")")

    # Replace all the punctuations with empty space
    punctuation = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punctuation:
        if p != '\'':
            plainText = plainText.replace(p, ' ')

    # '\n' would affect the result -> '\n'i, where i is the first word in a paragraph
    plainText = plainText.replace('\n', ' ')
    plainText = plainText.replace('\t', ' ')
    text = plainText.split(" ")
    while text.count(''): text.remove('')

    # Total number of words in the text
    wordCount = len(text)
    if wordCount==0:
        return [0]*54

    try:
        #ReadabilityScore
        readabilityScore = 206.835 - 1.015 * (wordCount / sentences) - 84.6 * (syllables / wordCount)
        #ReadabilityGrade
        ReadabilityGrade = 0.39 * (wordCount / sentences) + 11.8 * (syllables / wordCount) - 15.59
    except:
        readabilityScore = 0
        ReadabilityGrade = 0
    #Punctuations
    AllPunc = AllPunc / wordCount * 100
    Comma = Comma / wordCount * 100
    QMark  =QMark / wordCount * 100
    Colon = Colon / wordCount * 100
    Dash = Dash / wordCount * 100
    Parenth = Parenth / wordCount * 100
    #Direction Count
    DirectionCount  = 0
    DirectionCount = text.count("here") + text.count("there") + plainText.count("over there") + text.count("beyond") + text.count("nearly") + text.count("opposite") + text.count("under") + plainText.count("to the left") + plainText.count("to the right") + plainText.count("in the distance")
    #Exemplify count
    Exemplify = 0
    Exemplify = text.count("chiefly") + text.count("especially") + plainText.count("for instance") + plainText.count("in particular") + text.count("markedly") + text.count("namely") + text.count("particularly")+ text.count("incluiding") + text.count("specifically") + plainText.count("such as")
    #Analytical thinking
    #Analytic = 0 #LIWC Analysis
    #Aunthenticity
    #Authentic  = 0 #LIWC Analysis
    #Emotional tone
    #Tone = 0 #LIWC Analysis
    try:
        #words per sentence (average)
        WPS = 0
        numOfWords = len(text)
        numOfSentences = sentences
        WPS = numOfWords / numOfSentences
    except:
        WPS = 0
    #Six letter words
    Sixltr = 0
    # words = plainText.split()
    letter_count_per_word = {w:len(w) for w in text}
    for x in letter_count_per_word.values():
        if x >= 6:
            Sixltr = Sixltr + 1
    Sixltr = Sixltr / wordCount * 100
    #Function words
    function = 0
    #Pronouns
    pronoun = 0
    pronoun = len([x for x in text if x in LIWC["Pronoun"]])/wordCount * 100
    #Personal pronouns
    ppron = 0
    ppron = len([x for x in text if x in LIWC["Ppron"]])/wordCount * 100
    #I
    feature_i = 0
    feature_i = len([x for x in text if x in LIWC["i"]])/wordCount * 100
    #You
    you = 0
    you = len([x for x in text if x in LIWC["You"]])/wordCount * 100
    #Impersonal pronoun "one" / "it"
    ipron = 0
    # ipron = (text.count("one") + text.count("it"))/wordCount
    ipron = len([x for x in text if x in LIWC["ipron"]])/wordCount * 100
    #Prepositions
    prep = 0
    # prep = len([ (x,y) for x, y in result if y  == "IN" ])/wordCount
    prep = len([x for x in text if x in LIWC["Prep"]])/wordCount * 100
    # Verb
    verb = 0
    verb = len([x for x in text if x in LIWC["Verb"]])/wordCount * 100
    #Auxiliary verbs do/be/have
    auxverb = 0
    auxverb = len([x for x in text if x in LIWC["Auxverb"]])/wordCount * 100
    #Negations
    negate = 0
    negate = len([x for x in text if x in LIWC["Negate"]])/wordCount * 100
    #Count interrogatives
    #interrog = 0 #LICW Analysis
    #Count numbers
    number = 0
    number = len([x for x in text if x in LIWC["Number"]])/wordCount * 100

    # #tf-idf
    # tfidf = 0
    # response = tfidfV.transform([plainText])
    # feature_names = tfidfV.get_feature_names()
    # for col in response.nonzero()[1]:
    #     tfidf += response[0, col]

    # Transitional words
    transitional_words = 0
    sum_t1 = 0
    sum_t2 = 0

    t1 = ['also', 'again', 'besides', 'furthermore', 'likewise', 'moreover', 'similarly','accordingly', 'consequently', 'hence', 'otherwise'
    , 'subsequently', 'therefore', 'thus', 'thereupon', 'wherefore','contrast', 'conversely', 'instead', 'likewise', 'rather', 'similarly'
    , 'yet', 'but', 'however', 'still', 'nevertheless','here', 'there', 'beyond', 'nearly', 'opposite', 'under', 'above','incidentally'
    ,'chiefly', 'especially', 'particularly', 'singularly','barring', 'beside', 'except', 'excepting', 'excluding', 'save','chiefly', 'especially'
    , 'markedly', 'namely', 'particularly', 'including' , 'specifically','generally', 'ordinarily', 'usually','comparatively', 'correspondingly'
    , 'identically', 'likewise', 'similar', 'moreover','namely','next', 'then', 'soon', 'later', 'while', 'earlier','simultaneously', 'afterward'
    ,'briefly', 'finally']

    t2 = ['as well as', 'coupled with', 'in addition', 'as a result', 'for this reason', 'for this purpose', 'so then','by the same token', 'on one hand'
    , 'on the other hand', 'on the contrary', 'in contrast', 'over there', 'to the left', 'to the right', 'in the distance','by the way','above all'
    , 'with attention to','aside from', 'exclusive of', 'other than', 'outside of','for instance', 'in particular', 'such as','as a rule', 'as usual'
    , 'for the most part', 'generally speaking','for example', 'for instance', 'for one thing', 'as an illustration', 'illustrated with', 'as an example'
    , 'in this case','comparatively', 'correspondingly', 'identically', 'likewise', 'similar', 'moreover','in essence', 'in other words', 'that is'
    , 'that is to say', 'in short', 'in brief', 'to put it differently','at first', 'first of all', 'to begin with', 'in the first place'
    , 'at the same time', 'for now', 'for the time being', 'the next step', 'in time', 'in turn', 'later on', 'the meantime', 'in conclusion'
    , 'with this in mind', 'after all', 'all in all', 'all things considered', 'by and large', 'in any case', 'in any event', 'in brief'
    , 'in conclusion', 'on the whole', 'in short', 'in summary', 'in the final analysis', 'in the long run', 'on balance', 'to sum up', 'to summarize']

    for i in t1:
        sum_t1 = text.count(i)+ sum_t1
    for i in t2:
        sum_t2 = plainText.count(i)+ sum_t2
    transitional_words = (sum_t1/wordCount) * 100
    transitional_phrases = sum_t2

    # Transitional word1: addition
    sub_sum1 = 0
    sub_sum2 = 0
    addition_1 = ['also', 'again', 'besides', 'furthermore', 'likewise', 'moreover', 'similarly']
    addition_2 = ['as well as', 'coupled with', 'in addition', ]
    for i in  addition_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in addition_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    addition_words = (sub_sum1/wordCount) * 100
    addition_phrases = sub_sum2

    # Transitional word2: consequence
    sub_sum1 = 0
    sub_sum2 = 0
    consequence_1 = ['accordingly', 'consequently', 'hence', 'otherwise', 'subsequently', 'therefore', 'thus', 'thereupon', 'wherefore']
    consequence_2 = ['as a result', 'for this reason', 'for this purpose', 'so then']
    for i in consequence_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in consequence_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    consequence_words = (sub_sum1/wordCount) * 100
    consequence_phrases = sub_sum2

    # Transitional word3: contrast_and_Comparison
    sub_sum1 = 0
    sub_sum2 = 0
    contrast_and_Comparison_1 = ['contrast', 'conversely', 'instead', 'likewise', 'rather', 'similarly', 'yet', 'but', 'however', 'still', 'nevertheless']
    contrast_and_Comparison_2 = ['by the same token', 'on one hand', 'on the other hand', 'on the contrary', 'in contrast']
    for i in contrast_and_Comparison_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in contrast_and_Comparison_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    contrast_and_Comparison_words = (sub_sum1/wordCount) * 100
    contrast_and_Comparison_phrases = sub_sum2

    # Transitional word4: direction
    sub_sum1 = 0
    sub_sum2 = 0
    direction_1 = ['here', 'there', 'beyond', 'nearly', 'opposite', 'under', 'above']
    direction_2 = ['over there', 'to the left', 'to the right', 'in the distance']
    for i in direction_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in direction_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    direction_words = (sub_sum1/wordCount) * 100
    direction_phrases = sub_sum2

    # Transitional word5: diversion
    sub_sum1 = 0
    sub_sum2 = 0
    diversion_1 = ['incidentally']
    diversion_2 = ['by the way']
    for i in diversion_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in diversion_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    diversion_words = (sub_sum1/wordCount) * 100
    diversion_phrases = sub_sum2

    # Transitional word6: emphasis
    sub_sum1 = 0
    sub_sum2 = 0
    emphasis_1 = ['chiefly', 'especially', 'particularly', 'singularly']
    emphasis_2 = ['above all', 'with attention to']
    for i in emphasis_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in emphasis_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    emphasis_words = (sub_sum1/wordCount) * 100
    emphasis_phrases = sub_sum2

    # Transitional word7: exception
    sub_sum1 = 0
    sub_sum2 = 0
    exception_1 = ['barring', 'beside', 'except', 'excepting', 'excluding', 'save']
    exception_2 = ['aside from', 'exclusive of', 'other than', 'outside of']
    for i in exception_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in exception_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    exception_words = (sub_sum1/wordCount) * 100
    exception_phrases = sub_sum2

    # Transitional word8: exemplifying
    sub_sum1 = 0
    sub_sum2 = 0
    exemplifying_1 = ['chiefly', 'especially', 'markedly', 'namely', 'particularly', 'including' , 'specifically']
    exemplifying_2 = ['for instance', 'in particular', 'such as']
    for i in exemplifying_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in exemplifying_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    exemplifying_words = (sub_sum1/wordCount) * 100
    exemplifying_phrases = sub_sum2

    # Transitional word9: generalizing
    sub_sum1 = 0
    sub_sum2 = 0
    generalizing_1 = ['generally', 'ordinarily', 'usually']
    generalizing_2 = ['as a rule', 'as usual', 'for the most part', 'generally speaking']
    for i in generalizing_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in generalizing_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    generalizing_words = (sub_sum1/wordCount) * 100
    generalizing_phrases = sub_sum2

    # Transitional word10: illustration
    sub_sum1 = 0
    sub_sum2 = 0
    illustration_1 = []
    illustration_2 = ['for example', 'for instance', 'for one thing', 'as an illustration', 'illustrated with', 'as an example', 'in this case']
    for i in illustration_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in illustration_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    illustration_words = (sub_sum1/wordCount) * 100
    illustration_phrases = sub_sum2

    # Transitional word11: similarity
    sub_sum1 = 0
    sub_sum2 = 0
    similarity_1 = ['comparatively', 'correspondingly', 'identically', 'likewise', 'similar', 'moreover']
    similarity_2 = ['coupled with', 'together with']
    for i in similarity_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in similarity_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    similarity_words = (sub_sum1/wordCount) * 100
    similarity_phrases = sub_sum2

    # Ransitional word12: restatement
    sub_sum1 = 0
    sub_sum2 = 0
    restatement_1 = ['namely']
    restatement_2 = ['in essence', 'in other words', 'that is', 'that is to say', 'in short', 'in brief', 'to put it differently']
    for i in restatement_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in restatement_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    restatement_words = (sub_sum1/wordCount) * 100
    restatement_phrases = sub_sum2

    # Transitional word13: sequence
    sub_sum1 = 0
    sub_sum2 = 0
    sequence_1 = ['next', 'then', 'soon', 'later', 'while', 'earlier','simultaneously', 'afterward']
    sequence_2 = ['at first', 'first of all', 'to begin with', 'in the first place', 'at the same time', 'for now', 'for the time being'
    , 'the next step', 'in time', 'in turn', 'later on', 'the meantime', 'in conclusion', 'with this in mind']
    for i in sequence_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in sequence_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    sequence_words = (sub_sum1/wordCount) * 100
    sequence_phrases = sub_sum2


    # Transitional word14: summarizing
    sub_sum1 = 0
    sub_sum2 = 0
    summarizing_1 = ['briefly', 'finally']
    summarizing_2 = [ 'after all', 'all in all', 'all things considered', 'by and large', 'in any case', 'in any event'
    , 'in brief', 'in conclusion', 'on the whole', 'in short', 'in summary', 'in the final analysis', 'in the long run', 'on balance'
    , 'to sum up', 'to summarize']
    for i in summarizing_1:
        sub_sum1 = text.count(i)+ sub_sum1
    for i in summarizing_2:
        sub_sum2 = plainText.count(i)+ sub_sum2
    summarizing_words = (sub_sum1/wordCount) * 100
    summarizing_phrases = sub_sum2

    # prep = len([ (x,y) for x, y in result if y  == "CD" ])/wordCount
    #Cognitive processes
    #cogproc = 0 #LIWC Analysis
    #Cause relationships
    #cause = 0 #LIWC Analysis
    #Discrepencies
    #discrep = 0 #LIWC Analysis
    #Tenant
    #tentat = 0 #LIWC Analysis
    #Differtiation
    #differ = 0 #LIWC Analysis
    #Perceptual processes
    #percept = 0 #LIWC Analysis
    #Verbs past focus VBD VBN
    focuspast = 0
    # focuspast = len(focuspast_list)/wordCount
    focuspast = len([x for x in text if x in LIWC["FocusPast"]])/wordCount * 100
    #Verbs present focus VB VBP VBZ VBG
    focuspresent = 0
    focuspresent = len([x for x in text if x in LIWC["FocusPresent"]])/wordCount * 100
    #net speak
    #netspeak = 0 #LIWC Analysis
    #Assent
    #assent = 0 #LIWC Analysis
    #Non fluencies
    #nonflu = 0 #LIWC Analysis

    return [wordCount, readabilityScore, ReadabilityGrade, DirectionCount, WPS, Sixltr, pronoun, ppron, feature_i, you
    , ipron, prep, verb, auxverb, negate, focuspast, focuspresent, AllPunc, Comma, QMark, Colon, Dash, Parenth
    , Exemplify, transitional_words, transitional_phrases, addition_words, addition_phrases, consequence_words, consequence_phrases
    , contrast_and_Comparison_words, contrast_and_Comparison_phrases, direction_words, direction_phrases, diversion_words, diversion_phrases
    , emphasis_words, emphasis_phrases, exception_words, exception_phrases, exemplifying_words, exemplifying_phrases, generalizing_words, generalizing_phrases
    , illustration_words, illustration_phrases, similarity_words, similarity_phrases
    , restatement_words, restatement_phrases, sequence_words, sequence_phrases,summarizing_words,summarizing_phrases]


def numberlist(X):
    if "Tommy Robinson" in X :
        feature = [1,0]
    else:
        feature = [0,1]
    return feature

def takenotice(X):
    if "take notice" in X or "Take notice" in X:
        feature = [0,1]
    else:
        feature = [1,0]
    return feature

def congratulation(X):

    if "congratulations to all" in X.lower() or "thank you" in X.lower():
        feature = [1,0]
    else:
        feature = [0,1]
    return feature

### MAIN ###
def val(type):
    train_folder = "datasets/train-articles"
    dev_folder = "datasets/dev-articles"
    test_folder = "datasets/test-articles"
    train_labels_folder = "datasets/train-labels-SLC"
    task_SLC_output_file = "SLC_"+type+"_output.txt"
    try:
        with open('./models/emotion.p', 'rb') as f:
            features_train, features_dev, features_test= pickle.load(f)
    except:
        import emotion_features
        features_train, features_dev, features_test = emotion_features.emotion_features()
    train_article_ids, train_sentence_ids, sentence_list = read_articles_from_file_list(train_folder)
    reference_articles_id, reference_sentence_id_list, gold_labels = read_predictions_from_file_list(train_labels_folder, "*.task-SLC.labels")
    dev_sentence_list = []
    dev_article_id_list = []
    dev_sentence_id_list=[]
    features_val = []
    if type=='test':
        dev_article_id_list, dev_sentence_id_list, dev_sentence_list = read_articles_from_file_list(test_folder)
        features_val = features_test
    elif type=='dev':
        dev_article_id_list, dev_sentence_id_list, dev_sentence_list = read_articles_from_file_list(dev_folder)
        features_val = features_dev
    print("Loaded %d sentences from %d %s_articles" % (len(dev_sentence_list), len(set(dev_article_id_list)),type))
    # with open('./models/raw_data.p','wb') as file:
    #     pickle.dump((sentence_list,dev_sentence_list,test_sentence_list,gold_labels),file)
    # pd.DataFrame(dev_sentence_list).to_csv('./datasets/test.csv',index=False)

    # numberlist
    numberlisttrain = [numberlist(text) for text in sentence_list]
    numberlistdev = [numberlist(text) for text in dev_sentence_list]

    # takenotice
    takenoticetrain = [takenotice(text) for text in sentence_list]
    takenoticedev = [takenotice(text) for text in dev_sentence_list]

    # enough
    enoughtrain = [congratulation(text) for text in sentence_list]
    enoughdev = [congratulation(text) for text in dev_sentence_list]


    othertrain = []
    othertest = []

    bert_train = []
    bert_test = []
    x_train2 = np.load('./datasets/x_train_bert70.npy')
    x_test2 = np.load('./datasets/x_'+type+'_bert70.npy')
    i = 0
    for ss in sentence_list:
        if ss == '':
            bert_train.append(np.array([0] * 768).astype('int32'))
            othertrain.append([0]*54)
        else:
            bert_train.append(x_train2[i])
            i += 1
            othertrain.append(gettingFeatures(ss))
    ii = 0
    for ss in dev_sentence_list:
        if ss == '':
            bert_test.append(np.array([0] * 768).astype('int32'))
            othertest.append([0]*54)
        else:
            bert_test.append(x_test2[ii])
            ii += 1
            othertest.append(gettingFeatures(ss))

    # sentence_list = clean_data.clean(sentence_list)
    # dev_sentence_list = clean_data.clean(dev_sentence_list)

    # length
    train_length = np.array([len(sentence) for sentence in sentence_list]).reshape(-1, 1)
    dev_length = np.array([len(sentence) for sentence in dev_sentence_list]).reshape(-1, 1)


    # vectorize
    vec = TfidfVectorizer(ngram_range=(1, 4), use_idf=True, min_df=3,norm='l2')
    vec.fit(sentence_list)
    train_vec = vec.transform(sentence_list)
    dev_vec = vec.transform(dev_sentence_list)


    # miss vocabulary
    try:
        with open('./models/vocab.p', 'rb') as file:
            voc=pickle.load(file)
    except:
        voc = clean_data.build_missing_voc()
    vec2 = CountVectorizer(ngram_range=(1,1),binary=False, vocabulary=voc)
    vec2.fit(sentence_list)
    train_vec2 = vec2.transform(sentence_list)
    dev_vec2 = vec2.transform(dev_sentence_list)
    vec3 = CountVectorizer(ngram_range=(3, 3), binary=True, vocabulary=['sooner or later'])
    vec3.fit(sentence_list)
    train_vec3 = vec3.transform(sentence_list)
    dev_vec3 = vec3.transform(dev_sentence_list)


    token_sentence_train = [nltk.word_tokenize(text.lower()) for text in sentence_list]

    token_sentence_dev = [nltk.word_tokenize(text.lower()) for text in dev_sentence_list]


    # if token_count<8, it is short document, else it's long
    shortlongtrain=[]
    shortlongtest=[]
    for tst in token_sentence_train:
        if len(tst)<8:
            shortlongtrain.append(0)
        else:
            shortlongtrain.append(1)
    for tsd in token_sentence_dev:
        # print(tt)
        if len(tsd)<8:
            shortlongtest.append(0)
        else:
            shortlongtest.append(1)
    shortlongtrain = np.array(shortlongtrain).reshape(-1,1)
    shortlongtest = np.array(shortlongtest).reshape(-1, 1)

    # whatabout
    whatabouttrain = [whatabout(text) for text in sentence_list]
    whataboutdev = [whatabout(text) for text in dev_sentence_list]



    #howdareyou
    howdaretrain = [howdareyou(text) for text in sentence_list]
    howdaredev = [howdareyou(text) for text in dev_sentence_list]

    #timefor
    timefortrain = [timefor(text) for text in sentence_list]
    timefordev = [timefor(text) for text in dev_sentence_list]

    #hiter
    hitertrain = [hiter(text) for text in sentence_list]
    hiterdev = [hiter(text) for text in dev_sentence_list]

    #eitheror
    eitherortrain = [eitheror(text) for text in sentence_list]
    eitherordev = [eitheror(text) for text in dev_sentence_list]

    # sooner or later
    # soonertrain = [sooner(text) for text in sentence_list]
    # soonerdev = [sooner(text) for text in dev_sentence_list]

    #slogan
    slogantrain = [slogan(text) for text in sentence_list]
    slogandev = [slogan(text) for text in dev_sentence_list]



    # liwc
    liwctrain = pd.read_csv('./datasets/liwctrain.csv')[['WC', 'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'Sixltr', 'Dic', 'function', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'prep', 'auxverb', 'adverb', 'conj', 'negate', 'verb', 'adj', 'compare', 'interrog', 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation', 'achieve', 'power', 'reward', 'risk', 'focuspast', 'focuspresent', 'focusfuture', 'relativ', 'motion', 'space', 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler', 'AllPunc', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP']]
    liwctest = pd.read_csv('./datasets/liwc'+type+'.csv')[['WC', 'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'Sixltr', 'Dic', 'function', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'prep', 'auxverb', 'adverb', 'conj', 'negate', 'verb', 'adj', 'compare', 'interrog', 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation', 'achieve', 'power', 'reward', 'risk', 'focuspast', 'focuspresent', 'focusfuture', 'relativ', 'motion', 'space', 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler', 'AllPunc', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP']]
    print(np.array(liwctest).shape)
    print("start to select features")

    train1 = np.concatenate([np.array(liwctrain),np.array(othertrain),bert_train,train_length,train_vec2.toarray()],axis=1)
    dev1 = np.concatenate([np.array(liwctest),np.array(othertest),bert_test,dev_length,dev_vec2.toarray()],axis=1)
    model1 = SelectKBest(f_classif, k=260)
    train1 = model1.fit_transform(train1, gold_labels)
    dev1 = model1.transform(dev1)

    model2 = SelectKBest(f_classif, k=100)
    a1 = model2.fit_transform(train_vec.toarray(), gold_labels)
    a2 = model2.transform(dev_vec.toarray())


    model3 = SelectKBest(f_classif, k=251)
    a3 = model3.fit_transform(train_vec2.toarray(), gold_labels)
    a4 = model3.transform(dev_vec2.toarray())
    # model5 = SelectKBest(f_classif, k=2)
    # a5 = model5.fit_transform(enoughtrain, gold_labels)
    # a6 = model5.transform(enoughdev)
    train = np.concatenate([train1,features_train,np.array(slogantrain),a1,a3,howdaretrain,hitertrain,shortlongtrain,numberlisttrain,takenoticetrain],axis=1)
    dev = np.concatenate([dev1,features_val,np.array(slogandev),a2,a4,howdaredev,hiterdev,shortlongtest,numberlistdev,takenoticedev],axis=1)

    train = np.row_stack((train, np.array([[0]*(train.shape[1]-3)+[1,1,0]])))
    gold_labels.insert(-1,'propaganda')
    train = np.row_stack((train, np.array([[0] * (train.shape[1]-1) + [1]])))
    gold_labels.insert(-1, 'propaganda')

    model4 = SelectKBest(f_classif, k=635)
    train = model4.fit_transform(train, gold_labels)
    dev = model4.transform(dev)

    # pd.DataFrame(np.concatenate([np.array(sentence_list+['','']).reshape(-1,1),train,np.array(gold_labels).reshape(-1,1)],axis=1)).to_csv('./datasets/features_train.csv',index=False)
    # pd.DataFrame(np.concatenate([np.array(dev_sentence_list).reshape(-1,1),dev],axis=1)).to_csv('./datasets/features_dev.csv', index=False)

    # show features name
    name1 = ['WC', 'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'Sixltr', 'Dic', 'function',
            'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article',
            'prep', 'auxverb', 'adverb', 'conj', 'negate', 'verb', 'adj', 'compare',
            'interrog', 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger',
            'sad', 'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight', 'cause',
            'discrep', 'tentat', 'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio',
            'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation', 'achieve', 'power',
            'reward', 'risk', 'focuspast', 'focuspresent', 'focusfuture', 'relativ', 'motion',
            'space', 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal',
            'swear', 'netspeak', 'assent', 'nonflu', 'filler', 'AllPunc', 'Period', 'Comma', 'Colon',
            'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP']+['wordCount',
             ' readabilityScore', ' ReadabilityGrade', ' DirectionCount', ' myWPS', ' mySixltr', ' mypronoun', ' myppron',
            ' feature_i', ' myyou', ' myipron', ' myprep', ' myverb', ' myauxverb', ' mynegate', ' myfocuspast', ' myfocuspresent',
             ' myAllPunc', ' myComma', 'myQMark', ' myColon', ' myDash', ' myParenth', ' Exemplify',
             ' transitional_words', ' transitional_phrases', ' addition_words', ' addition_phrases',
             ' consequence_words', ' consequence_phrases', ' contrast_and_Comparison_words',
             ' contrast_and_Comparison_phrases', ' direction_words', ' direction_phrases',
              ' diversion_words', ' diversion_phrases', ' emphasis_words', ' emphasis_phrases',
             ' exception_words', ' exception_phrases', ' exemplifying_words', ' exemplifying_phrases',
             ' generalizing_words', ' generalizing_phrases', ' illustration_words', ' illustration_phrases',
             ' similarity_words', ' similarity_phrases', ' restatement_words', ' restatement_phrases',
             ' sequence_words', 'sequence_phrases', 'summarizing_words', 'summarizing_phrases']+["bert_"+str(i) for i in range(768)]+['length']+vec2.get_feature_names()
    outcome1 = list(model1.get_support(indices=True))
    newname1=[]
    for i in range(0, len(name1)):
        if i in outcome1:
            newname1.append(name1[i])
    name2=vec.get_feature_names()
    outcome2 = list(model2.get_support(indices=True))
    newname2 = []
    for i in range(0, len(name2)):
        if i in outcome2:
            newname2.append(name2[i])
    name3 = vec2.get_feature_names()
    outcome3 = list(model3.get_support(indices=True))
    newname3 = []
    for i in range(0, len(name3)):
        if i in outcome3:
            newname3.append(name3[i])
    name4 = newname1+['Valence', 'Arousal','Dominance', 'pos', 'neg', 'neu', 'anger',
     'disgust', 'fear', 'joy', 'sadness', 'surprise','anger_int', 'disgust_int', 'fear_int', 'joy_int',
      'sadness_int', 'surprise_int', 'affin', 'positive', 'negative', 'insult']+['slogan0','slogan1']+newname2+newname3\
    +['howdare0','howdare1','hitler0','hitler1','shortlongdoc','number0','number1','takenotice0','takenotice1']
    outcome4 = list(model3.get_support(indices=True))
    for i in range(0, len(name4)):
        if i in outcome4:
            print(name4[i])

    print(len(name4))
    print("start training")
    model = LogisticRegression(penalty='l2', class_weight='balanced', solver="lbfgs", max_iter=8000, C=1)
    model.fit(train, gold_labels)
    predictions = model.predict(dev)

    # predictions file with text
    with open("./datasets/full_"+type+"_predictions.tsv", "w") as fout:
        for article_id, sentence_id, sentence,prediction in zip(dev_article_id_list, dev_sentence_id_list,dev_sentence_list, predictions):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, sentence_id,sentence,prediction))

    # writing predictions to file
    with open(task_SLC_output_file, "w") as fout:
        for article_id, sentence_id, prediction in zip(dev_article_id_list, dev_sentence_id_list, predictions):
            fout.write("%s\t%s\t%s\n" % (article_id, sentence_id, prediction))
    print("Predictions written to file " + task_SLC_output_file)


if __name__ == '__main__':

    val('test')
