from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import numpy as np

ENGLISH_STOPWORDS = stopwords.words('english')
PUNCTUATIONS = string.punctuation
STEMMER = PorterStemmer()

def tokenize(sentence):
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = [word for word in words if word not in PUNCTUATIONS]
    return words

def stem(word):
    return STEMMER.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # example:
    # sentence = ["hello", "how", "are", "you"]
    # words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    # bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in sentence_words: 
            bag[i] = 1
    return bag