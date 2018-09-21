# coding: utf-8

# Meet Robo your friend
#
# Robo will respond to your greetings
# Robo will answer your query related to your trained data

import sys
import nltk
import random
# to process standard python strings
import string
# TF-IDF vector
from sklearn.feature_extraction.text import TfidfVectorizer
# for similarity between words entered by the user and the words in the corpus
from sklearn.metrics.pairwise import cosine_similarity

"""
Read sample Data
"""

f = open('./sample_data.txt', 'r', errors='ignore')
raw = f.read()
# converts to lowercase
raw = raw.lower()

# first-time use only
nltk.download('punkt')

# first-time use only
nltk.download('wordnet')

# converts to list of sentences
sent_tokens = nltk.sent_tokenize(raw)

# converts to list of words
word_tokens = nltk.word_tokenize(raw)


"""
pre-process raw data
"""
# WordNet is a semantically-oriented dictionary of English included in NLTK.
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def lemmatize_tokens(tokens):
    """ return the token """
    return [lemmer.lemmatize(token) for token in tokens]


def lemma_normalize(text):
    """ normalize the token """
    return lemmatize_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


"""
Keyword matching
"""
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


# returning selected greeting response as per matched input
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


"""
Generating Response
"""


def response(user_response):
    robo_response = ''
    tf_idf_vec = TfidfVectorizer(tokenizer=lemma_normalize, stop_words='english')
    tf_idf = tf_idf_vec.fit_transform(sent_tokens)
    vals = cosine_similarity(tf_idf[-1], tf_idf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tf_idf = flat[-2]
    if req_tf_idf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


flag = True
print("****"*20)
print("ROBO: Hi Robo here. I will answer your queries about Chatbots. If you want to exit, type Bye!\n")

while flag:
    if sys.argv[1] is None:
        user_response = input()
    else:
        user_response = "bye"

    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("ROBO: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! take care..")
