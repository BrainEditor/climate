import pandas as pd
import numpy as np
import tag_lemmatize.tag_lemmatize as tl

from collections import Counter
from itertools import chain


def word_count(corpus):
    """
    This function takes a list of strings and returns a Counter.
    Thanks to ShadowRanger on SO. https://stackoverflow.com/a/35857833
    """
    return Counter(chain.from_iterable(map(str.split, corpus)))


# prepare data
cspam = [i.lower() for i in
         pd.read_csv("cspam_climate.csv")['post'].
         replace('', np.nan).dropna().tolist()]

dnd = [i.lower() for i in
       pd.read_csv("dnd_climate.csv")['post'].
       replace('', np.nan).dropna().tolist()]

# get total word count
num_words = sum([len(post.split()) for post in (cspam + dnd)])

# lemmatize data
cspam_lem = [tl.tag_and_lem(post) for post in cspam]
dnd_lem = [tl.tag_and_lem(post) for post in dnd]

# store words in Counter to get frequencies
cspam_words = word_count(cspam_lem)
dnd_words = word_count(dnd_lem)

# number of instances of lmao per corpus
dnd_words['lmao']
cspam_words['lmao']

# get set of all words in each corpus
cspam_set = set([w for (w, i) in cspam_words.most_common()])
dnd_set = set([w for (w, i) in dnd_words.most_common()])

# subtract corpora from each other to get unique words
cspam_unique = cspam_set - dnd_set
dnd_unique = dnd_set - cspam_set

# filter lists of most common words by uniqueness
cspam_unique_freq = [(w, i) for (w, i) in cspam_words.most_common()
                     if w in cspam_unique]

dnd_unique_freq = [(w, i) for (w, i) in dnd_words.most_common()
                   if w in dnd_unique]
