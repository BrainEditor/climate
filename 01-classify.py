from nltk.corpus import stopwords

import tag_lemmatize.tag_lemmatize as tl
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def show_most_informative_features(vct, clf, n=20):
    """ with thanks to tobigue @ SO: https://stackoverflow.com/a/11140887 """
    feature_names = vct.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7,
                             stop_words=stopwords.words('english'),
                             ngram_range=(1, 2))

classifier = LogisticRegression(solver="liblinear", max_iter=100)

# Create a dataframe containing all posts ------------------------------------

cspam_climate = pd.read_csv("cspam_climate.csv")
dnd_climate = pd.read_csv("dnd_climate.csv")
cspam_primary = pd.read_csv("cspam_primary.csv")
dnd_primary = pd.read_csv("dnd_primary.csv")

topic_target = np.repeat([1, 0], 24000)
forum_target = np.repeat([1, 0, 1, 0], 12000)

threads = [cspam_climate, dnd_climate, cspam_primary, dnd_primary]
all_posts = pd.concat(threads)

all_posts['topic'] = topic_target
all_posts['forum'] = forum_target

# Classify climate change threads by subforum. -------------------------------

climate_posts = all_posts.loc[(all_posts.topic == 1)].copy(deep=True)

climate_posts['post'].replace('', np.nan, inplace=True)
climate_posts.dropna(subset=['post'], inplace=True)

data = [tl.tag_and_lem(post) for post in climate_posts['post'].tolist()]
target = climate_posts['forum'].tolist()

X = vectorizer.fit_transform(data)
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

show_most_informative_features(vectorizer,
                               classifier,
                               n=30)

# Classify climate change vs. Democratic primary posts, both forums. ---------

all_nona = all_posts.copy(deep=True)
all_nona.dropna(subset=['post'], inplace=True)

topic_data = [tl.tag_and_lem(post) for post in all_nona['post'].tolist()]
topic_target = all_nona['topic'].tolist()

X = vectorizer.fit_transform(topic_data)
y = topic_target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

topic_classifier = classifier
topic_classifier.fit(X_train, y_train)
y_pred = topic_classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

show_most_informative_features(vectorizer, classifier, n=10)

# Classify all posts by subforum. --------------------------------------------

forum_data = topic_data  # no need to re-tag and lemmatize here
forum_target = all_nona['forum'].tolist()

X = vectorizer.fit_transform(forum_data)
y = forum_target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

show_most_informative_features(vectorizer, classifier, n=30)
