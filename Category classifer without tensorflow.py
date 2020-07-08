import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from enum import Enum

import inspect


nltk.download('stopwords')

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

class WordNormalization(Enum):
    CHAR_ONLY_SPACE_REGEXP = r'[^a-zA-Z\s]'
    HTML_MARKUPS_SPACE_REGEXP = r'(<.*?>)'
    NON_ASCII_AND_DIGITS_SPACE_REGEXP = r'(\\W|\\d)'
    MIN_WORD_LENGTH = 3
    MAX_WORD_LENGTH = 16
    STOP_WORDS =  list(nltk.corpus.stopwords.words('english')) + list(nltk.corpus.stopwords.words('german'))

class ClassiferContsants(Enum):   
    CRITERION = 'entropy'
    MAX_DEPTH = 50
    N_ESTIMATORS = 30
    RANDOM_STATE = 50
    CLASS_WEIGHT = 'balanced'



class TextFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        result_df = pd.DataFrame(X)
        result_df['word_count'] = result_df['ocr_text'].apply(lambda x : len(x.split()))
        result_df['char_count'] = result_df['ocr_text'].apply(lambda x : len(x.replace(" ","")))
        result_df['word_density'] = result_df['word_count'] / (result_df['char_count'] + 1)
        result_df['total_length'] = result_df['ocr_text'].apply(len)
        result_df['capitals'] = result_df['ocr_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
        result_df['caps_vs_length'] = result_df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
        result_df['num_exclamation_marks'] =result_df['ocr_text'].apply(lambda x: x.count('!'))
        result_df['num_question_marks'] = result_df['ocr_text'].apply(lambda x: x.count('?'))
        result_df['num_punctuation'] = result_df['ocr_text'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
        result_df['num_symbols'] = result_df['ocr_text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
        result_df['num_unique_words'] = result_df['ocr_text'].apply(lambda x: len(set(w for w in x.split())))
        result_df['words_vs_unique'] = result_df['num_unique_words'] / result_df['word_count']
        result_df["word_unique_percent"] =  result_df["num_unique_words"]*100/result_df['word_count'] 
        return result_df


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X['ocr_text'] = X['ocr_text'].apply(self._preprocess_text)
        return X

    def _scrub_text(self, text):
        text = re.sub(WordNormalization.CHAR_ONLY_SPACE_REGEXP.value,' ', text)
        text = re.sub(WordNormalization.HTML_MARKUPS_SPACE_REGEXP.value,' ',text)
        text = re.sub(WordNormalization.NON_ASCII_AND_DIGITS_SPACE_REGEXP.value,' ',text)
        text = text.lower()
        text = text.strip()
        return text

    def _stem_text(self, text):
        porter_stemmer = PorterStemmer()
        text = porter_stemmer.stem(text)
        return text

    def _preprocess_text(self, text):
        text = self._scrub_text(text)
        text = self._stem_text(text)
        wpt = nltk.WordPunctTokenizer()
        tokens = wpt.tokenize(text)
        filtered_tokens = [token for token in tokens if (token not in WordNormalization.STOP_WORDS.value) and (len(token)>WordNormalization.MIN_WORD_LENGTH.value) and (len(token)<WordNormalization.MAX_WORD_LENGTH.value) and (token.isalpha())]
        return ' '.join(filtered_tokens)

class TfidfVectorizerWrapper(TfidfVectorizer):
    def fit_transform(self, features, y=None):
        return super().fit_transform(features.ocr_text)

    def transform(self, features, **kwargs):
        return super().transform(features.ocr_text)

class CategoryClassifer (BaseEstimator, ClassifierMixin):  
    def __init__(self,
                 criterion = ClassiferContsants.CRITERION.value,
                 max_depth = ClassiferContsants.MAX_DEPTH.value,
                 n_estimators = ClassiferContsants.N_ESTIMATORS.value,
                 random_state = ClassiferContsants.RANDOM_STATE.value,
                 class_weight = ClassiferContsants.CLASS_WEIGHT.value,
                ):

      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      values.pop("self")
      for arg, val in values.items():
        setattr(self, arg, val)
      
      # self.criterion = criterion
      # self.max_depth = max_depth,
      # self.n_estimators = n_estimators,
      # self.random_state = random_state,
      # self.class_weight = class_weight


    def _pipeline_constructor(self):
      self.pipeline_ = Pipeline(steps=[('features', TextFeaturesExtractor()),
                                       ('norm_text', TextPreprocessor()),
                                       ('vectorizer', TfidfVectorizerWrapper(max_df=0.98,
                                                                             min_df=0.02,
                                                                             )),
                                       ('classifier', RandomForestClassifier(criterion = self.criterion,
                                                                             max_depth = self.max_depth,
                                                                             n_estimators = self.n_estimators,
                                                                             random_state = self.random_state,
                                                                             class_weight = self.class_weight,
                                                                                                 ))])
   
    def fit(self, X, y=None):  
      self._pipeline_constructor()
      self.pipeline_.fit(X, y)
      self.category_result_ = self.predict(X)
      self.recall_precision_product_ = self.score(y, self.category_result_)
    #   self.feature_importances_ = self.pipeline_.feature_importances_
      return self

    def predict(self, X, y=None):
      return self.pipeline_.predict(X)

    def predict_proba(self, X, y=None):
      return self.pipeline_.predict_proba(X)

    def score(self, X, y=None):
      self.recall_ = recall_score(X, y, average='micro')
      self.precision_ = precision_score(X, y, average='micro')
      return self.recall_*self.precision_


# Define csv path
csv_path = '/content/dataset.csv'

# Load data
test_df = pd.read_csv(csv_path)

# Data wrangling
test_df = test_df.dropna(subset=['ocr_text'])
target_column = test_df.category
sub_df = test_df.ocr_text


# Split train test data
X_train, X_test, y_train, y_test = train_test_split(sub_df, target_column, test_size=0.2)

clf = CategoryClassifer ()
clf.fit(X_train, y_train)

print("Train data recall score: %.3f" % clf.recall_)
print("Train data precision score: %.3f" % clf.precision_)

yhat = clf.predict(X_test)
print("Test data recall score: %.3f" % recall_score(yhat, y_test, average='micro'))
print("Test data precision score: %.3f" % precision_score(yhat, y_test, average='micro'))

cross_validation_scoring = ['precision_micro', 'recall_micro']
clf2 = CategoryClassifer ()
scores = cross_validate(clf2, sub_df, target_column, scoring=cross_validation_scoring, cv=5, return_train_score=True)
print("Mean cross validation recall score weighted: %.3f" % scores['test_recall_micro'].mean())
print("Mean cross validation precision score weighted: %.3f" % scores['test_precision_micro'].mean())

# max_depth_range = [40, 50, 60]
# random_state_range = [50, 60, 70]
# n_estimators_range = [30, 40, 50]
# ngram_range_range = [(1,1), (1,2), (2,2)]
# min_df_range = [0, 0.01, 0.02]
# max_df_range = [1, 0.98, 0.95]
# sublinear_tf_range = [True, False]


# pipe1 = Pipeline(steps=[('features', TextFeaturesExtractor()),
#                         ('norm_text', TextPreprocessor()),
#                         ('vectorizer', TfidfVectorizerWrapper()),
#                         ('classifier', RandomForestClassifier(criterion = 'entropy',
#                                                               n_jobs=-1,
#                                                               class_weight = 'balanced',))
#                        ])

# tuned_params = {'vectorizer__min_df': min_df_range,
#                 'vectorizer__max_df': max_df_range,
#                 'vectorizer__sublinear_tf': sublinear_tf_range,
#                 'vectorizer__ngram_range': ngram_range_range,
#                 'classifier__max_depth': max_depth_range,
#                 'classifier__random_state': random_state_range,
#                 'classifier__n_estimators': n_estimators_range,
#                 }
# gs = GridSearchCV(pipe1, tuned_params)
# gs.fit(X_train, y_train)
# gs.best_params_

actual_categories = y_test
predicted_categories = clf.predict(X_test)
conf_matrix1 = pd.crosstab(actual_categories, predicted_categories, rownames=['Actual'], colnames=['Predicted'])
print (conf_matrix1)
conf_matrix2 = confusion_matrix(actual_categories, predicted_categories)
print (conf_matrix2)
print('Classification Report : ')
print (classification_report(actual_categories, predicted_categories))

    # def _class_weight_constructor(self):
    #   self.class_weight_ = dict(zip(set(self.categories_), [ProblematicCategories.CATEGORIES.value.get('baseline').get('class_weight') for i in range(len(set(self.categories_)))]))
    #   for category in ProblematicCategories.CATEGORIES.value.keys():
    #     if category != 'baseline':
    #       self.class_weight_[category] = ProblematicCategories.CATEGORIES.value.get(category).get('class_weight')

    # def _encoder_constructor(self, y):
    #   label_encoding = LabelEncoder()
    #   label_encoding.fit(y)
    #   self.categories_ = label_encoding.classes_

    # def _warmup_classifer(self, y):
    #   self._encoder_constructor(y)  
    #   self._class_weight_constructor()
    #   self._pipeline_constructor()

    # def _repredict(self, prob_array):
    #   prob_array[np.argmax(prob_array)] = 0
    #   return prob_array

    # def _interpert_proba(self, proba_result):
    #   actual_categories = pd.Series(list(proba_result)).apply(lambda x: self.categories_[np.argmax(x)])
    #   return actual_categories.values

    # def _predict_proba_with_repredict(self, X, y=None):
    #   self.y_prob_ = pd.Series(list(self.predict_proba(X))) 
    #   self.thresholds_ = [ProblematicCategories.CATEGORIES.value.get(category,{'repredict_threhold' : 1}).get('repredict_threshold', 1) for category in self.categories_] 
    #   self.y_prob_mask_ = self.y_prob_.apply(lambda x: max(x) > self.thresholds_[np.argmax(x)])
    #   self.y_prob_altered_ = self.y_prob_[self.y_prob_mask_].apply(self._repredict)
    #   self.y_prob_final_ = pd.concat([self.y_prob_altered_, self.y_prob_[self.y_prob_mask_.map({True : False, False : True})]]).sort_index()
    #   return np.array(self.y_prob_final_.tolist())
