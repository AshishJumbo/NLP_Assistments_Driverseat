import pandas as pd

df = pd.read_csv('cleaned.csv')
# df = pd.read_csv('training_set.csv')

df['combined_grade'] = 0

# clean up the training data:
#   remove the questions that have raw_answer_text and cleaned_answer_text as NaN
#   as having those values as NaN doesn't make any sense since what is to grade when
#   there isn't any stuff to actually grade

# df_train_clean = df_train.dropna( axis=0, subset=[ 'raw_answer_text', 'cleaned_answer_text'])

df.dropna( axis=0, subset=[ 'raw_answer_text', 'cleaned_answer_text'])

# print(df.head())

# print("\n -------------- \n info about data", df.info())
# print("\n -------------- \n data columns \n ", df.columns)

# dataset is the last column that has training and evaluating types for train and test dataset
print("\n -------------- \n unique values in dataset attribute \n", df['dataset'].value_counts())

# separate the data set into training and evaluation data set
df_evaluation = df.loc[df['dataset'] == 'evaluation']
df_train = df.loc[df['dataset'] == 'training']
df_train_clean = df_train.dropna( axis=0, subset=[ 'raw_answer_text', 'cleaned_answer_text'])

# dataset with dropped values after cleaning the train data set
# print("\n -------------- \n dataset after cleanup", df_train_clean['dataset'].value_counts())

# printing out the shapes of the evaluation and training dataset
# print("\n -------------- \n evaluation shape \n", df_evaluation.shape)
# print("\n training shape \n", df_train_clean.shape)

# the NaN grades in the cleaned datasets
# eval_NaN = df_evaluation.loc[(df_evaluation['grade_0'].notnull()) & (df_evaluation['grade_1'].notnull()) & (df_evaluation['grade_2'].notnull()) & (df_evaluation['grade_3'].notnull()) & (df_evaluation['grade_4'].notnull())]
# train_NaN = df_train_clean.loc[(df_train_clean['grade_0'].isnull()) & (df_train_clean['grade_1'].isnull()) & (df_train_clean['grade_2'].isnull()) & (df_train_clean['grade_3'].isnull()) & (df_train_clean['grade_4'].isnull())]
df_train_clean_NaN = df_train_clean.loc[df_train_clean['grade_0'].isnull()]
df_train_clean_notNaN = df_train_clean.loc[df_train_clean['grade_0'].notnull()]

# print("\n -------------- \n NaN grades in training \n ", df_train_clean_NaN.shape)
# print("\n numeric grades in training \n", df_train_clean_notNaN[['problem_id', 'problem_set']])

# df_train_clean_problemids = df_train_clean['problem_id'].unique()
#
# def multiplesets_sameproblemid():
#     df_problemsets = df_train_clean[['problem_set', 'problem_id']]
#     for id in df_train_clean_problemids:
#         set_prbsets = []
#         for index, row in df_problemsets.iterrows():
#             if (id == row['problem_id']):
#                 if (row['problem_set'] not in set_prbsets):
#                     set_prbsets.append(row['problem_set'])
#
#         print(" The id ", id ,"exists in the following problem sets : ", set_prbsets)
#
#
# multiplesets_sameproblemid()

# df_train_clean_problemids = df_train_clean.groupby(['problem_id', 'problem_set']).size()
# print("\n ------------- \n unique problem ids", df_train_clean_problemids)


df_train_clean_notNaN.loc[df.grade_0 == 1, 'combined_grade'] = 0.0
df_train_clean_notNaN.loc[df.grade_1 == 1, 'combined_grade'] = 0.25
df_train_clean_notNaN.loc[df.grade_2 == 1, 'combined_grade'] = 0.5
df_train_clean_notNaN.loc[df.grade_3 == 1, 'combined_grade'] = 0.75
df_train_clean_notNaN.loc[df.grade_4 == 1, 'combined_grade'] = 1.0

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# for problem_id = 2
labels = df_train_clean_notNaN.loc[(df_train_clean_notNaN['problem_id'] == 15)]['combined_grade']
text = df_train_clean_notNaN.loc[(df_train_clean_notNaN['problem_id'] == 15)]['raw_answer_text']

print(labels.value_counts())

x_train, x_test, y_train, y_test = train_test_split( text, labels, random_state=0, test_size=0.3)

#bag-of-words feature matrix
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=20000, stop_words='english')
bow_x_train = bow_vectorizer.fit_transform(x_train)
bow_x_test = bow_vectorizer.fit_transform(x_test)

# tf-idf feature matrix
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=2000, stop_words='english')
tfidf_x_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_x_test = tfidf_vectorizer.fit_transform(x_test)

labels = LabelEncoder()
labels_y_train_bow = labels.fit(y_train)
labels_y_test_bow = labels.fit(y_test)
labels_y_train_tfidf = labels.transform(y_train)
labels_y_test_tfidf = labels.transform(y_test)

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn import metrics

linear_svc = LinearSVC()
clf = linear_svc.fit(tfidf_x_train, labels_y_train_tfidf)
calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc, cv="prefit")
calibrated_svc.fit(tfidf_x_train, labels_y_train_tfidf)
predicted = calibrated_svc.predict(tfidf_x_test)


print('average accuracy of svc = {} \n'. format(np.mean(predicted == (labels_y_test_tfidf))))
print(' the predicted values are: ')
print(pd.DataFrame(calibrated_svc.predict_proba(tfidf_x_test)*100, columns=labels.classes_))

print('rmse : ', np.sqrt(metrics.mean_squared_error(labels_y_test_tfidf, predicted)))

fpr, tpr, thresholds = metrics.roc_curve(labels_y_test_tfidf, predicted, pos_label=2)
print('auc : ', metrics.auc(fpr, tpr))
print('auc 2: ', metrics.roc_auc_score(labels_y_test_tfidf, predicted))

print('cohen kappa score : ', metrics.cohen_kappa_score(labels_y_test_tfidf, predicted))


# from sklearn.tree import DecisionTreeClassifier
#
# dtree_model = DecisionTreeClassifier().fit(tfidf_x_train, labels_y_train_tfidf)
# dtree_predictions = dtree_model.predict(tfidf_x_test)
#
# print('average accuracy od decision tree = {} \n'. format(np.mean(dtree_predictions == labels.transform(labels_y_test_tfidf))))
#
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=8).fit(tfidf_x_train, labels_y_train_tfidf)
#
# accuracy1 = knn.score(tfidf_x_test, labels_y_test_tfidf)
# print('accuracy of knn \n', accuracy1)
#
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB().fit(tfidf_x_train, labels_y_train_tfidf)
#
# gnb_prediction = gnb.predict(tfidf_x_test)
#
# accuracy2 = gnb.score(tfidf_x_test, labels_y_test_tfidf)
#
# print('accuracy of gnb \n', accuracy2)
#
#
#
#
#
