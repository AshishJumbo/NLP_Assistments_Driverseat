import pandas as pd

df = pd.read_csv('cleaned.csv')

df['combined_grade'] = 0

# clean up the training data:
#   remove the questions that have raw_answer_text and cleaned_answer_text as NaN
#   as having those values as NaN doesn't make any sense since what is to grade when
#   there isn't any stuff to actually grade

df.dropna( axis=0, subset=[ 'raw_answer_text', 'cleaned_answer_text'])

# print("\n -------------- \n unique values in dataset attribute \n", df['dataset'].value_counts())

# separate the data set into training and evaluation data set
# df_evaluation = df.loc[df['dataset'] == 'evaluation']
# df_train = df.loc[df['dataset'] == 'training']

df_train_clean = df.dropna( axis=0, subset=[ 'raw_answer_text', 'cleaned_answer_text'])

print(df_train_clean.loc[(df_train_clean['dataset'] == 'evaluation')&(df_train_clean['problem_id'] == 14)]['grade_0'])

# the NaN grades in the cleaned datasets
df_train_clean_NaN = df_train_clean.loc[df_train_clean['grade_0'].isnull()]
df_train_clean_notNaN = df_train_clean.loc[df_train_clean['grade_0'].notnull()]

df_train_clean_notNaN.loc[df_train_clean_notNaN.grade_0 == 1, 'combined_grade'] = 0.0
df_train_clean_notNaN.loc[df_train_clean_notNaN.grade_1 == 1, 'combined_grade'] = 0.25
df_train_clean_notNaN.loc[df_train_clean_notNaN.grade_2 == 1, 'combined_grade'] = 0.5
df_train_clean_notNaN.loc[df_train_clean_notNaN.grade_3 == 1, 'combined_grade'] = 0.75
df_train_clean_notNaN.loc[df_train_clean_notNaN.grade_4 == 1, 'combined_grade'] = 1.0

# list of all the problem ids
problem_ids = df_train_clean_notNaN['problem_id'].unique()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn import metrics

count = 0
rmse_array = 0
auc_array = 0

answers = pd.DataFrame({0.0:[], 0.25:[], 0.5:[], 0.75:[],1.0:[]});
col_list = (answers).columns.tolist()

for problem_id in problem_ids:
    train = df_train_clean.loc[(df_train_clean['problem_id'] == problem_id)]['raw_answer_text']
    x_train = df_train_clean_notNaN.loc[(df_train_clean_notNaN['problem_id'] == problem_id) & (df_train_clean_notNaN['dataset'] == 'training')]['raw_answer_text']
    x_test = df_train_clean.loc[(df_train_clean['problem_id'] == problem_id) & (df_train_clean['dataset'] == 'evaluation')]['raw_answer_text']
    y_train = df_train_clean_notNaN.loc[(df_train_clean_notNaN['problem_id'] == problem_id) & (df_train_clean_notNaN['dataset'] == 'training')]['combined_grade']
    # x_train, x_test, y_train, y_test = train_test_split( text, labels, random_state=0, test_size=0.3)
    #bag-of-words feature matrix
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=10000, stop_words='english')

    bow_x_train = bow_vectorizer.fit_transform(x_train)
    bow_x_test = bow_vectorizer.fit_transform(x_test)

    # tf-idf feature matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')#max_df=0.90, min_df=2, max_features=1000,
    tfidf_vectorizer.fit(pd.concat([x_train, x_test]))
    tfidf_x_train = tfidf_vectorizer.transform(x_train)
    tfidf_x_test = tfidf_vectorizer.transform(x_test)
    print(tfidf_x_train.shape)
    print(tfidf_x_test.shape)

    labels = LabelEncoder()
    labels_y_train_bow = labels.fit(y_train)
    # labels_y_test_bow = labels.fit(y_test)
    labels_y_train_tfidf = labels.transform(y_train)
    # labels_y_test_tfidf = labels.transform(y_test)

    print('for problem id : ', problem_id)
    # print('average accuracy of svc = {} \n')  # . format(np.mean(predicted == (labels_y_test_tfidf))))
    print(' the predicted values are: ')
    print('labels', labels.classes_)

    linear_svc = LinearSVC()
    clf = linear_svc.fit(tfidf_x_train, labels_y_train_tfidf)
    calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc, cv="prefit")
    calibrated_svc.fit(tfidf_x_train, labels_y_train_tfidf)
    predicted = calibrated_svc.predict(tfidf_x_test)

    predicted_df = pd.DataFrame(calibrated_svc.predict_proba(tfidf_x_test)*100, columns=labels.classes_)
    predicted_df = predicted_df.loc[:, col_list].fillna(0)
    answers = pd.concat([answers, predicted_df])

    # answers.concat([answers, predicted_df])
    # print(predicted_df)

    # print('rmse : ', np.sqrt(metrics.mean_squared_error(labels_y_test_tfidf, predicted)))
    #
    # fpr, tpr, thresholds = metrics.roc_curve(labels_y_test_tfidf, predicted, pos_label=2)
    # print('auc 2: ', metrics.roc_auc_score(labels_y_test_tfidf, predicted))
    # rmse_array += (np.sqrt(metrics.mean_squared_error(labels_y_test_tfidf, predicted)))
    # auc_array += (metrics.roc_auc_score(labels_y_test_tfidf, predicted))

# print(' rmse : ', rmse_array/(count -1))
# print(' auc : ', auc_array/(count - 1))
answers.to_csv('result.csv')
