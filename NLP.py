import pandas as pd

df = pd.read_csv('training_set.csv')

df['combined_grade'] = 0

# clean up the training data:
#   remove the questions that have raw_answer_text and cleaned_answer_text as NaN
#   as having those values as NaN doesn't make any sense since what is to grade when
#   there isn't any stuff to actually grade

df_simplified = df[
    ['id', 'problem_id', 'cleaned_answer_text', 'grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4', 'combined_grade',
     'dataset']]

df_training = df_simplified.loc[(df_simplified['dataset'] == 'training')] # shape(145638, 10)
df_evaluation = df_simplified.loc[(df_simplified['dataset'] == 'evaluation')] # shape(2000, 10)
print(df_training.shape)
print(df_evaluation.shape)

df_training = df_training[pd.notnull(df_training['grade_0'])]

df_training.loc[df_training.grade_0 == 1, 'combined_grade'] = 0.0
df_training.loc[df_training.grade_1 == 1, 'combined_grade'] = 0.25
df_training.loc[df_training.grade_2 == 1, 'combined_grade'] = 0.5
df_training.loc[df_training.grade_3 == 1, 'combined_grade'] = 0.75
df_training.loc[df_training.grade_4 == 1, 'combined_grade'] = 1.0

unique_train = df_training['problem_id'].unique()
unique_eval = df_evaluation['problem_id'].unique()

def make_prediction():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict
    from sklearn.naive_bayes import MultinomialNB

    answers = pd.DataFrame({0.0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: [], 'id': []})
    col_list = (answers).columns.tolist()

    special_cases = []
    # 83 has all ones only one 0
    # 496, 494, 1092, 1120, 1217, 1228 only has stop words
    # 1619 empty
    # 493, 494, 1219, 1316, 1106, 1332, 1112, 1169, 1172, 1173, 1182, 1188, 1222, 1228, 1221, 1224 no terms remain after pruning

    for problem_id in unique_eval:
        # train = df_train_clean.loc[(df_train_clean['problem_id'] == problem_id)]['raw_answer_text']
        x_train = df_training.loc[(df_training['problem_id'] == problem_id)]['cleaned_answer_text']
        y_train = df_training.loc[(df_training['problem_id'] == problem_id)]['combined_grade']
        x_test = df_evaluation.loc[(df_evaluation['problem_id'] == problem_id)]['cleaned_answer_text']
        x_test_id = df_evaluation.loc[(df_evaluation['problem_id'] == problem_id)][['id']]
        print('for problem id : ', problem_id)

        X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, max_features=1000, stop_words='english')
        tfidf_vectorizer.fit(x_train.values.astype('U'))
        tfidf_X_train = tfidf_vectorizer.transform(X_train.values.astype('U'))
        tfidf_X_test = tfidf_vectorizer.transform(X_test.values.astype('U'))

        print(tfidf_vectorizer.get_feature_names())
        print(tfidf_X_train.shape)
        print(tfidf_X_test)

        labels_encoder = LabelEncoder()
        labels_encoder.fit(Y_train)
        labels_Y_train = labels_encoder.transform(Y_train)
        labels_Y_test = labels_encoder.transform(Y_test)

        clf = MultinomialNB().fit(tfidf_X_train, labels_Y_train)

        predicted = clf.predict(tfidf_X_test)
        print("\n -------------------------- \n result:", np.mean(predicted == labels_Y_test))

    # answers.to_csv('result.csv')


make_prediction()

