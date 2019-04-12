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

print(df_simplified.loc[(df_simplified['problem_id'] == 1619)])

# df_simplified['combined_grade'].fillna(" ", inplace=True)
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

# print(df_training['problem_id'].value_counts())
# print(len(df_training['problem_id'].unique()))

unique_train = df_training['problem_id'].unique()
unique_eval = df_evaluation['problem_id'].unique()

print('\n----------------------\n')
print(df_evaluation['problem_id'].unique().shape)
print(df_evaluation['problem_id'].shape)
print('\n----------------------\n')


# print(' A - B : ', list(set(unique_train)-set(unique_eval)))
# print(' B - A : ', list(set(unique_eval)-set(unique_train)))


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
    from sklearn.model_selection import cross_val_predict
    import matplotlib.pyplot as plt
    # from sklearn import metrics

    count = 0
    rmse_array = 0
    auc_array = 0

    answers = pd.DataFrame({0.0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: [], 'id': []})
    col_list = (answers).columns.tolist()

    special_cases = [496, 1200, 1196, 1619, 1113, 493, 494, 1219, 1316, 1106, 1092, 1332, 1112, 1120, 1169, 1172, 1173, 1182, 1188, 1217,
                     1222, 1228, 1221, 1224]
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

        list_of_y = y_train.unique()

        # if (problem_id == 493):
        #     test_sadfasdf = (min(y_train.value_counts().unique()) > 1)
        #     print('break point here')

        # the third condition 'th_flag' was put in because certain sets had only one graded as 0 and all
        # graded as 1 which meant it couldn't be trained properly
        th_flag = False
        if (len(y_train) > 0) :
            th_flag = (min(y_train.value_counts().unique()) > 1)

        if (len(list_of_y) > 1) and (not problem_id in special_cases) and th_flag:
            # tf-idf feature matrix

            # bag-of-words feature matrix
            bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
            bow_matrix = bow_vectorizer.fit_transform(pd.concat([x_train, x_test]).values.astype('U'))


            print(bow_vectorizer.get_feature_names())
            print('\n----------------------------------------')
            print(bow_matrix.toarray())
            print('\n----------------------------------------')


            tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000,
                                               stop_words='english')  # max_df=0.90, min_df=2, max_features=1000,
            tfidf_vectorizer.fit(bow_matrix)

            tfidf_x_train_train = tfidf_vectorizer.transform(x_train.values.astype('U'))
            tfidf_x_train_test = tfidf_vectorizer.transform(x_test.values.astype('U'))
            # tfidf_x_train_train = tfidf_vectorizer.fit_transform(bow_x_train)
            # tfidf_x_test = tfidf_vectorizer.fit_transform(bow_x_test)

            labels = LabelEncoder()
            labels_y_train_bow = labels.fit(y_train)
            # labels_y_test_bow = labels.fit(y_test)
            labels_y_train_tfidf = labels.transform(y_train)
            labels_y_test_tfidf = labels.transform(y_train_test)

            svc_classifier = SVC(probability=True, kernel='rbf')
            svc_classifier.fit(tfidf_x_train_train, labels_y_train_tfidf)
            # prediction = cross_val_predict(svc_classifier, tfidf_x_train_train, labels_y_train_tfidf, cv=3, method='predict_proba')
            prediction = svc_classifier.predict_proba(tfidf_x_train_test)
            if( len(np.unique(labels_y_test_tfidf)) > 1  and len(labels.classes_) < 3) :
                print('\n ROC-AUC yields : ' + str(roc_auc_score(labels_y_test_tfidf, prediction[:, 1])))
            else:
                print('\n ROC-AUC yields : 1; all the labels were the same thing')

            # fig, ax = plt.subplots()
            # ax.scatter(y_train, prediction, edgecolors=(0, 0, 0))
            # ax.plot([0, 100], [0, 100], 'k--', lw=4)
            # ax.set_xlabel('Measured')
            # ax.set_ylabel('Predicted')
            # plt.show()

            predicted_df = pd.DataFrame(svc_classifier.predict_proba(tfidf_x_test) * 100, columns=labels.classes_)
            # predicted_df['id'] = x_test_id['id'].astype(float)
            x_test_id.reset_index(drop=True, inplace=True)
            predicted_df.reset_index(drop=True, inplace=True)
            predicted_df2 = pd.concat([predicted_df, x_test_id], axis=1)
            predicted_df2 = predicted_df2.loc[:, col_list].fillna(0)

            answers = pd.concat([answers, predicted_df2])

        else:
            # print('all the labels were the same')
            list_input = [100, 0, 0, 0, 0, problem_id]
            df_input = []

            if len(list_of_y) == 1:
                max_occurance = list_of_y[0]
            elif(len(y_train)>0):
                max_occurance = y_train.value_counts().argmax()
            else:
                max_occurance = 0.00

            if not (max_occurance == 0.00 or max_occurance == 0.25):
                list_input = [0, 0, 0, 0, 100, problem_id]

            for i in x_test:
                df_input.append(list_input)

            if len(df_input) > 0:
                predicted_df = pd.DataFrame(df_input, columns=[0.0, 0.25, 0.5, 0.75, 1.0, 'problem_id'])
                x_test_id.reset_index(drop=True, inplace=True)
                predicted_df.reset_index(drop=True, inplace=True)
                predicted_df2 = pd.concat([predicted_df, x_test_id], axis=1)
                answers = pd.concat([answers, predicted_df2])

    print('\n----------------------------------------\n')
    print(x_test.value_counts())
    print('\n----------------------------------------\n')
    answers.to_csv('result.csv')


make_prediction()

