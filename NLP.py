import pandas as pd

df = pd.read_csv('training_set.csv')

df['combined_grade'] = 0

# clean up the training data:
#   remove the questions that have raw_answer_text and cleaned_answer_text as NaN
#   as having those values as NaN doesn't make any sense since what is to grade when
#   there isn't any stuff to actually grade

df_simplified = df[['id', 'problem_id', 'cleaned_answer_text', 'grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4', 'combined_grade', 'dataset']]

print(df_simplified.loc[(df_simplified['problem_id'] == 1619)])

# df_simplified['combined_grade'].fillna(" ", inplace=True)
df_training = df_simplified.loc[(df_simplified['dataset'] == 'training')]
df_evaluation = df_simplified.loc[(df_simplified['dataset'] == 'evaluation')]
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
    from sklearn.calibration import CalibratedClassifierCV
    # from sklearn.model_selection import train_test_split
    # import numpy as np
    # from sklearn import metrics

    count = 0
    rmse_array = 0
    auc_array = 0

    answers = pd.DataFrame({0.0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: [], 'id': []})
    col_list = (answers).columns.tolist()

    special_cases = [496, 1200, 1196, 1113, 210, 1418, 1419, 493, 494, 1219, 1067, 1083, 1427, 1151, 1316, 1106, 1092,
                     1332, 1112, 244, 1120, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1182, 1188, 1217, 1222, 1228,
                     1221, 1243, 1245, 1224, 1684, 1267, 1263, 8889, 1699]
    # 496, 1113 had too many stop words
    # 1200, 1196, 210, 1418 had no features left after pruning

    for problem_id in unique_eval:
        # train = df_train_clean.loc[(df_train_clean['problem_id'] == problem_id)]['raw_answer_text']
        x_train = df_training.loc[(df_training['problem_id'] == problem_id)]['cleaned_answer_text']
        y_train = df_training.loc[(df_training['problem_id'] == problem_id)]['combined_grade']
        x_test = df_evaluation.loc[(df_evaluation['problem_id'] == problem_id)]['cleaned_answer_text']
        x_test_id = df_evaluation.loc[(df_evaluation['problem_id'] == problem_id)][['id']]
        print('for problem id : ', problem_id)
        # print(x_train.value_counts())
        # print(x_train.head())
        # print('most frequent occurance : ', x_train.value_counts().argmax())

        print('\n------------------------------------\n')
        print(answers.shape)
        print(x_test.shape)
        print('\n------------------------------------\n')

        if(problem_id == 1619):
            print("this is where the crash happened")

        list_of_y = y_train.unique()
        if (len(list_of_y) > 1) and (not problem_id in special_cases):
            # tf-idf feature matrix
            # TODO: this logic is wrong empty answers should be given a 0 where as argmax() gives the most frequent answer which might be scored as 1
            #  Solution: track their problem ids and make adjustments to the relevant locations?

            if not x_train.empty and not x_train.isnull().all():
                x_train.fillna(x_train.value_counts().argmax(), inplace=True)

            # problem_id = 32 has NaN in the question as well -_-
            if not x_test.empty and not x_test.isnull().all():
                x_test.fillna(x_test.value_counts().argmax(), inplace=True)

            tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000,
                                               stop_words='english')  # max_df=0.90, min_df=2, max_features=1000,
            tfidf_vectorizer.fit(pd.concat([x_train, x_test]))
            tfidf_x_train = tfidf_vectorizer.transform(x_train)
            tfidf_x_test = tfidf_vectorizer.transform(x_test)

            labels = LabelEncoder()
            labels_y_train_bow = labels.fit(y_train)
            # labels_y_test_bow = labels.fit(y_test)
            labels_y_train_tfidf = labels.transform(y_train)
            # labels_y_test_tfidf = labels.transform(y_test)

            # print('average accuracy of svc = {} \n')  # . format(np.mean(predicted == (labels_y_test_tfidf))))
            # print(' the predicted values are: ')
            # print('labels', labels.classes_)

            linear_svc = LinearSVC()
            clf = linear_svc.fit(tfidf_x_train, labels_y_train_tfidf)
            calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc, cv="prefit")
            calibrated_svc.fit(tfidf_x_train, labels_y_train_tfidf)
            predicted = calibrated_svc.predict(tfidf_x_test)

            predicted_df = pd.DataFrame(calibrated_svc.predict_proba(tfidf_x_test) * 100, columns=labels.classes_)
            # predicted_df['id'] = x_test_id['id'].astype(float)
            x_test_id.reset_index(drop=True, inplace=True)
            predicted_df.reset_index(drop=True, inplace=True)
            predicted_df2 = pd.concat([predicted_df, x_test_id], axis=1)
            predicted_df2= predicted_df2.loc[:, col_list].fillna(0)

            answers = pd.concat([answers, predicted_df2])

            # answers.concat([answers, predicted_df])
            # print(predicted_df)

            # print('rmse : ', np.sqrt(metrics.mean_squared_error(labels_y_test_tfidf, predicted)))
            #
            # fpr, tpr, thresholds = metrics.roc_curve(labels_y_test_tfidf, predicted, pos_label=2)
            # print('auc 2: ', metrics.roc_auc_score(labels_y_test_tfidf, predicted))
            # rmse_array += (np.sqrt(metrics.mean_squared_error(labels_y_test_tfidf, predicted)))
            # auc_array += (metrics.roc_auc_score(labels_y_test_tfidf, predicted))
        else:
            # print('all the labels were the same')
            list_input = [100, 0, 0, 0, 0, problem_id]
            df_input = []

            if len(list_of_y) == 1:
                max_occurance = list_of_y[0]
            else:
                max_occurance = y_train.value_counts().argmax()

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







    # print(' rmse : ', rmse_array/(count -1))
    # print(' auc : ', auc_array/(count - 1))
    print('\n----------------------------------------\n')
    print(x_test.value_counts())
    print('\n----------------------------------------\n')
    answers.to_csv('result.csv')


make_prediction()

