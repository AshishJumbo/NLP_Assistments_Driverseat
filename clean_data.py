import pandas as pd

df = pd.read_csv('training_set.csv')
print(df.head())

# remove html tags from the raw answer
import re
import numpy as np

def remove_html_contents (input_txt):
    TAG_HTML = re.compile(r'<.*?>')
    TAG_SPECIAL_CHAR = re.compile(r'&.*?;')
    TAG_R = re.compile(r'\s')

    for index, text in enumerate(input_txt):
        if pd.isnull(text):
            continue
        text = re.sub(TAG_R, ' ', text)
        text = re.sub(TAG_HTML, ' ', text)
        text = re.sub(TAG_SPECIAL_CHAR, ' ', text)
        input_txt[index] = text
    return input_txt

# remove html tags
columns_to_clean = [ 'question_text', 'raw_answer_text', 'cleaned_answer_text']
# columns_to_clean = [ 'raw_answer_text' ]

for column_name in columns_to_clean:
    # df[column_name+'_clean'] = remove_html_contents(df[column_name])
    print("\n", column_name, ": \n", df[column_name][:5])
    remove_html_contents(df[column_name])
    print("\n",column_name, ": \n",df[column_name][:5])

df.to_csv('cleaned.csv')