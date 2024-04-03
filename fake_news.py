import streamlit as st
import pandas as pd
import re
import time
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# clean the text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# split the data
df_fake["class"] = 0
df_true["class"] = 1
df = pd.concat([df_fake.head(1000), df_true.head(1000)], axis=0)
df['text'] = df['text'].apply(wordopt)
x = df['text']
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# vectorize the text
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# train models
LR = LogisticRegression().fit(xv_train, y_train)
DT = DecisionTreeClassifier().fit(xv_train, y_train)
GB = GradientBoostingClassifier(random_state=0).fit(xv_train, y_train)
RF = RandomForestClassifier(random_state=0).fit(xv_train, y_train)

# use models for prediction
def manual_testing(news):
    with st.spinner('Classifying...'):
        testing_news = {'text': [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test['text']
        new_xv_test = vectorization.transform(new_x_test)
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GB = GB.predict(new_xv_test)
        pred_RF = RF.predict(new_xv_test)

    return {
        "LR": "True" if pred_LR[0] == 1 else "Fake",
        "DT": "True" if pred_DT[0] == 1 else "Fake",
        "GB": "True" if pred_GB[0] == 1 else "Fake",
        "RF": "True" if pred_RF[0] == 1 else "Fake"
    }

# streamlit application
def main():
    st.title("Fake News Classifier")
    news_text = st.text_area("Enter News Text", height=200)
    if st.button("Classify"):
        # progress_bar = st.progress(0)
        # for percentage_complete in range(100):
        #     time.sleep(0.1)
        #     progress_bar.progress(percentage_complete + 1)
        results = manual_testing(news_text)
        # progress_bar.progress(100) 
        st.write(f"LR Prediction: {results['LR']}")
        st.write(f"DT Prediction: {results['DT']}")
        st.write(f"GB Prediction: {results['GB']}")
        st.write(f"RF Prediction: {results['RF']}")

if __name__ == "__main__":
    main()