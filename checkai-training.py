
import numpy as np
import pandas as pd
import ast
import joblib

df=pd.read_csv ('.\Dataset\_fake_reviews_dataset.csv')

print(df.shape)
df = df.drop(columns=["category",'rating'],axis=1)
# print(df)

# def extract_element(text):
#     try:
#         # Use ast.literal_eval to safely evaluate the string as a list
#         elements = ast.literal_eval(text)
#         # Return the first element of the list
#         return elements[0]
#     except (SyntaxError, ValueError):
#         return None

# df['paraphrased'] = df['paraphrases'].apply(extract_element)

# df['paraphrased'] = df['paraphrased'].apply(lambda x: x + " /chatgpt-generation/")

# df = df.drop(columns="paraphrases")


# stacked_df = pd.concat([df['text'], df['paraphrased']], ignore_index=True)
# stacked_df = stacked_df.rename('stacked').to_frame()

# print(stacked_df)



print(df['label'].value_counts())
stacked_df=df

stacked_df['label'] = np.where(stacked_df['label'].str.contains("CG"), "ai", "human")
stacked_df=stacked_df.sample(frac=1)
#stacked_df=stacked_df[:20000]
print(stacked_df)
X=stacked_df['text_']
y=stacked_df['label']


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
joblib.dump(vectorizer,'vectorizer2.pkl')
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.ensemble import ExtraTreesClassifier




etc = ExtraTreesClassifier(n_estimators=50,random_state=2)



def prediction(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    print(model)
    joblib.dump(model,'.\model\model2.pkl')
    pr = model.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, pr)
    f1 = metrics.f1_score(y_test, pr, average="binary", pos_label="ai")
    
    return acc_score, f1

acc_score = {}
f1_score = {}

# clfs = {
#     'LR': lg,
#     'SVM': sv,
#     'DTC': dtc,
#     'KNN': knn,
#     'RFC': rfc,
#     'ETC': etc,
#     'ABC': abc,
#     'BG': bg,
#     'GBC': gbc,
# }

clfs={'ETC':etc}


for name, clf in clfs.items():
    print('Training ',name,"model")
    acc_score[name], f1_score[name] = prediction(clf, X_train_tfidf, X_test_tfidf, y_train, y_test)

# View those scores
for name, acc in acc_score.items():
    print(f'Accuracy for {name}: {acc}')

for name, f1 in f1_score.items():
    print(f'F1 score for {name}: {f1}')


#etc.fit(X_train_tfidf,y_train)



