import pickle
import numpy as np
import pandas as pd
from preprocess import Preprocess
from sklearn.metrics import classification_report


# read csv file or fetch data

address = "/home/rezvan/Desktop/ss-project/X_test.sav"
X = pickle.load(open(address, "rb"))

address = "/home/rezvan/Desktop/ss-project/y_test.sav"
y = pickle.load(open(address, "rb"))

# # preprocess data
# processor = Preprocess()
# processor.fit(X)

# # merge
# frames = [processor.en_data, processor.fa_data]
# data = pd.concat(frames)

# # shuffle data
# data = data.sample(frac=1)
# data.reset_index(drop=True, inplace=True)

# X = data.text.apply(lambda x: np.str_(x))

# vectorize data
loaded_vectorizer = pickle.load(
    open("/home/rezvan/Desktop/ss-project/Sentiment-Analysis/tfidf_bilang.sav", "rb")
)
X_vector = loaded_vectorizer.transform(X)

# rf model
loaded_rf_model = pickle.load(
    open("/home/rezvan/Desktop/ss-project/finalized_rf_model_bilang.sav", "rb")
)
prediction = loaded_rf_model.predict(X_vector)

print(classification_report(y, prediction))
