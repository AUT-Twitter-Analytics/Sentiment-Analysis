import pickle
import numpy as np
import pandas as pd
from preprocess import Preprocess


# read csv file or fetch data
data = []
address = "/home/rezvan/Desktop/ss-project/tweets.csv"
data = pd.read_csv(address, encoding="utf-8")

# preprocess data
processor = Preprocess()
processor.fit(data)

# # merge
# frames = [processor.en_data, processor.fa_data]
# data = pd.concat(frames)

# # shuffle data
# data = data.sample(frac=1)
# data.reset_index(drop=True, inplace=True)

X = data.text.apply(lambda x: np.str_(x))

# vectorize data
loaded_vectorizer = pickle.load(
    open("/home/rezvan/Desktop/ss-project/tfidf_fa.csv", "rb")
)
X_vector = tfidfVectorizer.transform(X)

# rf model
loaded_rf_model = pickle.load(
    open("/home/rezvan/Desktop/ss-project/finalized_rf_fa_model.csv", "rb")
)
prediction = loaded_rf_model.predict(X_vector)
