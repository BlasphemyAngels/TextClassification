import fasttext
from sklearn.model_selection import KFold
import numpy as np
from utils import read

texts, labels = read("./data/cut_data.txt")

labels = list(map(lambda x: "__label__" + str(x), labels))

data = zip(texts, labels)


data = list(map(lambda x: " ".join(x), data))
data = np.array(data)


kf = KFold(n_splits=8)

for train_index, test_index in kf.split(data):

    print("Train:", train_index, "Test:", test_index)

    train = data[train_index]
    test = data[test_index]



    classifier = fasttext.supervised("a.txt", "fasttext_model", label_prefix="__label__")

