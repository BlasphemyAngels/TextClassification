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

    with open("data/ft_train", "w") as f:
        f.write("\n".join(train))

    with open("data/ft_test", "w") as f:
        f.write("\n".join(test))

    classifier = fasttext.supervised("ft_train", "fasttext_model", label_prefix="__label__")

    result = classifier.test("ft_test")

    print(result.precision)
    print(result.recall)

