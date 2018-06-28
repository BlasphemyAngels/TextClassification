from fastText import FastText
from sklearn.model_selection import KFold
import numpy as np
from utils import read


class TCFastText(object):


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

    #  ft = FastText.train_supervised("data/ft_train", dim=128, epoch=60, minCount=4, wordNgrams=5, label="__label__")
    ft = FastText.train_supervised("data/ft_train", dim=128, epoch=60, minCount=5, wordNgrams=3, label="__label__")
    #  ft = FastText.train_supervised("data/ft_train", dim=80, epoch=60, minCount=5, wordNgrams=3, label="__label__")

    result = ft.test("data/ft_test")

    ft.save_model("fastText")

    print(result)

    break
