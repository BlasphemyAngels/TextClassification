import os
import sys
import re
import random

import jieba
from utils import cut

def clean(text):
    text = re.sub("(.*?)", "", text)


if __name__ == '__main__':
    use_stop = True
    stop_file = "./data/stop.txt"
    with open(stop_file, "r") as f:
        stop_list = f.readlines()
        stop_list = list(map(lambda x: x.rstrip("\n"), stop_list))
    with open("./data/adult_content.txt", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.rstrip("\n"), lines))

        lines_n = []
        for line in lines:
            line = line.split("\t")

            line = " ".join(line[-4:])
            #  line = "".join(line.split())
            line = " ".join(line.split())
            line = " ".join(list(cut([line], use_stop=use_stop, stop_list=stop_list)))
            line += "\t1"
            lines_n.append(line)

    with open("./data/normal_content.txt", "r") as f:
        lines_p = f.readlines()
        lines_p = list(map(lambda x: x.rstrip("\n"), lines_p))
        lines_p = list(map(lambda x: " ".join(x.split()), lines_p))
        lines_p = cut(lines_p, use_stop=use_stop, stop_list=stop_list)
        lines_p = list(map(lambda x: x + "\t0", lines_p))

    lines_all = lines_n + lines_p

    random.shuffle(lines_all)

    with open("data/data.txt", "w") as f:
        f.write("\n".join(lines_all))

    with open("./tmp_test_content.txt", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.rstrip("\n"), lines))
        lines = list(map(lambda x: " ".join(x.split()), lines))
        lines = cut(lines, use_stop=use_stop, stop_list=stop_list)

    with open("./tmp_test_content", "w") as f:
        f.write("\n".join(lines))
