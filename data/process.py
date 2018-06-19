import os
import sys
import re
import random

def clean(text):
    text = re.sub("(.*?)", "", text)


if __name__ == '__main__':
    with open("./adult_content.txt", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.rstrip("\n"), lines))

        lines_n = []
        for line in lines:
            line = line.split("\t")

            line = " ".join(line[-4:])
            line += "\t0"
            lines_n.append(line)

    with open("./normal_content.txt", "r") as f:
        lines_p = f.readlines()
        lines_p = list(map(lambda x: x.rstrip("\n"), lines_p))
        lines_p = list(map(lambda x: " ".join(x.split("\t")), lines_p))
        lines_p = list(map(lambda x: x + "\t1", lines_p))

    lines_all = lines_p + lines_n

    random.shuffle(lines_all)

        
    with open("data.txt", "w") as f:
        f.write("\n".join(lines_all))

