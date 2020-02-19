import os
import sys
import codecs
import chardet
import chardet

DIR_PATHS = [r"../data/books_utf_8"]
VOCAB_FILE = "../data/vocab.txt"


def main():
    words = set()
    for DIR_PATH in DIR_PATHS:
        for i, filename in enumerate(os.listdir(DIR_PATH)):
            f_path = os.path.join(DIR_PATH, filename)
            with open(f_path, "r+", encoding="utf-8") as f:
                w = f.read(1)
                while w:

                    if w == '\n' or w == '\r' or w == '\t' or w == ' ':
                        pass
                    else:
                        words.add(w)
                    w = f.read(1)

    with open(VOCAB_FILE, "w+", encoding="utf-8") as f:
        f.write("[Start] [Enter] [Unk] [Space] [End] ")
        f.write(" ".join(words))
        f.flush()
        print("字去重已完成！")


main()
