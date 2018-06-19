import os
import sys

import logging
import argparse

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
    logging.root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, required=True, help="The path of data")

    args, _ = parser.parse_known_args()
