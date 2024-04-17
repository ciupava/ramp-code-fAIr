#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################


# Standard library imports
import argparse
import csv
import os
import random
import re
import shutil
import sys
from pathlib import Path

# Third party imports
import tqdm

RAMP_HOME = os.environ["RAMP_HOME"]

# Standard library imports
# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(
        description=""" 
    Create randomized lists of files for training and validation datasets, and save them to files.
    Test datasets should already have been set aside.

    Example: make_train_val_split_lists.py -src chips -trn 0.70 -val 0.15 -prd 0.15
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-src",
        "--src_dir",
        type=str,
        required=True,
        help=r"Path to directory containing TQ geotiffs.",
    )
    parser.add_argument(
        "-pfx",
        "--list_prefix",
        type=str,
        required=False,
        default="datasplit",
        help=r"prefix for train, validation and test file lists",
    )
    parser.add_argument(
        "-trn",
        "--train_fraction",
        type=float,
        required=True,
        help="fraction of files to assign to training set",
    )
    parser.add_argument(
        "-val",
        "--val_fraction",
        type=float,
        required=False,
        default=0.0,
        help="fraction of files to assign to validation set",
    )
    parser.add_argument(
        "-prd",
        "--pred_fraction",
        type=float,
        required=True,
        help="fraction of files to assign to prediction set",
    )
    args = parser.parse_args()

    trn = args.train_fraction
    val = args.val_fraction
    prd = args.pred_fraction

    # force fractions to sum to 1
    normalizer = trn + val + prd
    if normalizer == 0.0:
        raise ValueError("train, validation and prediction fractions must sum to 1")
    trn = trn / normalizer
    val = val / normalizer
    prd = prd / normalizer

    # check that source directory exists and is readable
    src_dir = args.src_dir
    if not Path(src_dir).is_dir():
        raise ValueError(f"source directory {src_dir} is not readable")

    # construct filenames for file lists
    file_rootname = args.list_prefix
    trn_csv = file_rootname + "_train.csv"
    val_csv = file_rootname + "_val.csv"
    prd_csv = file_rootname + "_pred.csv"

    # construct list of all filenames in src dir, shuffle in place
    files = list(Path(src_dir).glob("**/*"))
    random.shuffle(files)
    numfiles = len(files)
    log.debug(f"Total no of files {numfiles}")
    if numfiles < 2:
        raise ValueError("Too small training dataset")

    # get numbers of files
    numval = int(val * numfiles)
    if numval < 1:
        numval = 1
    log.debug(f"Num of validation {numval}")
    
    numprd = int(prd * numfiles)
    if numprd < 1:
        numprd = 1
    log.debug(f"Num of prd {numprd}")
    
    numtrain = numfiles - numval - numprd
    log.debug(f"No of trainings files {numtrain}")

    trainlist = files[:numtrain]
    vallist = files[numtrain:numtrain+numval]
    prdlist = files[numtrain+numval:]

    with open(trn_csv, "w") as trnfp:
        log.info(f"Writing {trn_csv}")
        trnfp.write("\n".join([str(item) for item in trainlist]))

    if numval != 0:
        with open(val_csv, "w") as valfp:
            log.info(f"Writing {val_csv}")
            valfp.write("\n".join([str(item) for item in vallist]))

    if numprd != 0:
        with open(prd_csv, "w") as prdfp:
            log.info(f"Writing {prd_csv}")
            prdfp.write("\n".join([str(item) for item in prdlist]))



if __name__ == "__main__":
    main()
