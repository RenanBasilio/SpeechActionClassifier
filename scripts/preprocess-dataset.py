from math import floor, ceil
from datetime import date
from pathlib import Path
from termcolor import cprint, colored
import shutil
import pandas as pd
import sys
import os
import subprocess

log_file = open("log.txt", "w+")

errors = ['Partial', '!! BAD', 'Inaudible', 'Maybe']

def print_progress(curr, total):
    bars = floor((curr/total)*10)*'|'
    dashes = (11 - ceil((curr/total)*10))*'='
    print("\r[{}{}]".format(bars, dashes), end="")

def count_entries(class_files):
    total = 0
    for file in class_files:
        total += sum(1 for line in open(file))
    return total

def get_classes(classes_file):
    with open(classes_file) as classes_file:
        CLASS_NAMES = classes_file.read().splitlines()
    return CLASS_NAMES

def get_splits(splits_file):
    splits = {
        'Names': []
    }
    with open(splits_file) as splits_file:
        for line in splits_file.read().splitlines():
            name, members = line.split()
            for member in members.split(","):
                splits[member] = name
            if name not in splits['Names']:
                splits['Names'].append(name)
    return splits

def sort_dataset(splits, classes, unsorted_dir, out_dir):
    # Create directories for splits
    for split in splits["Names"]:
        (out_path / split).mkdir(parents=True, exist_ok=True)

    manual_classes = list(unsorted_path.glob("**/_classes.csv"))
    count = 0
    copied = 0
    skipped = 0
    total = count_entries(manual_classes)

    for csv in manual_classes:
        data = pd.read_csv(csv, header=None)
        
        # Directory with video segments corresponding to this csv file
        seg_dir = csv.parent / 'segments'

        # Set output root directory
        curr = str(csv.relative_to(unsorted_path).parent)
        out_dir = out_path / splits[curr] / curr
        out_dir.mkdir(parents=True, exist_ok=True)

        curr_splits = 0

        # Create output directories for each class
        for c in classes:
            (out_dir / c).mkdir(parents=True, exist_ok=True)

        # For each row of the manual classification file
        for index, row in data.iterrows():
            infile = seg_dir / "{:06d}.mp4".format(row[0])
            outfile = out_dir / row[1] / "{:06d}.mp4".format(row[0])

            print_progress(count, total)

            # If second row contains an error type, skip
            if pd.notnull(row[2]) and any([x in row[2] for x in errors]):
                print("", end="\r")
                cprint(" Skipped ({:06d}, {}, {})             ".format(row[0], row[1], row[2]), 'red', end="", flush=True)
                log_file.write("Skipped copying file {} with entry ({:06d}, {}, {}). Reason: Matched a known error subclass.\n".format(infile, row[0], row[1], row[2]))
                skipped += 1

            # If second row contains a speaker change, split video at change and skip
            elif pd.notnull(row[2]) and "Speaker Change" in row[2]:
                print("", end="\r")
                cprint("Found Speaker Change in file {} at {:06d}. Splitting data into new set.".format(csv, row[0]), 'blue', flush=True)
                curr_splits += 1
                curr =  str(csv.relative_to(unsorted_path).parent) + "-" + str(curr_splits)
                out_dir = out_dir.parent / curr
                out_dir.mkdir(parents=True, exist_ok=True)
                for c in classes:
                    (out_dir / c).mkdir(parents=True, exist_ok=True)
                log_file.write("Split video dataset defined in {} at {} due to speaker change. New ID: {}\n".format(csv, row[0],  curr))
                skipped += 1

            # If second row is not null, show a message but copy anyways
            elif pd.notnull(row[2]):
                cprint(" Copying ({:06d}, {}, {})             ".format(row[0], row[1], row[2]), 'yellow', end="", flush=True)
                shutil.copy(str(infile), str(outfile))
                copied += 1

            # If first row is a class valid class name, copy file to output directory
            elif row[1] in classes:
                print(" Copying ({:06d}, {})...               ".format(row[0], row[1]), end="", flush=True)
                shutil.copy(str(infile), str(outfile))
                copied += 1

            # Otherwise, show an error and skip
            else:
                print("", end="\r")
                cprint("Failed to copy ({:06d}, {}, {}): Unknown class '{}'".format(row[0], row[1], row[2], row[1]), 'red', flush=True)
                log_file.write("Skipped copying file {} with entry ({:06d}, {}, {}). Reason: Unknown class {}.\n".format(infile, row[0], row[1], row[2], row[1]))
                skipped += 1
            count += 1

    print("\rDone. Copied {} files with {} classes. ({} skipped, see log file for details)".format(copied, len(classes), skipped))
    log_file.close()

def count_samples(data_dir):
    classes = {
        "Errors": 0
    }
        
    classes_files = list(data_dir.glob("**/_classes.csv"))

    for f in classes_files:
        data = pd.read_csv(f, header=None)
        for index, row in data.iterrows():
            if row[1] not in classes:
                classes[row[1]] = 0
            
            if pd.notnull(row[2]) and any([x in row[2] for x in errors]):
                classes["Errors"] += 1
                continue
            else:
                classes[row[1]] += 1
            
            if pd.notnull(row[2]) and not any([x in row[2] for x in errors]):
                for c in row[2].split(", "):
                    if c not in classes:
                        classes[c] = 0
                    classes[c] += 1

    return classes

def print_usage():
    print("Usage: preprocess_dataset.py <command> <dataset_directory>")
    print("")
    print("Commands:")
    print("package - Package dataset into a tarball")
    print("sort    - Sort dataset into directories according to splits.txt and classes.txt")
    print("update  - Updates dataset.info in dataset directory")

def print_info(data_dir):
    counts = count_samples(data_dir)
    counts_sorted = sorted(counts, key=counts.get, reverse=True)

    info_file = data_dir / "dataset.info"
    with open(info_file, "w+", encoding='utf-8') as info_file:
        info_file.write("Este arquivo enumera e define as classes de segmentos de vídeo presentes no dataset:\n\n")
        info_file.write("Idle    - O depoente se encontra parado ou realizando outra ação irrelevante ao propósito do reconhecedor.\n")
        info_file.write("Speak   - O depoente se encontra falando.\n")
        info_file.write("Nod     - O depoente balança a cabeça como se para dizer que sim.\n")
        info_file.write("Shake   - O depoente balança a cabeça como se para dizer que não.\n")
        info_file.write("Examine - O depoente se aproxima da câmera para examinar alguma evidência que lhe é apresentada.\n")
        info_file.write("\n")
        info_file.write("As seguintes classes estão sob análise devido à falta de vídeos representativos da mesma nos dados diarizados até agora.\n")
        info_file.write("\n")
        info_file.write("KindOf - O depoente balança a cabeça como que para dizer que \"mais ou menos\"\n")
        info_file.write("\n")
        info_file.write("Total de Amostras por Classe (Atualizado em {}):\n".format(date.today()))
        for c in counts_sorted:
            if c is not "Errors":
                info_file.write("{}: {}\n".format(c, counts[c]))
        info_file.write("\n")
        info_file.write("Total de Amostras Inutilizáveis: {}".format(counts["Errors"]))

if __name__ == '__main__':
    if len(sys.argv) > 2:
        data_dir = Path(sys.argv[2]).resolve()

        unsorted_path = data_dir / "unsorted"
        out_path = data_dir
        classes_file = data_dir / "classes.txt"
        splits_file = data_dir / "splits.txt"

        if sys.argv[1] == "sort":

            if data_dir.is_dir() and (classes_file).exists() and (splits_file).exists() and (unsorted_path).is_dir():
                print("Sorting dataset...")
                print("Classes: {} ({})".format(get_classes(classes_file), classes_file))
                print("Splits: {} ({})".format(get_splits(splits_file)["Names"], splits_file))

                sort_dataset(get_splits(splits_file), get_classes(classes_file), unsorted_path, out_path)

            else:
                print_usage()
        
        elif sys.argv[1] == "package":
            if sys.platform == "linux" or sys.platform == "linux2":
                subprocess.call([format(os.path.dirname(os.path.realpath(sys.argv[0]))) + "/package_dataset.sh", data_dir])
            else:
                cprint("Command only available in linux. Current platform: {}".format(sys.platform), 'red')

        elif sys.argv[1] == "update":
            print_info(data_dir)
        else:
            print_usage()

    else:
        print_usage()
