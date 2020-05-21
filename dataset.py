from math import floor, ceil
from datetime import date, datetime
from termcolor import cprint, colored
from modules.loader import load_video_as_ndarray
from modules.utils import print_progress, write_tee
import dlib
import cv2 as cv2
import shutil
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import tarfile
import zipfile
import argparse
import pytz
import pathlib

log_file = open("log.txt", "w+")

errors = ['Partial', '!! BAD', 'Inaudible', 'Maybe']

face_detector = None

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
            name, members = line.split(maxsplit=1)
            for member in members.split(" "):
                splits[member] = name
            if name not in splits['Names']:
                splits['Names'].append(name)
    return splits

def sort_dataset(splits, classes, unsorted_dir, out_dir):
    # Create directories for splits
    for split in splits["Names"]:
        (out_path / split).mkdir(parents=True, exist_ok=True)

    classes_files = list(unsorted_path.glob("**/_classes.csv"))
    manual_classes = [k for k in classes_files if str(k.relative_to(unsorted_path).parent) in splits]
    count = 0
    copied = 0
    skipped = 0
    total = count_entries(manual_classes)

    for csv in manual_classes:
        data = pd.read_csv(csv, dtype={'id': np.int32})
        
        # Directory with video segments corresponding to this csv file
        seg_dir = csv.parent / 'segments'

        # Set output root directory
        curr = str(csv.relative_to(unsorted_path).parent)
        if curr in splits:
            out_dir = out_path / splits[curr] / curr
            out_dir.mkdir(parents=True, exist_ok=True)

            curr_splits = 0

            # Create output directories for each class
            for c in classes:
                (out_dir / c).mkdir(parents=True, exist_ok=True)

            # For each row of the manual classification file
            for index, row in enumerate(data.itertuples()):
                infile = seg_dir / "{:06d}.mp4".format(row.id)

                print_progress(count, total, end="")

                # If second row contains an error type, skip
                if pd.notnull(row.notes) and any([x in row.notes for x in errors]):
                    cprint(" Skipped ({:06d}, {}, {})             ".format(row.id, row.tag, row.notes, row.validated), 'red', end="\r", flush=True)
                    log_file.write("Skipped copying file {} with entry ({:06d}, {}, {}). Reason: Matched a known error subclass.\n".format(infile, row.id, row.tag, row.notes))
                    skipped += 1

                # If second row contains a speaker change, split video at change and skip
                elif pd.notnull(row.notes) and "Speaker Change" in row.notes:
                    print("", end="\r")
                    cprint("Found Speaker Change in file {} at {:06d}. Splitting data into new set.".format(csv, row.id), 'blue', flush=True)
                    curr_splits += 1
                    curr =  str(csv.relative_to(unsorted_path).parent) + "-" + str(curr_splits)
                    out_dir = out_dir.parent / curr
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for c in classes:
                        (out_dir / c).mkdir(parents=True, exist_ok=True)
                    log_file.write("Split video dataset defined in {} at {} due to speaker change. New ID: {}\n".format(csv, row.id,  curr))
                    skipped += 1

                # If validation failed, skip
                elif pd.isnull(row.validated) or row.validated == False:
                    cprint(" Skipped ({:06d}, {}, {})             ".format(row.id, row.tag, row.notes, row.validated), 'red', end="\r", flush=True)
                    log_file.write("Skipped copying file {} with entry ({:06d}, {}, {}). Reason: Validation failure.\n".format(infile, row.id, row.tag, row.notes))
                    skipped += 1

                # If second row is not null, show a message but copy anyways
                elif pd.notnull(row.notes):
                    cprint(" Copying ({:06d}, {}, {})             ".format(row.id, row.tag, row.notes, row.validated), 'yellow', end="\r", flush=True)
                    outfile = out_dir / row.tag / "{:06d}.mp4".format(row.id)
                    shutil.copy(str(infile), str(outfile))
                    copied += 1

                # If first row is a valid class name, copy file to output directory
                elif row.tag in classes:
                    print(" Copying ({:06d}, {})...               ".format(row.id, row.tag), end="\r", flush=True)
                    outfile = out_dir / row.tag / "{:06d}.mp4".format(row.id)
                    shutil.copy(str(infile), str(outfile))
                    copied += 1

                # Otherwise, show an error and skip
                else:
                    print("", end="\r")
                    cprint("Failed to copy ({:06d}, {}, {}): Unknown class '{}'".format(row.id, row.tag, row.notes, row.tag), 'red', flush=True)
                    log_file.write("Skipped copying file {} with entry ({:06d}, {}, {}). Reason: Unknown class {}.\n".format(infile, row.id, row.tag, row.notes, row.tag))
                    skipped += 1
                
                count += 1
                sys.stdout.flush()

    print("\rDone. Copied {} files with {} classes. ({} skipped, see log file for details)                      ".format(copied, len(classes), skipped))
    log_file.close()

def count_samples(data_dir):
    classes = {
        "Errors": 0,
        "Validated": 0
    }
        
    classes_files = list(data_dir.glob("**/_classes.csv"))

    for f in classes_files:
        data = pd.read_csv(f, dtype={'id': np.int32})
        for index, row in enumerate(data.itertuples()):
            if pd.notnull(row.tag) and row.tag not in classes:
                classes[row.tag] = 0
            
            if pd.notnull(row.notes) and any([x in row.notes for x in errors]) :
                classes["Errors"] += 1
                continue
            
            if pd.notnull(row.tag):
                classes[row.tag] += 1
                if 'validated' in data and row.validated is True:
                    classes["Validated"] += 1

            if pd.notnull(row.notes) and not any([x in row.notes for x in errors]):
                for c in row.notes.split(", "):
                    if c not in classes:
                        classes[c] = 0
                    classes[c] += 1

    return classes

def validate(data_dir, revalidate=False, verbose=True):
    classes_files = list(data_dir.glob("**/_classes.csv"))
    count = 0
    validated = 0
    failed = 0

    for f in classes_files:
        data = pd.read_csv(f, dtype={'id': 'int32'})
        
        if revalidate or  'validated' not in data or data['validated'].isnull().sum() > 0:
            curr = str(f.relative_to(unsorted_path).parent)
            data['validated'] = np.nan

            for index, row in enumerate(data.itertuples()):
                if revalidate or pd.isnull(row.validated):
                    count += 1
                    infile = f.parent / 'segments' / "{:06d}.mp4".format(row.id)

                    if validate_face(infile):
                        cprint("File {} successfully validated.".format(infile), 'green')
                        data.loc[index, 'validated'] = True
                        validated += 1
                    else:
                        cprint("File {} failed to validate.".format(infile), 'red')
                        data.loc[index, 'validated'] = False
                        failed += 1

            f.replace(f.parent / "_classes.csv.old")
            data.to_csv(f, index=False)
            
            cprint("Wrote validation results to {}.".format(f), 'cyan')

    print("Validated {} files ({} succeeded, {} failed).".format(count, validated, failed))

def validate_face(file):
    global face_detector
    if face_detector is None:
        face_detector = dlib.get_frontal_face_detector()
        #face_detector = dlib.cnn_face_detection_model_v1("resources/mmod_human_face_detector.dat")

    try:
        video = load_video_as_ndarray(file, color_mode='raw', optical_flow=False, warnings='except')

        for frame in video:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector(frame, 1)
            if len(list(enumerate(faces))) == 0:
                return False
        
        return True
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        return False


def update_info(data_dir, verbose=True):
    counts = count_samples(data_dir)
    counts_sorted = sorted(counts, key=counts.get, reverse=True)

    info_file = data_dir / "dataset.info"
    with open(info_file, "w+", encoding='utf-8') as info_file:
        write_tee(info_file, "D3PERJ-COPPETEC\n\n", verbose)
        write_tee(info_file, "Este arquivo enumera e define as classes de segmentos de vídeo presentes no dataset:\n\n", verbose)
        write_tee(info_file, "Idle    - O depoente se encontra parado ou realizando outra ação irrelevante ao propósito do reconhecedor.\n", verbose)
        write_tee(info_file, "Speak   - O depoente se encontra falando.\n", verbose)
        write_tee(info_file, "Nod     - O depoente balança a cabeça como se para dizer que sim.\n", verbose)
        write_tee(info_file, "Shake   - O depoente balança a cabeça como se para dizer que não.\n", verbose)
        write_tee(info_file, "Examine - O depoente se aproxima da câmera para examinar alguma evidência que lhe é apresentada.\n", verbose)
        write_tee(info_file, "\n", verbose)
        write_tee(info_file, "As seguintes classes estão sob análise devido à falta de vídeos representativos da mesma nos dados diarizados até agora.\n", verbose)
        write_tee(info_file, "\n", verbose)
        write_tee(info_file, "KindOf - O depoente balança a cabeça como que para dizer que \"mais ou menos\"\n", verbose)
        write_tee(info_file, "\n", verbose)
        write_tee(info_file, "Total de Amostras por Classe (Atualizado em {}):\n".format(date.today()), verbose)
        for c in counts_sorted:
            if c not in ["Errors", "Validated"]:
                write_tee(info_file,"{}: {}\n".format(c, counts[c]), verbose)
        write_tee(info_file, "\n", verbose)
        write_tee(info_file, "Total de Amostras Inutilizáveis: {}\n".format(counts["Errors"]), verbose)
        write_tee(info_file, "Total de Amostras com Reconhecimento Facial: {}\n".format(counts["Validated"]), verbose)

def package_dataset(out_file, data_dir):
    unsorted_path = data_dir / "unsorted"
    classes_file = data_dir / "classes.txt"
    splits_file = data_dir / "splits.txt"
    info_file = data_dir / "dataset.info"
    changelog = data_dir / "changes.log"
    manifest = data_dir / "MANIFEST"

    files = [manifest, info_file, classes_file, splits_file, changelog]
    files.extend(list(unsorted_path.glob("**/*.*")))

    with open(manifest, 'w', encoding='utf-8') as f:
        for item in files:
            f.write("%s\n" % pathlib.Path(item).relative_to(data_dir))

    archive = None
    if out_file.suffix == '.zip':
        archive = zipfile.ZipFile(out_file, 'w', zipfile.ZIP_LZMA)
        for idx, name in enumerate(files):
            archive.write(pathlib.Path(name), arcname=pathlib.Path(name).relative_to(data_dir))
            print_progress(idx, len(files) - 1)
    elif '.tar' in out_file.suffix == '.tar':
        archive = tarfile.open(out_file, "w")
        for idx, name in enumerate(files):
            archive.add(pathlib.Path(name), arcname=pathlib.Path(name).relative_to(data_dir))
            print_progress(idx, len(files) - 1)
    elif out_file.suffixes == ['.tar', '.gz'] or out_file.suffix == '.tgz':
        archive = tarfile.open(out_file, "w:gz")
        for idx, name in enumerate(files):
            archive.add(pathlib.Path(name), arcname=pathlib.Path(name).relative_to(data_dir))
            print_progress(idx, len(files) - 1)
    else:
        cprint("Unsupported archive extension: {}".format(''.join(out_file.suffixes)), color='red', file=sys.stderr)
    
    archive.close()

def update_changelog(data_dir):
    try:
        message = input("Changelog Message: ").strip()
    except KeyboardInterrupt:
        cprint("\nCancelled by user.", 'red')
        exit()
    if not message:
        cprint("No message set.", 'red')
        return False
    else:
        with open(changelog, 'a', encoding='utf-8') as file:
            file.write("\n")
            file.write(datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M %z"))
            file.write("\n")
            file.write(message)
            file.write("\n")
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for dataset management.", add_help=False)

    subcommands = parser.add_subparsers(title="commands", dest="command", required=True, metavar="{command}", help="additional help" )

    update_parser = subcommands.add_parser('update', help='update dataset.info in dataset directory')
    update_parser.add_argument('-n', '--no-changelog', help="do not ask for a changelog message", action='store_true')
    update_parser.add_argument('-r', '--revalidate', help="revalidate files already validated", action='store_true')
    update_parser.add_argument("dataset_directory", help='the directory containing the dataset')

    validate_parser = subcommands.add_parser('validate', help='validate facial recognition')
    validate_parser.add_argument('-r', '--revalidate', help="revalidate files already validated", action='store_true')
    validate_parser.add_argument("dataset_directory", help='the directory containing the dataset')

    package_parser = subcommands.add_parser('package', help='package dataset into an archive', epilog='supported archive formats: .zip (default), .tar, .tar.gz')
    package_parser.add_argument('-u', '--update', help="update dataset information before packaging", action='store_true')
    package_parser.add_argument('-n', '--no-changelog', help="do not ask for a changelog message", action='store_true')
    package_parser.add_argument('-o', '--output', help="output file name (default same as dataset_directory)", metavar='PATH', nargs='?')
    package_parser.add_argument("dataset_directory", help='the directory containing the dataset')

    sort_parser = subcommands.add_parser('sort', help='sort dataset into directories according to splits')
    sort_parser.add_argument('-o', '--output', help="output file name (default same as dataset_directory)", metavar='PATH', nargs='?')
    sort_parser.add_argument("dataset_directory", help='the directory containing the dataset')

    clean_parser = subcommands.add_parser('clean', help='removes folders created by sort command')
    clean_parser.add_argument('-d', '--dir', help="directory to clean (default same as dataset_directory)", metavar='PATH', nargs='?', dest="output")
    clean_parser.add_argument('-o', '--output', help=argparse.SUPPRESS, metavar='PATH', nargs='?', dest="output")
    clean_parser.add_argument("dataset_directory", help='the directory containing the dataset')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-h", "--help", action="help", help="show this help message and exit")

    parser.add_argument("placeholder", help='the directory containing the dataset', nargs='?', metavar='<dataset_directory>')
    
    args = parser.parse_args()

    data_dir = pathlib.Path(args.dataset_directory).resolve()
    unsorted_path = data_dir / "unsorted"
    classes_file = data_dir / "classes.txt"
    splits_file = data_dir / "splits.txt"
    info_file = data_dir / "dataset.info"
    changelog = data_dir / "changes.log"

    out_path = data_dir

    if args.command == "sort":
        if args.output is not None:
            out_path = pathlib.Path(args.output)

        if data_dir.is_dir() and (classes_file).exists() and (splits_file).exists() and (unsorted_path).is_dir():
            print("Sorting dataset...")
            print("Classes: {} ({})".format(get_classes(classes_file), classes_file))
            print("Splits: {} ({})".format(get_splits(splits_file)["Names"], splits_file))

            sort_dataset(get_splits(splits_file), get_classes(classes_file), unsorted_path, out_path)
    
    elif args.command == "package":
        if args.output is not None:
            out_path = pathlib.Path(args.output)

        if not args.no_changelog and not update_changelog(data_dir):
            cprint("Packaging dataset without adding a changelog message.", 'red')

        if args.update:
            print("Updating dataset information...")
            update_info(data_dir, verbose=False)
        
        print("Packaging dataset...")
        out_file = None
        if len(out_path.suffixes) == 0:
            out_path.mkdir(parents=True, exist_ok=True)
            dataset_name="NULL"
            with open(info_file) as f:
                dataset_name = f.readline()
            out_file = out_path / ("{}-{}.zip".format(dataset_name, date.today().strftime("%Y-%m-%d")))
        else:
            out_file = pathlib.Path(sys.argv[3]).resolve()

        package_dataset(out_file, data_dir)

        print("\nDone")

    elif args.command == "update":
        print("Updating dataset information...")
        update_info(data_dir)
        if not args.no_changelog and not update_changelog(data_dir):
            cprint("Skipped adding a changelog message.", 'red')

    elif args.command == "validate":
        print("Validating files...")
        validate(data_dir, revalidate=args.revalidate)
    
    elif args.command == "clean":
        if args.output is not None:
            out_path = pathlib.Path(args.output)

        splits = get_splits(splits_file)
        print("This will delete all content from the following directories:")
        for split in splits['Names']:
            print("  {}".format(out_path / split))
        ans = input("Are you sure you wish to continue? (Y/n) ")
        if ans is "Y":
            for split in splits['Names']:
                shutil.rmtree(out_path / split, ignore_errors=True)
            print("Done")
        else:
            print("Operation cancelled by user.")