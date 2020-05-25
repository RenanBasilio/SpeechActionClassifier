#!/usr/bin/python
import sys
import os
import argparse
import xml.etree.ElementTree as ET


def parse_xml(infile, outfile):
    tree = ET.parse(infile)
    root = tree.getroot()
    for child in root:
        outfile.write("SPEAKER {file} 1 {turn_onset} {turn_dur:.3f} <NA> <NA> speaker <NA> <NA>\n".format(
            file=os.path.basename(infile.name).replace('.segments.xml', ''),
            turn_onset=child.attrib['transcriber_start'],
            turn_dur=(float(child.attrib['transcriber_end']) - float(child.attrib['transcriber_start']))
        ))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert a NITE-formatted XML file into an RTTM transcription file.")
    argparser.add_argument('infile', type=argparse.FileType('r'))
    argparser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file", metavar="PATH")
    args = argparser.parse_args()
    
    parse_xml(args.infile, args.outfile)
