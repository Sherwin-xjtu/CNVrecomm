#!/usr/bin/env python3
import subprocess
import pysam
import rpy2.robjects as robjects
import rpy2
from rpy2.robjects.packages import importr
import sys
import argparse
sys.path.append('../')



def getEstimated(arguments, workdir):
    # python PEcnv.py run tumor.bam --normal normal.bam  --targets CNVTarget.bed --fasta reference.fa --annotate refFlat.txt --preprocess dukeExcludeRegions.bed --output-refBaseline refBaseline.tsv --output-dir our_dir

     # input
    # cbam_file = "NA10851.test.bam"
    # nbam_file = "NA10851.test.bam"
    # targets = "CNVTarget.bed"
    # reference = "reference.fa"
    # refFlat = "refFlat.txt"
    # dukeExcludeRegions = "dukeExcludeRegions.bed"
    # output_refBaseline = "refBaseline.tsv"
    # our_dir = "our_dir"
    # purity = 0
    cbam_file = arguments.cbam
    nbam_file = arguments.nbam
    targets = arguments.target
    reference = workdir + '/data/hs37d5.fa'
    refFlat = workdir + '/data/refFlat.txt'
    dukeExcludeRegions = workdir + '/data/dukeExcludeRegions.bed'
    output_refBaseline = workdir + '/bam2tsv/pecnvout/refBaseline.tsv'
    out_dir = workdir + '/bam2tsv/pecnvout/'
    PEcnv =  workdir +'/' + 'bam2tsv/PEcnv/PEcnv.py'
    

    subprocess.run(['python', PEcnv, 'run', cbam_file, '--normal', nbam_file, '--targets', targets,
                '--fasta', reference, '--annotate', refFlat, '--preprocess', dukeExcludeRegions,
                '--output-refBaseline', output_refBaseline, '--output-dir', out_dir])
    return out_dir


def getLenDep(arguments):
    cbam_file = arguments.cbam
    samtools_cmd = ["samtools", "depth", cbam_file]
    try:
        output = subprocess.check_output(samtools_cmd, universal_newlines=True)

        total_depth = 0
        num_positions = 0
        for line in output.splitlines():
            values = line.strip().split("\t")
            depth = values[2]
            if depth != 0:
                total_depth += int(depth)
                num_positions += 1

        average_depth = total_depth / num_positions
        # print("average_depth:", average_depth)

    except subprocess.CalledProcessError as e:
        print("Error:", e)

    bam_reader = pysam.AlignmentFile(cbam_file, "rb")

    read_length = 0
    for read in bam_reader:
        read_length = read.query_length
        print(read_length)
        break
    bam_reader.close()
    # print("Read length:", read_length)

    return read_length, average_depth


def main(arguments):

    
    getEstimated(arguments)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--cbam', required=True, help='Input the case sample')
    parser.add_argument('--nbam', required=True, help='Input the control sample')
    parser.add_argument('--target', required=True, help='Input the target file')
    parser.add_argument('--reference', required=True, help='Input the reference file')
    parser.add_argument('--refflat', required=True, help='Input the refflat file')
    parser.add_argument('--dukeExcludeRegions', required=True, help='Input the dukeExcludeRegions file')
    parser.add_argument('--purity', type=float, required=True, help='Input the purity of tumor')
    parser.add_argument('--refBaseline', required=True, help='Output the refBaseline file')
    parser.add_argument('--ourDir', required=True, help='Output the output dir')
    args = parser.parse_args()
    
    readLen, averageDep = getLenDep(args)