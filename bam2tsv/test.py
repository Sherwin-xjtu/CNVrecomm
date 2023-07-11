import subprocess
import pysam
import rpy2.robjects as robjects
import rpy2
from rpy2.robjects.packages import importr
import sys
sys.path.append('../')

# input
cbam_file = "NA10851.test.bam"
nbam_file = "NA10851.test.bam"
targets = "CNVTarget.bed"
reference = "reference.fa"
refFlat = "refFlat.txt"
dukeExcludeRegions = "dukeExcludeRegions.bed"
output_refBaseline = "refBaseline.tsv"
our_dir = "our_dir"
purity = 0




samtools_cmd = ["samtools", "depth", cbam_file]

# python PEcnv.py run tumor.bam --normal normal.bam  --targets CNVTarget.bed --fasta reference.fa --annotate refFlat.txt --preprocess dukeExcludeRegions.bed --output-refBaseline refBaseline.tsv --output-dir our_dir

subprocess.run(['python', 'PEcnv/PEcnv.py', 'run ' + cbam_file + ' --normal ' + nbam_file  + '--targets ' + targets + \
                '--fasta ' + reference + '--annotate ' + refFlat + '--preprocess ' + dukeExcludeRegions +  \
                    '--output-refBaseline ' + output_refBaseline + '--output-dir ' + our_dir])

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

    print("average_depth:", average_depth)

except subprocess.CalledProcessError as e:
    print("Error:", e)



bam_reader = pysam.AlignmentFile(cbam_file, "rb")

read_length = 0
for read in bam_reader:
    read_length = read.query_length
    print(read_length)
    break

bam_reader.close()
print("Read length:", read_length)

stats = importr('stats')
ggplot2 = importr('ggplot2')
base = importr('base')
# 创建R向量并进行操作
x = robjects.IntVector([1, 2, 3, 4, 5])
mean_x = base.mean(x)
print(f"Mean of x: {mean_x[0]}")

# 执行R函数
r_code = """
    my_function <- function(x) {
        x_squared <- x^2
        return(x_squared)
    }
"""
robjects.r(r_code)
my_function = robjects.globalenv['my_function']
result = my_function(5)
print(f"Result of my_function: {result[0]}")
