import os


missing_files = r'/Volumes/MyBook/Project/CNVrecommendation/realData/missingFiles'
f1 = open(missing_files,'w')
f = open("/Volumes/MyBook/Project/CNVrecommendation/cnvkitCalling/cnMopsSampname", "r")
path = r'/Volumes/MyBook/DATA/CNV/realData'
bams = []
over_bams = []
for line in f.readlines():
    line = line.strip('\n')
    line = line.strip('\r')
    bam = line + '.bam'
    bams.append(bam)

    for tf in os.listdir(path):
        if bam == tf:
            over_bams.append(bam)
missings = list(set(bams) - set(over_bams))
for miss in missings:
    f1.write(miss + '\n')
f1.close()
