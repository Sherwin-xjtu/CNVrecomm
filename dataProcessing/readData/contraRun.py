# coding=utf-8
import os
import math
import os.path
import shutil


def writingSh(path1, path2, tool):
    f = open(path2 + tool + '/run.sh', 'w')
    for fe in os.listdir(path1):
        npath = path1 + ""
        if "NA10851" not in fe and fe.endswith('.bai'):
            tbuf = fe[:-4]
            if tbuf in os.listdir(path1):
                if tool == 'contra':
                    out = '/Volumes/MyBook/Project/CNVrecommendation/cnvCalling/contra/' + tbuf
                    f1 = open(path2 + tool + '/' + tbuf + '.sh', 'w')
                    tpath = path1 + tbuf
                    npath = path1 + 'NA10851.mapped.ILLUMINA.bwa.CEU.exome.20130415.bam'
                    f1.write(
                        '/Users/sherwinwang/opt/anaconda3/envs/py2/bin/python  '
                        '/Volumes/MyBook/Project/CNVrecommendation/CONTRA.v2.0.8/contra.py '
                        '-t /Volumes/MyBook/Project/CNVrecommendation/ref/new_ncbi_anno_rel104_db_hg19_TXS.bed  '
                        '-f /Volumes/MyBook/Project/CNVrecommendation/ref/tgp_phase2_flat/hs37d5.fa '
                        '--sampleName contra ')
                    f1.write('-s ' + tpath + ' ')
                    f1.write('-c ' + npath + ' ')
                    f1.write('-o ' + out + ' ')
                    f1.write('--minExon 100')
                    testedFile = 'contra.CNATable.10rd.10bases.20bins.txt'
                    if not os.path.exists(out):
                        f.write('sh ' + path2 + tool + '/' + tbuf + '.sh' + '\n')
                    elif not os.path.exists(out + '/table'):
                        shutil.rmtree(out)
                        f.write('sh ' + path2 + tool + '/' + tbuf + '.sh' + '\n')
                    elif testedFile not in os.listdir(out + '/table'):
                        shutil.rmtree(out)
                        f.write('sh ' + path2 + tool + '/' + tbuf + '.sh' + '\n')
                    elif os.path.getsize(
                            out + '/table/' + 'contra.CNATable.10rd.10bases.20bins.DetailsFILTERED.txt') == 284:

                        shutil.rmtree(out)
                        f.write('sh ' + path2 + tool + '/' + tbuf + '.sh' + '\n')
                    f.close()

if __name__ == "__main__":
    tumor = '/Volumes/MyBook/DATA/CNV/realData/'
    run = '/Volumes/MyBook/Project/CNVrecommendation/run/'
    toolname = 'contra'
    writingSh(tumor, run, toolname)




