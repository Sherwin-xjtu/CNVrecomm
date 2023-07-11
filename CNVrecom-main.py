# from CNVrecomSST.dglHeteroModel import CNVrecommMainSST
from CNVrecomST.dglHeteroModel import CNVrecommMainST
# from CNVrecomSTT.dglHeteroModel import CNVrecommMainSTT
from dataProcessing import nodes
import pandas as pd
import os
import argparse
from bam2tsv import b2t
from collections import Counter


def CNVrecommResults(r1, r2, r3, mt):
    result = []
    file_path = '/Volumes/MyBook/2023work/CNVrecommendation/CNVrecomResults/predictResults/ExtractingMetaTargetScarler_{}pre.csv'.format(mt)
    if len(r1) == len(r2) == len(r3):
        for i in range(len(r1)):
            temp = [r1['MMGR_pre'][i], r2['MMGR_pre'][i], r3['MMGR_pre'][i]]
            result.append(max(temp,key=temp.count))
    else:
        print('Some processes may be error!')
    r1['MMGR_pre'] = result

    r1.to_csv(file_path, index=False)


def CNVrecommMainT(mtl, pat):

    CNVrecommST_result = CNVrecommMainST(mtl[2], pat)
    # CNVrecommSST_result = CNVrecommMainSST(mtl[1], pat)
    # CNVrecommSTT_result = CNVrecommMainSTT(mtl[2], pat)
    # CNVrecommResults(CNVrecommST_result, CNVrecommSST_result, CNVrecommSTT_result, mtl[0][0])

def CNVrecommMain(workdir, pat, sampleid):
    tag = pat
    # pat = 'predict'
    results = CNVrecommMainST(workdir, pat, sampleid)
    tools = ['cnMops', 'facets', 'CNVpytor', 'CODEX', 'exomeCopy', 'cnvkit', 'contra']
    CNVrecomm_list = []
    for key, value in results.items():
        counter = Counter(value[0])
        most_common = counter.most_common(1)
        most_common_element = most_common[0][0]
        CNVrecomm_list.append(tools[most_common_element])
    CNVrecomm_dict = {}
    CNVrecomm_dict['meta-target-Sensitivity'] = [CNVrecomm_list[0]]
    CNVrecomm_dict['meta-target-Precision'] = [CNVrecomm_list[0]]
    CNVrecomm_dict['meta-target-F1_score'] = [CNVrecomm_list[0]]
    CNVrecomm_results = pd.DataFrame(CNVrecomm_dict)
    return CNVrecomm_results


def main(arguments):

    purity = arguments.purity
    readLen = arguments.readLength
    readDepth = arguments.sampleDepth
    file_path = arguments.cbam
    file_name = os.path.basename(file_path)
    # file_name_without_extension = os.path.splitext(file_name)[0]
    # print(file_name)
    # print(file_name_without_extension)
    # exit()
    sample = file_name
    # sample_arrdf = pd.DataFrame({'purity':[0.6], 'shortCNV':[0.3], 'middleCNV':[0.3], 'largeCNV':[0.1], 'deletion':[0.3], 'readLen':[101], 'readDepth':[100], 'Stype':['Stype'], 'Ftype':['Ftype'], 'Ptype':['Ptype'], 'sample':['sampleid']})
    current_path = os.path.dirname(os.path.abspath(__file__))
    # sampleid = sample_arrdf['sample'][0]
    pecnvout_dir = current_path + '/bam2tsv/pecnvout/'
    if not os.path.exists(pecnvout_dir):
        pecnvout_dir = b2t.getEstimated(arguments, current_path)

    for filename in os.listdir(pecnvout_dir):
        if filename.endswith('.segm.tsv'):
            segm_file = pecnvout_dir + '/' + filename
            break
    segm_df = pd.read_csv(segm_file, encoding='gbk', low_memory=False, sep='\t')
    shortCNV = 0
    middleCNV = 0
    largeCNV = 0
    deletion = 0
    for idx, row in segm_df.iterrows():
        if row['log2'] > 0.01:
            if row['end'] - row['start'] < 100000:
                shortCNV +=1
            elif row['end'] - row['start'] >= 100000 and row['end'] - row['start'] < 1000000:
                middleCNV +=1
            else:
               largeCNV +=1
        elif row['log2'] < -0.01:
            deletion +=1
    allCNV = shortCNV + middleCNV + largeCNV + deletion
    if allCNV == 0:
        shortCNVP = 0
        middleCNVP = 0
        largeCNVP = 0
        deletionP = 0
    else:
        shortCNVP = shortCNV/allCNV
        middleCNVP = middleCNV/allCNV
        largeCNVP = largeCNV/allCNV
        deletionP = deletion/allCNV

    sample_arrdf = pd.DataFrame({'purity':[purity], 'shortCNV':[shortCNVP], 'middleCNV':[middleCNVP], 'largeCNV':[largeCNVP], 'deletion':[deletionP], 'readLen': [readLen], 'readDepth': [readDepth],'sample': [sample]})
    nodes.mian(sample_arrdf, current_path)
    pat = 'predict'
    CNVrecomm_results = CNVrecommMain(current_path, pat, sample)
    CNVrecomm_results.to_csv(arguments.output, index=False, sep='\t')


if __name__ == '__main__':
    meta_tag_listS = ['SST', 'SSST', 'SSTT']
    meta_tag_listP = ['PST', 'PSST', 'PSTT']
    meta_tag_listF = ['FST', 'FSST', 'FSTT']
    # CNVrecomm_pattern = 'train'
    meta_tag_lists = [meta_tag_listS, meta_tag_listP, meta_tag_listF]
    # for meta_tag_list in meta_tag_lists:
    #     CNVrecommMain(meta_tag_list, CNVrecomm_pattern)
    # CNVrecommMain(meta_tag_listF, CNVrecomm_pattern)
    meta_tag_listST = ['SST', 'PST', 'FST']
    # CNVrecomm_pattern = 'test'

    # CNVrecomm_pattern = 'predict'
    # CNVrecommMainT(meta_tag_listST, CNVrecomm_pattern)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cbam', required=True, help='Input the case sample')
    parser.add_argument('-n', '--nbam', required=True, help='Input the control sample')
    parser.add_argument('-t', '--target', required=True, help='Input the target file')
    # parser.add_argument('--reference', required=True, help='Input the reference file')
    # parser.add_argument('--refflat', required=True, help='Input the refflat file')
    # parser.add_argument('--dukeExcludeRegions', required=True, help='Input the dukeExcludeRegions file')
    parser.add_argument('-p', '--purity', type=float, required=True, help='Input the purity of tumor')
    parser.add_argument('-l', '--readLength', type=int, required=True, help='Input the read length')
    parser.add_argument('-d', '--sampleDepth', type=float, required=True, help='Input the sample average depth')
    parser.add_argument('-o', '--output', required=True, help='The tsv file of the CNVrecom results')
    # parser.add_argument('-or', '--refBaseline', required=True, help='Output the refBaseline file')
    # parser.add_argument('-od', '--ourDir', required=True, help='Output the output dir')
    args = parser.parse_args()
    main(args)