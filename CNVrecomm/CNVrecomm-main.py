from CNVrecomm.CNVrecommSST.dglHeteroModel import CNVrecommMainSST
from CNVrecomm.CNVrecommST.dglHeteroModel import CNVrecommMainST
from CNVrecomm.CNVrecommSTT.dglHeteroModel import CNVrecommMainSTT


def CNVrecommResults(r1, r2, r3, mt):
    result = []
    file_path = 'F:/CNVrecommendation/newCalling/results/ExtractingMetaTargetScarler_{}pre.csv'.format(mt)
    if len(r1) == len(r2) == len(r3):
        for i in range(len(r1)):
            temp = [r1['MMGR_pre'][i], r2['MMGR_pre'][i], r3['MMGR_pre'][i]]
            result.append(max(temp,key=temp.count))
    else:
        print('Some processes may be error!')
    r1['MMGR_pre'] = result

    r1.to_csv(file_path, index=False)


def CNVrecommMain(mtl, pat):

    CNVrecommST_result = CNVrecommMainST(mtl[0], pat)
    CNVrecommSST_result = CNVrecommMainSST(mtl[1], pat)
    CNVrecommSTT_result = CNVrecommMainSTT(mtl[2], pat)
    CNVrecommResults(CNVrecommST_result, CNVrecommSST_result, CNVrecommSTT_result, mtl[0][0])


if __name__ == '__main__':
    meta_tag_list = ['SST', 'SSST', 'SSTT']
    # meta_tag_list = ['PST', 'PSST', 'PSTT']
    # meta_tag_list = ['FST', 'FSST', 'FSTT']
    CNVrecomm_pattern = 'train'
    CNVrecommMain(meta_tag_list, CNVrecomm_pattern)