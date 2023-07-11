from random import randint
import pandas as pd
from numpy import mean
import numpy as np
np.set_printoptions(suppress=True)


def calculatingRA(t, data_df, tools):
    RA_recom_li = []
    RA_random_li = []
    RA_tools_li = []
    for tindex, row1 in data_df.iterrows():
        temp = []
        recomd_tool = row1['MMGR_pre']
        random_tool = randint(0, 6)
        for tool in tools:
            temp.append(row1[t + tool])

        e_best = max(temp)
        e_worst = min(temp)

        e_recomd = temp[recomd_tool]
        e_random = temp[random_tool]

        RA_recom = (e_recomd - e_worst)/(e_best - e_worst)
        RA_random = (e_random - e_worst)/(e_best - e_worst)
        RA_recom_li.append(RA_recom)
        RA_random_li.append(RA_random)
        RA_tools = []
        for e in temp:
            RA_tool = (e - e_worst)/(e_best - e_worst)
            RA_tools.append(RA_tool)
        RA_tools_li.append(RA_tools)
    return [RA_recom_li, RA_random_li, RA_tools_li]


def RAxls(ralist, tag):
    RA_recom_li, RA_random_li, RA_tools_li = ralist
    cnMops = []
    facets = []
    CNVpytor = []
    CODEX = []
    exomeCopy = []
    cnvkit = []
    contra = []
    for row in RA_tools_li:
        cnMops.append(row[0])
        facets.append(row[1])
        CNVpytor.append(row[2])
        CODEX.append(row[3])
        exomeCopy.append(row[4])
        cnvkit.append(row[5])
        contra.append(row[6])


    RA_dic = {'CNVrecom':RA_recom_li,
              'RANDOM':RA_random_li,
              'cnMops':cnMops,
              'facets': facets,
              'CNVpytor': CNVpytor,
              'CODEX': CODEX,
              'exomeCopy': exomeCopy,
              'cnvkit': cnvkit,
              'contra': contra
              }
    df = pd.DataFrame(RA_dic)
    df.to_excel('/Volumes/MyBook/2023work/CNVrecommendation/CNVrecomResults/predictResults/exomeDatahypothesisTesting' + tag + '.xlsx', index=False)

if __name__ == '__main__':
    ExtractingMetaTargetScarler = '/Volumes/MyBook/2023work/CNVrecommendation/CNVrecomResults/predictResults/exomeDataExtractingMetaTargetScarler_PSTpre.csv'
    ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
    tools = ['cnMops', 'facets', 'CNVpytor', 'CODEX', 'exomeCopy', 'cnvkit', 'contra']
    t = 'P'
    data_df = ExtractingMetaTargetScarler_df
    RA_recom_li, RA_random_li, RA_tools_li = calculatingRA(t, data_df, tools)
    print(mean(RA_recom_li))
    print(mean(RA_random_li))
    ide = 0

    for i in RA_tools_li:
        if ide == 0:
            ide +=1
            c = np.array(i)
        else:
            c = c + np.array(i)

    print(c/len(data_df))


    ralist = calculatingRA(t, data_df, tools)
    RAxls(ralist, t)



    # ExtractingMetaTargetScarler_df.to_csv('F:/CNVrecommendation/newCalling/results/ExtractingMetaTargetScarler_Fpre.csv', index=False)