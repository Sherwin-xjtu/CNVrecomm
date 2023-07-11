import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.notebook_repr_html',False)

def winDrawLoseAnalysis(RAfile, tag):
    # excel = pd.read_excel(io=r'F:/CNVrecommendation/newCalling/results/hypothesisTestingS.xlsx')
    excel = pd.read_excel(io=RAfile)
    compareRecom = ['RANDOM', 'cnMops', 'facets', 'CNVpytor', 'CODEX', 'exomeCopy', 'cnvkit', 'contra']
    win = {'RANDOM':0, 'cnMops':0, 'facets':0,'CNVpytor':0, 'CODEX':0, 'exomeCopy':0,'cnvkit':0, 'contra':0}
    draw = {'RANDOM':0, 'cnMops':0, 'facets':0,'CNVpytor':0, 'CODEX':0, 'exomeCopy':0,'cnvkit':0, 'contra':0}
    lose = {'RANDOM':0, 'cnMops':0, 'facets':0,'CNVpytor':0, 'CODEX':0, 'exomeCopy':0,'cnvkit':0, 'contra':0}
    for idx, row in excel.iterrows():
        for cm in compareRecom:
            if row['CNVrecom'] > row[cm]:
                win[cm] +=1
            elif row['CNVrecom'] == row[cm]:
                draw[cm] +=1
            else:
                lose[cm] +=1

    N = len(compareRecom)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35 # the width of the bars: can also be len(x) sequence
    win_li = []
    draw_li = []
    lose_li = []
    for tool in compareRecom:
        win_li.append(win[tool])
        draw_li.append(draw[tool])
        lose_li.append(lose[tool])

    d=[]
    for i in range(0,len(win_li)):
        sum = win_li[i] + draw_li[i]
        d.append(sum)

    # # 设置figsize的大小
    # plt.figure(figsize=(5, 5), dpi=80)
    p1 = plt.bar(ind, win_li, width, color='#2C847B')  # , yerr=menStd)
    p2 = plt.bar(ind, draw_li, width, color='#97B1AB', bottom=win_li)  # , yerr=womenStd)
    p3 = plt.bar(ind, lose_li, width, color='#AB5C94', bottom=d)
    plt.ylabel('Scores')
    plt.title('win draw lose')
    # # 设置x轴字体的大小
    plt.xticks(fontsize=8)
    plt.xticks(ind, ('RANDOM', 'cnMops', 'facets', 'CNVpytor', 'CODEX', 'exomeCopy', 'cnvkit', 'contra'))
    # plt.yticks(np.arange(0, 1000, 100)) simulated sequencing samples
    plt.yticks(np.arange(0, 300, 50))
    plt.legend((p1[0], p2[0], p3[0]), ('win', 'draw', 'lose'))
    plt.savefig('/Volumes/MyBook/2023work/CNVrecommendation/CNVrecomResults/predictResults/exomeDatawinDrawLose' + tag + '.tif', dpi=300)
    plt.show()
    results_dic = {'tools':compareRecom, 'win':win_li, 'draw':draw_li, 'lose':lose_li}
    df = pd.DataFrame(results_dic)
    df.to_csv('/Volumes/MyBook/2023work/CNVrecommendation/newCalling/results/exomeDatawinDrawLose' + tag + '.csv', index=False)

if __name__ == '__main__':

    ra_file = '/Volumes/MyBook/2023work/CNVrecommendation/CNVrecomResults/predictResults/exomeDatahypothesisTestingP.xlsx'
    tag_type = 'P'
    winDrawLoseAnalysis(ra_file, tag_type)