import pandas as pd
import numpy as np

def getCV(tls, data_df):
    sensitivity_mean = []
    sensitivity_variance = []
    sensitivity_std = []
    precision_mean = []
    precision_variance = []
    precision_std = []
    f1_score_mean = []
    f1_score_variance = []
    f1_score_std = []
    sensitivityCV = []
    precisionCV = []
    f1_scoreCV = []
    sampleID = []
    for sindex, srow in data_df.iterrows():

        sensitivity = []
        precision = []
        f1_score = []
        for tool in tls:
            sensitivity.append(srow['S' + tool])
            precision.append(srow['P' + tool])
            f1_score.append(srow['F' + tool])
        sensitivity_mean.append(np.mean(sensitivity))
        precision_mean.append(np.mean(precision))
        f1_score_mean.append(np.mean(f1_score))

        sensitivity_variance.append(np.var(sensitivity))
        precision_variance.append(np.var(precision))
        f1_score_variance.append(np.var(f1_score))

        sensitivity_std.append(np.std(sensitivity))
        precision_std.append(np.std(precision))
        f1_score_std.append(np.std(f1_score))
        ID = sindex + 1
        sampleID.append(ID)
        sensitivityCV.append(np.std(sensitivity) / np.mean(sensitivity))
        precisionCV.append(np.std(precision) / np.mean(precision))
        f1_scoreCV.append(np.std(f1_score) / np.mean(f1_score))

    data_dic = {'sampleID': sampleID,
                'sensitivity_mean': sensitivity_mean,
                'sensitivity_variance': sensitivity_variance,
                'sensitivity_std': sensitivity_std,
                'precision_mean': precision_mean,
                'precision_variance': precision_variance,
                'precision_std': precision_std,
                'f1_score_mean': f1_score_mean,
                'f1_score_variance': f1_score_variance,
                'f1_score_std': f1_score_std,
                'sensitivityCV': sensitivityCV,
                'precisionCV': precisionCV,
                'f1_scoreCV': f1_scoreCV,
                }
    data_df = pd.DataFrame(data_dic)
    data_df.to_csv('F:/CNVrecommendation/newCalling/NecessityExperimentResults/NecessityExperiment.csv', index=False)


if __name__ == '__main__':
    ExtractingMetaTargetScarler = 'F:/CNVrecommendation/newCalling/ExtractingMetaTargetScarler.csv'
    ExtractingMetaTargetScarler_df = pd.read_csv(ExtractingMetaTargetScarler, encoding='gbk', low_memory=False)
    tools = ['cnMops', 'facets', 'CNVpytor', 'CODEX', 'exomeCopy', 'cnvkit', 'contra']
    getCV(tools, ExtractingMetaTargetScarler_df)
