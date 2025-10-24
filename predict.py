
import test_indep_S552
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from collections import defaultdict
# 运行独立测试集的结果
if __name__ == "__main__":
    # 下例中 i取值为1-3, j取值为1-10
    #all_PDb = np.lib.format.open_memmap('./features_npy/S552/S552_181_pdbid.npy')

    # label_indices = defaultdict(list)
    # for index, label in enumerate(all_PDb):
    #     label_indices[label].append(index)
    # # 获取所有独特的类标
    # unique_labels = list(label_indices.keys())
    # for i in range(len(unique_labels)):
    #     test_indep_S552.evaluate_regression(f'new_checkpoint221_loo_deeplearn_b16_474DSSS120_{unique_labels[i]}.h5', i, 1)
    #     print(unique_labels[i])
    for i in range(1, 2):
        for j in range(1, 11):
            #C:/Users/admin/Desktop/MutResults
            test_indep_S552.evaluate_regression(f'C:/Users/admin/Desktop/MutResults/currentGOOD/model_10fold_181_SSDNA54_{i}_{j}.h5', i, j)
            #.evaluate_regression(f'new_checkpoint221_loo_deeplearn_b16_474DSSS120_4R56.h5', i, j)
                 #f'../save_model/Kfold/S1006/model8/model_10fold_161_31_20_regular.h5', i, j)



