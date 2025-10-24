
import test_indep_S552
import warnings
import numpy as np
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    for i in range(1, 2):
        for j in range(1, 11):
            test_indep_S552.evaluate_regression(f'C:/Users/admin/Desktop/MutResults/currentGOOD/model_10fold_181_SSDNA54_{i}_{j}.h5', i, j)




