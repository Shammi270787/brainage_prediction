import numpy as np
import pickle
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt



# MOST LIKELY THIS IS WRONG
# def bias_variance_decomposition_classification(model_nm, y_true, p_dict):
#     uni_y = np.unique(y_true)
#     print(uni_y)
#
#     pred_means, pred_variance = [], []
#     for j in uni_y:
#         print('-----------j', j)
#
#         pred_rpts = []
#         pred_rpts_avg =[]
#         for rpt_no in range(1, 11):
#             key = 'repeat' + str(rpt_no)
#             pred_values = p_dict[model_nm][key]['Predicted_values']
#             true_values = p_dict[model_nm][key]['True_values']
#
#             pred_splits = []
#             for split_no in range(0, 10):
#                 index = np.where(true_values[split_no]==j)
#                 if np.size(index) != 0:
#                     pred = np.mean(pred_values[split_no][index])
#                     pred_splits.append(pred)
#
#             # print('pred_splits', pred_splits)
#             pred_rpts_avg.append(np.mean(pred_splits))
#             pred_rpts.append(pred_splits)
#
#         # print('pred_rpts', pred_rpts)
#         print('pred_rpts_avg',pred_rpts_avg, 'j', j)
#         pred_means.append(np.mean(pred_rpts_avg))
#         pred_variance.append(np.var(pred_rpts_avg))
#
#
#     print(len(pred_means), len(pred_variance))
#
#     bias_squared = np.mean((pred_means - uni_y) ** 2)
#     variance = np.mean(pred_variance)
#
#
#     print(bias_squared, variance)


np.set_printoptions(threshold=sys.maxsize)

def bias_variance_decomposition(model_nm, y_true, p_dict):
    """

    :param model_nm: model name
    :param y_true: true value of y
    :param p_dict: saved dictionary for predictions
    :return: expected loss, bias, variance
    """

    all_pred = np.zeros([10, len(y_true)])

    for rpt_no in range(1, 11):
        key = 'repeat' + str(rpt_no)
        pred_values = p_dict[model_nm][key]['Predicted_values']
        idx = p_dict[model_nm][key]['test_idx']

        for split_no in range(0, 10):
            idx_split = idx[split_no]
            preds_split = pred_values[split_no]

            # print('idx_split', idx_split, idx_split.shape)
            # print('preds_split', preds_split, preds_split.shape)

            all_pred[rpt_no - 1, idx_split] = preds_split

    # TAKEN FROM
    # Sebastian Raschka 2014-2020
    # mlxtend Machine Learning Library Extensions
    # Nonparametric Permutation Test
    # Author: Sebastian Raschka <sebastianraschka.com>

    main_predictions = np.mean(all_pred, axis=0)
    avg_expected_loss = np.apply_along_axis(lambda x: ((x - y_true) ** 2).mean(), axis=1, arr=all_pred).mean()
    avg_bias = np.sum((main_predictions - y_true) ** 2) / y_true.size
    avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size
    # print('avg_expected_loss', avg_expected_loss, 'avg_bias', avg_bias, 'avg_var', avg_var)

    # MY WAY of calculating
    # print('shapes', all_pred.shape, y_true.to_numpy().reshape(1,-1).shape)
    mse = np.mean(np.mean(((all_pred - y_true.to_numpy().reshape(1,-1))**2), axis=1))
    avg_bias2 = np.mean((main_predictions - y_true) ** 2)
    avg_var2 = np.mean((np.var(all_pred, axis=0)))
    mae = np.mean(np.mean((np.abs(all_pred - y_true.to_numpy().reshape(1, -1))), axis=1))
    # corr = np.corrcoef(all_pred, y_true.to_numpy().reshape(1, -1))[1, 0] # this is incorrect
    print('mse', mse, 'avg_bias', avg_bias2, 'avg_var', avg_var2, 'mae', mae)

    # colors = ["orange", "green", "blue", "red", "pink", "purple"]
    # slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_true)
    # line = slope * y_true + intercept
    # plt.scatter(y_true, main_predictions, color=colors[1])
    # plt.plot(y_true, line, linewidth=1, color='black', alpha=0.3)
    # plt.xlabel('True age (in years)', fontsize=14)
    # plt.ylabel('Predicted age (in years)', fontsize=14)
    # plt.savefig('RVM_smote' + '.png', dpi=500)

    return avg_expected_loss, avg_bias, avg_var


# if __name__ == '__main__':
#     datafile = '/data/project/brainage/data/BSF_423/ixi_camcan_features'
#     predictfile_cv = '/data/project/brainage/CoRR_test_retest/results2/ixi_camcan_rvm_smote_trial2.predict'
#
#
#     p_dict = pickle.load(open(predictfile_cv, "rb"))
#     df2 = pickle.load(open(datafile, "rb"))
#     df = pd.DataFrame(df2)
#     age = df['age']
#     age = age.astype(float).round()
#     age = age.astype(int)
#     y_true = age
#
#     avg_expected_loss, avg_bias, avg_var = bias_variance_decomposition('RVM', y_true, p_dict)
