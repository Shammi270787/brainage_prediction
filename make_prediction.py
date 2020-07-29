#!/home/smore/.venvs/py3smore/bin/python3
import argparse
import pickle
import datetime
import time
import sys
#from group_pred import *
from additional_funs import *
from train_predict import *
from sklearn.model_selection import train_test_split
from site_pred import site_fun_pred
import copy
from scipy import stats
from bias_variance import bias_variance_decomposition
from skrvm import RVR
from sklearn.model_selection import train_test_split


def performance_measures(y_pred, y_test):
    MAE = np.mean(np.abs(y_pred - y_test))
    CORR = np.corrcoef(y_pred, y_test)[1, 0]
    return MAE, CORR


if __name__ == '__main__':

    datafile1 = '/data/project/brainage/data/BSF_173/camcan_bsf_173_features'
    datafile2 = '/data/project/brainage/data/BSF_173/ixi_bsf_173_features'
    modelfile_cv = '/data/project/brainage/CoRR_test_retest/results2/ixi_100_camcan_70_30_smote_bsf173.model'
    predictfile_cv = '/data/project/brainage/CoRR_test_retest/results2/ixi_100_camcan_70_30_smote_bsf173.predict'
    modelfile = '/data/project/brainage/CoRR_test_retest/results2/ixi_100_camcan_70_30_smote_bsf173.model_comb'

    df1 = pickle.load(open(datafile1, "rb"))
    df2 = pickle.load(open(datafile2, "rb"))
    model_file = pickle.load(open(modelfile, "rb"))
    m_dict = pickle.load(open(modelfile_cv, "rb"))
    p_dict = pickle.load(open(predictfile_cv, "rb"))

    df1 = pd.DataFrame(df1)
    age = df1['age']
    df = pd.DataFrame({'age': age})
    # create 3 bins of ages of equal sizes
    qc = pd.cut(df.age.tolist(), bins=3, precision=1)
    print('age_bins', qc.categories)
    labels = qc.codes
    df1['bin'] = qc.codes

    train_df1, test_df = train_test_split(df1, test_size=0.3, stratify=df1['bin'], random_state=200)


    df2 = pd.DataFrame(df2)
    age2 = df2['age']
    df = pd.DataFrame({'age': age2})
    # create 3 bins of ages of equal sizes
    qc2 = pd.cut(df2.age.tolist(), bins=3, precision=1)
    print('age_bins', qc2.categories)
    df2['bin'] = qc2.codes

    train_data = pd.concat([train_df1, df2], axis=0)
#     train_data = train_df1

    Y_train = train_data['age']
    Y_train = Y_train.astype(float).round()
    Y_train = Y_train.astype(int)
    X_train = train_data.drop(['age', 'gender', 'site', 'subject','bin'], axis=1).values
    print('CAMCAN TRAIN:  ', 'X and Y shapes: ', X_train.shape, X_train.shape)
    print('max-min age', np.max(Y_train), np.min(Y_train))

    Y_test1 = test_df['age']
    Y_test1 = Y_test1.astype(float).round()
    Y_test1 = Y_test1.astype(int)
    X_test1 = test_df.drop(['age', 'gender', 'site', 'subject', 'bin'], axis=1).values
    print('CAMCAN TEST:  ', 'X and Y shapes: ', X_test1.shape, Y_test1.shape)
    print('max-min age', np.max(Y_test1), np.min(Y_test1))


# RESULTS: Cross-validation results , 10 repeats, 10 CV
    print('---CV RESULTS---')
    for model_nm, model_item in m_dict.items():
        MAE_all, CORR_all, MSE_all, best_param, best_score = list(),list(),list(), list(),list()

        for i in range(1,11):
            key = 'repeat'+ str(i)
            best_param.append(m_dict[model_nm][key]['best_param'])
            best_score.append(m_dict[model_nm][key]['best_score'])
            MAE_all.append(np.mean(p_dict[model_nm][key]['MAE']))
            CORR_all.append(np.mean(p_dict[model_nm][key]['CORR']))
            MSE_all.append(np.mean(p_dict[model_nm][key]['MSE']))
        print('MODEL NAME:', model_nm, '\n MAE :', np.mean(MAE_all), '\n CORR: ',  np.mean(CORR_all), '\n MSE: ',
              np.mean(MSE_all), '\n best_param: ' , best_param, '\n best_score:' , best_score)
        
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomposition(model_nm, Y_train, p_dict)
        print('avg_expected_loss:', avg_expected_loss, '\n avg_bias:', avg_bias, '\n avg_var:', avg_var,'\n')


# RESULTS: predict on test data
    print('\n---TEST RESULTS---')
    ses1_all = list()
    ses2_all = list()
    pred_values_all, mae_all, corr_all, mod_name_all = list(),list(),list(),list()

    for key, model_item in model_file.items():
        modname = key
        print('model:', modname)

        if modname != 'COMB':
            model = model_file[modname]['model']
            scaler = model_file[modname]['pca']['scaler']
            pca = model_file[modname]['pca']['pca']
            num_comps = model_file[modname]['pca']['pca_comp_used']
            print('total pca comp: ', model_file[modname]['pca']['pca_comp_gen'])
            print('pca comps default: ', num_comps)
            print('scaler', scaler)

            print('\ntrain_pred:', '\nMAE: ', model_file[modname]['train_predictions']['MAE'],
                  '\nCORR', model_file[modname]['train_predictions']['CORR'],
                  '\nMSE:',model_file[modname]['train_predictions']['MSE'])

            print('\ntest_pred_saved: ', '\nMAE:', model_file[modname]['test_predictions']['MAE'],
                  '\nCORR:', model_file[modname]['test_predictions']['CORR'],
                  '\nMSE:', model_file[modname]['test_predictions']['MSE'])

            X_test_pca1 = X_test1[:,0:423]
            pred_var1 = predict(modname, model, X_test_pca1, Y_test1)
            ses1_pred = pred_var1['Predicted_values']
            print('\nTEST_pred: ', '\nMAE:', pred_var1['MAE'], '\nCORR:', pred_var1['CORR'], '\nMSE:',pred_var1['MSE'])

        elif modname == 'COMB':
            model_list = model_file[modname]['model']
            scaler = model_file[modname]['pca']['scaler']
            pca = model_file[modname]['pca']['pca']
            num_comps = model_file[modname]['pca']['pca_comp_used']
            print('total pca comp: ', model_file[modname]['pca']['pca_comp_gen'])
            print('pca comps used: ', num_comps)

            broad_lad, group_lad, model_2 = model_list[0], model_list[1], model_list[2]

            print('\n train_pred: ', model_file[modname]['train_predictions']['MAE'],
                  model_file[modname]['train_predictions']['CORR'], model_file[modname]['train_predictions']['MSE'])
            print('ses1_pred_saved: ', model_file[modname]['test_predictions']['MAE'],
                  model_file[modname]['test_predictions']['CORR'], model_file[modname]['test_predictions']['MSE'])


            X_test_pca1 = X_test1
            pred_all_test = make_predictions(X_test_pca1, broad_lad, group_lad)
            pred_var1 = predict(modname, model_2, pred_all_test, Y_test1)
            ses1_pred = pred_var1['Predicted_values']
            print('\nsTEST_pred: ', ses1_pred, pred_var1['MAE'], pred_var1['CORR'], pred_var1['MSE'])


























