#!/home/smore/.venvs/py3smore/bin/python3
import sys
import datetime
import time
import random
import pandas as pd
from mord import LAD
import numpy as np
import os
from multiprocessing import Pool
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from train_predict import *
import itertools
import matplotlib.pyplot as plt
from perform_PCA import perform_pca
from operator import itemgetter
from smote_int import smote_data



def random_sampling(data):  # give dataframe as input
    df = data
    # df['age'] = df.age.astype(float).round()
    # df['age'] = df.age.astype(int)

    final_sample = pd.DataFrame()
    # remove samples from a site which have less than 100 samples
    site_rem = list()
    for sites in np.sort(df.site.unique()):
        df_site = df[df.site == sites]
        if len(df_site) < 100:
            site_rem.append(sites)
            print('sites %d samples %d' % (sites, len(df_site)))

    [df.drop(df[df.site == item].index, inplace=True) for item in site_rem]

    # create bins based on age
    qc = pd.cut(df.age.tolist(), bins=15, precision=1)
    np.set_printoptions(threshold=sys.maxsize)
    print('categories', qc.categories)
    df['bin'] = qc.codes

    df_sites_bins = pd.DataFrame()
    for sites, bins in itertools.product(np.sort(df.site.unique()), np.sort(df.bin.unique())):
        data = {
            'site': sites,
            'bin': bins,
            'n_subjects': len(df[(df.site == sites) & (df.bin == bins)])
        }

        df_sites_bins = df_sites_bins.append(data, ignore_index=True)
    np.set_printoptions(threshold=sys.maxsize)
    print('df_sites_bins', df_sites_bins)

    final_sample = pd.DataFrame()
    for sites, bins in itertools.product(np.sort(df.site.unique()), np.sort(df.bin.unique())):
        tmp = df[(df.site == sites) & (df.bin == bins)]
        if len(tmp) == 0:
            continue
        p = 10 if len(tmp) >= 10 else len(tmp)
        tmp = tmp.sample(n=p)
        final_sample = final_sample.append(tmp)

    return final_sample


def initialize(input_list):
    model_dict = dict()
    for item in input_list:
        model_dict['split_' + str(item[2])] = {'broad_age': {}, 'specific_age': {}} #item[3]=split num

    pred_train_res_dict = dict()
    pred_train_dict = dict()

    for item in input_list:
        pred_train_res_dict['split_' + str(item[2])] = {}
        pred_train_dict['split_' + str(item[2])] = {}

    return model_dict, pred_train_res_dict, pred_train_dict


def train_test(att):
    """"Train and predict function"""
    mod_nm, x_train, y_train, x_test, y_test, item_rpt, split_no, seed_no, train_idx, test_idx = att
    # print('train set', x_train.shape, y_train.shape)
    # print('test set', x_test.shape, y_test.shape)

    # con_X_train = confound_features[train_id]
    # con_X_test = confound_features[test_id]
    # con_X_train = con_X_train.astype('float')
    # con_X_test = con_X_test.astype('float')
    # print('train set', x_train.shape, con_X_train.shape, y_train.shape)
    # print('test set', x_test.shape, con_X_test.shape, y_test.shape)
    # confound_models = confound_train_model(x_train, con_X_train)
    # X_train_features = residual_features_fun(con_X_train, confound_models, x_train)
    # X_test_features = residual_features_fun(con_X_test, confound_models, x_test)
    # confound_dict = {'X_train_features': X_train_features, 'X_test_features': X_test_features,
    #                  'counfound_models': confound_models, 'Y_train': Y_train, 'Y_test': Y_test}
    # model, best_score, best_param = train(mod_nm, X_train_features, y_train)
    # pred_var = predict(mod_nm, model, X_test_features, y_test)

    # print("running on model: {}, repeat: {}, split: {}, process id: {}".format(mod_nm, str(item_rpt), split_no,
    #                                                                        os.getpid()))

    print('IN training', x_train.shape, y_train.shape)
    model, best_score, best_param, pred_var, train_var = train(mod_nm, x_train, y_train, x_test, y_test)
    out_list = [mod_nm, model, pred_var, item_rpt, split_no, best_score, best_param, train_var, train_idx, test_idx]
    return out_list


def create_input_list(x, y, num_repeats, seed_num, model_name):
    input_list = list()
    if seed_num is None:
        master_seed = seed_num.randint(0, 99999)
    else:
        master_seed = seed_num
    random.seed(master_seed)

    seed_no = [random.randint(0, 99999) for p in range(0, num_repeats)]  # create as many seeds as num_repeats
    for mod_nm in model_name:
        for item_repeat in range(0, num_repeats, 1):
            info_list = [x, y, str(item_repeat + 1), seed_no[item_repeat], mod_nm]
            input_list.append(info_list)
    return input_list


def kfolds_fun_pred(x, y, y_labels, num_repeats, num_kfolds, master_seed, modelnames, num_process, pca_status):
    input_list = list()   # list of info about input data
    input_list2 = list()

    if master_seed is None:
        master_seed = random.randint(0, 99999)

    random.seed(master_seed)
    seed_num = [random.randint(0, 99999) for p in range(0, num_repeats)]  # create as many seeds as num_repeats

    for item_repeat in range(0, num_repeats, 1):
        kfold = StratifiedKFold(n_splits=num_kfolds, random_state=seed_num[item_repeat], shuffle=True)  # prepare cross validation
        split_num = 0
        for train_idx, test_idx in kfold.split(x, y_labels):
            print('FOR SPLIT NO %d' % split_num)
            print('0:', np.count_nonzero(y_labels[train_idx] == 0), np.count_nonzero(y_labels[test_idx] == 0))
            print('1:', np.count_nonzero(y_labels[train_idx] == 1), np.count_nonzero(y_labels[test_idx] == 1))
            print('2:', np.count_nonzero(y_labels[train_idx] == 2), np.count_nonzero(y_labels[test_idx] == 2))
            split_num = split_num + 1

            X_train_features = x[train_idx]
            X_test_features = x[test_idx]
            print(X_train_features.shape, X_test_features.shape)

            if pca_status == 1:
                X_train_pca, X_test_pac, scaler, pca, num_comp = perform_pca(X_train_features, X_test_features)
                print('X_train_pca, X_test_pac', X_train_pca.shape, X_test_pac.shape)
            else:
                X_train_pca, X_test_pac, scaler, pca, num_comp = X_train_features, X_test_features, 0, 0, 0
            # for mod_name in modelnames:
            #     info_list = [mod_name, X_train_pca, y[train_idx], X_test_pac, y[test_idx],
            #                  'repeat'+str(item_repeat+1), split_num, seed_num[item_repeat]]

            info_list = [X_train_pca, y[train_idx], X_test_pac, y[test_idx], 'repeat'+str(item_repeat+1), split_num,
                          seed_num[item_repeat]]
            info_list2 = [train_idx, test_idx, 'repeat' + str(item_repeat + 1), split_num, seed_num[item_repeat]]


            input_list.append(info_list)
            input_list2.append(info_list2)

    # input_list = sorted(input_list, key=itemgetter(0))
    return input_list, input_list2


def kfolds_fun_pred_smote(x, y, y_labels, num_repeats, num_kfolds, master_seed, modelnames, num_process, pca_status, site, smote_status):
    input_list = list()   # list of info about input data
    input_list2 = list()

    if master_seed is None:
        master_seed = random.randint(0, 99999)

    random.seed(master_seed)
    seed_num = [random.randint(0, 99999) for p in range(0, num_repeats)]  # create as many seeds as num_repeats

    for item_repeat in range(0, num_repeats, 1):
        kfold = StratifiedKFold(n_splits=num_kfolds, random_state=seed_num[item_repeat], shuffle=True)  # prepare cross validation
        split_num = 0
        for train_idx, test_idx in kfold.split(x, y_labels):
            print('FOR SPLIT NO %d' % split_num)
            print('0:', np.count_nonzero(y_labels[train_idx] == 0), np.count_nonzero(y_labels[test_idx] == 0))
            print('1:', np.count_nonzero(y_labels[train_idx] == 1), np.count_nonzero(y_labels[test_idx] == 1))
            print('2:', np.count_nonzero(y_labels[train_idx] == 2), np.count_nonzero(y_labels[test_idx] == 2))
            split_num = split_num + 1

            X_train_features = x[train_idx]
            X_test_features = x[test_idx]
            y_train = y[train_idx]
            print('Before smote', X_train_features.shape, X_test_features.shape)

            if smote_status == 1:
                x_train_new, y_train_new = smote_data(x[train_idx], y[train_idx], site[train_idx])
                X_train_features = x_train_new
                y_train = y_train_new
                print('After smote', X_train_features.shape, X_test_features.shape)

            if pca_status == 1:
                X_train_pca, X_test_pac, scaler, pca, num_comp = perform_pca(X_train_features, X_test_features)
                print('X_train_pca, X_test_pac', X_train_pca.shape, X_test_pac.shape)
            else:
                X_train_pca, X_test_pac, scaler, pca, num_comp = X_train_features, X_test_features, 0, 0, 0

            info_list = [X_train_pca, y_train, X_test_pac, y[test_idx], 'repeat'+str(item_repeat+1), split_num,
                          seed_num[item_repeat], train_idx, test_idx]
            info_list2 = [train_idx, test_idx, 'repeat' + str(item_repeat + 1), split_num, seed_num[item_repeat]]

            input_list.append(info_list)
            input_list2.append(info_list2)

    # input_list = sorted(input_list, key=itemgetter(0))
    # print('input_list', input_list)
    return input_list, input_list2



def prepare_input(x, y, info_list):
    input_list = list()
    for item in info_list:
        train_idx = item[0]
        test_idx = item[1]
        X_train_features = x[train_idx]
        X_test_features = x[test_idx]
        X_train_pca, X_test_pac = perform_pca(X_train_features, X_test_features)
        print('X_train_pca, X_test_pac', X_train_pca.shape, X_test_pac.shape)

        info_list = [X_train_pca, y[train_idx], X_test_pac, y[test_idx], item[2], item[3], item[4]]
        input_list.append(info_list)

    return input_list


def save_kfolds(modelnames, num_repeats, num_kfolds, input_list, out_list, master_seed, time_in, time_in_sec):
    model_dict = dict()   # for saving models
    pred_dict = dict()    # for prediction results
    log_dict = dict()     # for train idx, seed no. etc
    out_dict = dict()     # log_dict + processing time, master seed
    output_list = list()  # list of output values- models, predictions

    for key in modelnames:
        model_dict[key] = {}
        pred_dict[key] = {}


    for key in modelnames:
        for item_repeat in range(0, num_repeats, 1):
            model_dict[key]['repeat'+ str(item_repeat + 1)] = {'model':[], 'best_score':[], 'best_param': []}
            pred_dict[key]['repeat'+ str(item_repeat + 1)] = {'MAE': [], 'CORR': [], 'MSE': [], 'Predicted_values': [],
                                                              'True_values':[], 'train_MAE': [], 'train_CORR': [],
                                                              'train_MSE': [], 'train_Predicted_values': [], 'test_idx':[]}
            # pred_dict[key]['repeat' + str(item_repeat + 1)] = {'MAE': [], 'CORR': [], 'Predicted_values': []}

    for item_repeat in range(0, num_repeats, 1):
        log_dict['repeat' + str(item_repeat + 1)] = []
    print('log_dict', log_dict)

    if out_list:
        for item_in in out_list:
            m = item_in[0]
            model = item_in[1]
            pred_var = item_in[2]
            item_rpt = item_in[3]
            split_no = item_in[4]
            best_score = item_in[5]
            best_param = item_in[6]
            train_var = item_in[7]
            test_idx = item_in[-1]

            if m in model_dict:
                m_dict = model_dict[m]
                p_dict = pred_dict[m]

                if item_rpt in m_dict and p_dict:
                    m_dict[item_rpt]['model'].append(model)
                    m_dict[item_rpt]['best_score'].append(best_score)
                    m_dict[item_rpt]['best_param'].append(best_param)
                    p_dict[item_rpt]['MAE'].append(pred_var['MAE'])
                    p_dict[item_rpt]['CORR'].append(pred_var['CORR'])
                    p_dict[item_rpt]['MSE'].append(pred_var['MSE'])
                    p_dict[item_rpt]['Predicted_values'].append(pred_var['Predicted_values'])
                    p_dict[item_rpt]['True_values'].append(pred_var['True_values'])
                    p_dict[item_rpt]['train_MAE'].append(train_var['MAE'])
                    p_dict[item_rpt]['train_CORR'].append(train_var['CORR'])
                    p_dict[item_rpt]['train_MSE'].append(train_var['MSE'])
                    # p_dict[item_rpt]['train_Predicted_values'].append(train_var['Predicted_values'])
                    p_dict[item_rpt]['test_idx'].append(test_idx)

        # print('model_dict', model_dict)
    # print('pred_dict', pred_dict)

    # if out_list_rfr:
    #     for item_in in out_list_rfr:
    #         m = 'COMB'
    #         model = item_in[1]
    #         pred_var = item_in[2]
    #         item_rpt = item_in[3]
    #         split_no = item_in[4]
    #         best_score = item_in[5]
    #         best_param = item_in[6]
    #
    #         if m in model_dict:
    #             m_dict = model_dict[m]
    #             p_dict = pred_dict[m]
    #
    #             if item_rpt in m_dict and p_dict:
    #                 m_dict[item_rpt]['model'].append(model)
    #                 m_dict[item_rpt]['best_score'].append(best_score)
    #                 m_dict[item_rpt]['best_param'].append(best_param)
    #                 p_dict[item_rpt]['MAE'].append(pred_var['MAE'])
    #                 p_dict[item_rpt]['CORR'].append(pred_var['CORR'])
    #                 p_dict[item_rpt]['Predicted_values'].append(pred_var['Predicted_values'])
    #     print('model_dict', model_dict)
    #     print('pred_dict', pred_dict)

    # for i in range(0, num_repeats * num_kfolds):
    #     item_master = input_list[i]
    #     print('item_master', item_master)
    #     rpt_no = item_master[3]
    #     print('rpt_no', rpt_no)
    #     if rpt_no in log_dict:
    #         log_dict[rpt_no].append({'train_idx': item_master[1], 'test_idx': item_master[2],
    #                                  'split_num': item_master[4], 'seed_num': item_master[5]})
    #
    # out_dict = {'repeats+splits': log_dict, 'master_seed': master_seed, 'start_time': time_in,
    #             'end_time': datetime.datetime.now(), 'processing_time': time.time() - time_in_sec}

    for i in range(0, num_repeats * num_kfolds):
        item_master = input_list[i]
        rpt_no = item_master[5]
        if rpt_no in log_dict:
            log_dict[rpt_no].append({'split_num': item_master[6], 'seed_num': item_master[7]})

    out_dict = {'repeats+splits': log_dict, 'master_seed': master_seed, 'start_time': time_in,
                'end_time': datetime.datetime.now(), 'processing_time': time.time() - time_in_sec}

    return model_dict, pred_dict, out_dict




# def train_models(x, y, num_repeats, seed_num, num_process, model_name):
#     # input_list = create_input_list(x, y, num_repeats, seed_num, model_name)
#     # print('input_list', input_list, len(input_list))
#     p = Pool(num_process)  # to initiate multiple processes
#     out_list = p.map(train_test, input_list)
#     # print('out_list', out_list)
#     p.close()
#     p.join()
#     return out_list

def train_models(input_list, num_process, model_name):
    print('train broad age model')
    p = Pool(num_process)  # to initiate multiple processes
    out_list = p.map(train_test, input_list)
    # print('out_list', out_list)
    p.close()
    p.join()
    return out_list


def performance_measures(y_pred, y_test):
    MAE = np.mean(np.abs(y_pred - y_test))
    CORR = np.corrcoef(y_pred, y_test)[1, 0]
    return MAE, CORR


# def make_predictions(x, lad_model1, group_models):
#     pred_all = np.array([])  # prediction from specific age model which includes broad model prediction
#
#     for test_ind, test_item in enumerate(x):
#         mod_pred_all = list()
#
#         mod1_pred = lad_model1.predict(test_item.reshape(1, -1))
#         mod1_pred = np.round(mod1_pred)
#         mod_pred_all.append(mod1_pred)
#
#         for key_nm, mod in group_models.items():  # make predictions using age-specific group
#             lad = group_models[key_nm]['model']
#             lad_pred = lad.predict(test_item.reshape(1, -1))
#             lad_pred = np.round(lad_pred)
#             mod_pred_all.append(lad_pred)
#
#         # xx = np.array(mod_pred_all)
#         # xxt = np.array(mod_pred_all).transpose()
#         # print('len(mod_pred_all)', len(mod_pred_all), xx.shape, xxt.shape)
#
#         if pred_all.size == 0:
#             pred_all = np.array(mod_pred_all).transpose()
#         else:
#             pred_all = np.concatenate((pred_all, np.array(mod_pred_all).transpose()), axis=0)
#     print('size', pred_all.shape)
#     return pred_all

def random_samples_training(X_train, Y_train, site):

    # group the data site-wise
    site_dict = {}
    unique_sites = list(set(site))
    site_dict = {value: [i for i, v in enumerate(site) if v == value] for value in unique_sites}
    # print('site_dict', site_dict)

    for idx, item in site_dict.items():
        print(idx, len(item))


    # create age bins
    age_range = list()
    max_age = np.max(Y_train)
    min_age = np.min(Y_train)
    for i in range(min_age, max_age+1):  # get all ages in the range of min and max of the age
        age_range.append(i)
    print('max-min of ages', max_age, min_age, age_range)

    n = 4  # groups of 4
    m = 0  # overlapping 0 ages
    age_bin_all = list()

    for i in range(0, max_age-min_age, n - m):
        age_bin = age_range[i:i + n]  # creates a age-group with 10 ages
        age_bin_all.append(age_bin)
        print('age_bin', age_bin)

    # for site_id, site_item in site_dict.items():
    #     print('site', site_id, site_item)
    #     for bin_item in age_bin_all:  # for every age in the age group collect the training data by getting the indices
    #         # print('bin_item',bin_item)
    #         for item in bin_item:
    #             print('item', item)
    #             x = [site_item for x in Y_train[site_item] if x == item]
    #             print(x)

        #     for idx, val in enumerate(Y_train):
        #         if val == item:
        #             idx_grp.append(idx)
        #
        # # print(age_group, len(idx_grp))
        # if len(idx_grp) > 100:
        #     idx_grp_new = random.sample(idx_grp, k=100)
        # else:
        #     idx_grp_new = idx_grp
        # # print(len(idx_grp_new), idx_grp_new)
        #
        # for ids in idx_grp_new:
        #     train_idx.append(ids)


    # print(len(train_idx), train_idx)

    return X_train[train_idx], Y_train[train_idx]

    # for site_id, site_item in site_dict.items():
    #     idx_grp = list()
    #     for bin_item in age_bin_all:  # for every age in the age group collect the training data by getting the indices
    #         for idx, val in enumerate(Y_train):
    #             if val == item:
    #                 idx_grp.append(idx)
    #
    #     # print(age_group, len(idx_grp))
    #     if len(idx_grp) > 100:
    #         idx_grp_new = random.sample(idx_grp, k=100)
    #     else:
    #         idx_grp_new = idx_grp
    #     # print(len(idx_grp_new), idx_grp_new)
    #
    #     for ids in idx_grp_new:
    #         train_idx.append(ids)
    #
    #
    # # print(len(train_idx), train_idx)
    #
    # return X_train[train_idx], Y_train[train_idx]









