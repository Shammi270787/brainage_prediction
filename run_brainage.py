#!/home/smore/.venvs/py3smore/bin/python3
import argparse
import pickle
import datetime
import time
import pandas as pd
from group_pred import *
from additional_funs import *
from train_predict import *
from sklearn.model_selection import train_test_split
from confound_removal import *
import pdb
import copy
from read_data_funcs import load_pca_files, create_labels_3grps, read_data_new2
from bias_variance import bias_variance_decomposition


if __name__ == '__main__':
    time_in = datetime.datetime.now()
    time_in_sec = time.time()
    all_xtrain = list()
    all_ytrain = list()
    all_pred = list()
    all_pred_flat = list()
    all_xtrain_flat = list()
    all_ytrain_flat = list()
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, nargs='?', const=1, default="RR",
                       help="models to use (comma seperated no space): RR,LinearSVC")
    parser.add_argument("num_repeats", type=int, help="Number of Repeats", nargs='?', const=1, default=1)
    parser.add_argument("num_folds", type=int, help="Number of folds", nargs='?', const=1, default=3)
    parser.add_argument("master_seed", type=int, help="Random seed", nargs='?', const=1) # will take default as none
    parser.add_argument("num_processes", type=int, help="Number of Processors to use", nargs='?', const=1, default=1)
    parser.add_argument("which_site", type=int, help="Put site number", nargs='?', const=1)
    parser.add_argument("Idx_features", type=int, help="Idx Feature Nos.", nargs='?', const=1)
    # parser.add_argument("inputfilenm", type=str, help="Input file name")  # ixi_camcan # used for saving dictionary for future use
    parser.add_argument("inputpath1", type=str, help="Input data path")  # /data/project/brainage/corr_data_processing/ixi_camcan_features'
    parser.add_argument("inputpath2", type=str,
                        help="Input data path")  # /data/project/brainage/corr_data_processing/ixi_camcan_features'
    parser.add_argument("outputpath", type=str, help="Output file path") #'/data/project/brainage/CoRR_test_retest/results/'
    parser.add_argument("outputfilenm", type=str, help="Output file name")  # corr_ses1_200
    parser.add_argument("features_or_inputlist", type=int, help="Original features or CV input list", nargs='?', const=1, default=1)
    parser.add_argument("num_pca", type=int, help="Number of PCA components to use", nargs='?', const=1, default=25)
    parser.add_argument("pca_status", type=int, help="Number of PCA components to use", nargs='?', const=1, default=0)
    parser.add_argument("smote_status", type=int, help="Smote status", nargs='?', const=1, default=0)

    args = parser.parse_args()
    print(args.models, type(args.models))
    modelnames = [x.strip() for x in args.models.split(',')]  # converts string into list
    print('modelnames', modelnames)

    datafile1 = args.inputpath1
    datafile2 = args.inputpath2
    outputpath = args.outputpath
    outputfilenm = args.outputfilenm

    input1 = args.features_or_inputlist
    num_comps = args.num_pca
    pca_status = args.pca_status
    smote_status = args.smote_status

    print('input1', input1)

    if input1 == 1:  # load features file- mat,csv,pickled dataframe
        df1, num_repeats, num_kfolds, master_seed, num_process = read_data_new2(args, datafile1)
        train_df1, test_df = train_test_split(df1, test_size=0.3, stratify=df1['bin'], random_state=master_seed)

        df2, num_repeats, num_kfolds, master_seed, num_process = read_data_new2(args, datafile2)

        train_df = pd.concat([train_df1, df2], axis=0)
        # train_df = train_df1

        y_labels, y, gender, site, sub_nm = train_df['bin'].values, train_df['age'].values, train_df['gender'].values,\
                                            train_df['site'].values, train_df['subject'].values
        x = train_df.drop(['age', 'gender', 'site', 'bin', 'subject'], axis=1).values

        print('final data matrix x and y shapes', x.shape, y.shape)
        print('y_labels:{},  y: {}, gender: {},  site: {} '.format(str(y_labels.shape), str(y.shape), str(gender.shape),
                                                                   str(site.shape)))

        # X_train, Y_train, Y_train_labels = x[:, 0:423], y, y_labels
        X_train, Y_train, Y_train_labels = x, y, y_labels

        data_dict = {'X_train': X_train, 'Y_train': Y_train, 'Y_train_labels': y_labels}

        input_list, input_info_list = kfolds_fun_pred_smote(X_train, Y_train, Y_train_labels, num_repeats, num_kfolds,
                                                            master_seed, modelnames, num_process, pca_status, site,
                                                            smote_status)


    elif input1 == 2:  # PCA processed pickled input list for all the models
        num_repeats, master_seed, num_process, num_kfolds, input_list, X_train, Y_train = load_pca_files(args, outputfilenm, outputpath)
        Y_train_labels = create_labels_3grps(Y_train)

    if pca_status == 1:
        comp = np.size(X_train, 0) - int(np.size(X_train, 0)/num_kfolds)
#       num_comps = int(input('Enter number of principal components to use (max components < {}): '.format(str(comp))))
        print(num_comps, type(num_comps))
        for item in input_list:
            item[0] = item[0][:, 0:num_comps]  # for train
            item[2] = item[2][:, 0:num_comps]  # for test
            print(item[0].shape)

    new_input_list = list()
    new_list = list()
    for model_name in modelnames:
        print('model_name', model_name)
        for item in input_list:
            item2 = list()
            item2 = copy.deepcopy(item)
            item2.insert(0, model_name)
            new_input_list.append(item2)
    print('len(new_input_list)', len(new_input_list))
    new_input_list = sorted(new_input_list, key=itemgetter(0))

    p = Pool(num_process)  # to initiate multiple processes
    out_list = p.map(train_test, new_input_list)

    p.close()
    p.join()
    print('-----------DONE-----------')

    # saving results
    model_dict, pred_dict, out_dict = save_kfolds(modelnames, num_repeats, num_kfolds, new_input_list, out_list, master_seed, time_in, time_in_sec)

    output_path = outputpath + outputfilenm
    pickle.dump(model_dict, open(output_path + '.model', "wb"))
    pickle.dump(pred_dict, open(output_path + '.predict', "wb"))
    pickle.dump(out_dict, open(output_path + '.log', "wb"))
    pickle.dump(input_info_list, open(output_path + '.input_info', "wb"))
    pickle.dump(out_list, open(output_path + '.output_info', "wb"))
    # pickle.dump(out_list, open(output_path + '.out_list', "wb"))


    # Load test data:
    # df2 = pickle.load(open('/data/project/brainage/data/BSF_423/corr_ses1', "rb"))
    df2 = test_df
    df = pd.DataFrame(df2)
    Y_test = df['age']
    X_test = df.drop(['age', 'gender', 'site', 'subject', 'bin'], axis=1).values
    # X_test = X_test[:,0:423]
    Y_test = Y_test.astype(float).round()
    Y_test = Y_test.astype(int)


   # Generating final models
    model_dict_comb = dict()
    if smote_status == 1:
        X_train, Y_train = smote_data(X_train, Y_train, site)

    if pca_status == 1:
        X_train_pca, X_test_pca, scaler, pca, pca_comps = perform_pca(X_train, X_test)
        X_train_pca = X_train_pca[:, 0:num_comps]
        X_test_pca = X_test_pca[:, 0:num_comps]
    else:
        X_train_pca, X_test_pca, scaler, pca, pca_comps = X_train, X_test, 0, 0, 0

    for model_name in modelnames:
        print('Final training with model', model_name)
        print('After smoting', X_train_pca.shape)
        final_mod, best_score, best_params, pred_var, train_var = train(model_name, X_train_pca, Y_train, X_test_pca, Y_test)


        model_dict_comb[model_name] = {'model': final_mod, 'best_score': best_score, 'best_params': best_params,
                                      'test_predictions': pred_var, 'train_predictions': train_var, 'pca':
                                          {'scaler': scaler, 'pca': pca, 'pca_comp_gen': pca_comps, 'pca_comp_used': num_comps}}
        print('model_dict_comb', model_dict_comb)

    pickle.dump(model_dict_comb, open(output_path + '.model_comb', "wb"))










