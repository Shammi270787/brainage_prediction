#!/home/smore/.venvs/py3smore/bin/python3
import pandas as pd
import h5py
import os
import numpy as np
import pickle

def read_data_new1(args, df):
    num_repeats = args.num_repeats
    seed_num = args.master_seed
    num_process = args.num_processes
    num_folds = args.num_folds

    df = df.drop(['subject'], axis=1)

    # data = df.as_matrix(columns=None)
    age = df['age']
    age = age.astype(float).round()
    age = age.astype(int)
    df['age'] = age

    print('initial data size', df.shape)

    # get data from single site
    # if args.which_site is not None:
    #     df = df[df.site == args.which_site]
    #     print('args.which_site', args.which_site)
    #     # print('df_new',  df.head())

    labels = create_labels_3grps(df['age'])
    print('labels.shape', labels.shape)
    df['bin'] = labels
    print('df.shape', df.shape)
    print(df.head())
    return df, num_repeats, num_folds, seed_num, num_process


def read_data_new2(args, datafile):
    num_repeats = args.num_repeats
    seed_num = args.master_seed
    num_process = args.num_processes
    num_folds = args.num_folds

    filename, file_extension = os.path.splitext(datafile)

    if file_extension == '.mat':  # age_all(with different num of features)
        f = h5py.File(datafile)
        print(f.keys())
        struct_arr = f['GLMFlags']

        # print(f[struct_arr['dat']])
        x = f[struct_arr['dat'][args.Idx_features, 0]].value
        x = np.transpose(x)
        print('initial data size', x.shape)

        # site = struct_arr['Site'].value

        cov = struct_arr['Cov'].value
        age = cov[0, :]
        # age = age.astype(float).round()
        # age = age.astype(int)
        gender = cov[1, :].astype(int)  # Gender_F0_M1
        site_labels = cov[2, :].astype(int)
        print('shapes-age,gender,site', age.shape, gender.shape, site_labels.shape)

        df = pd.DataFrame(data=x)
        df['age'] = age
        df['site'] = site_labels
        df['gender'] = gender

        if args.which_site is not None: #take as input if one site data is required
            df = df[df.site == args.which_site]
            print('args.which_site', args.which_site)
            print('Number of samples in site', args.which_site, ':', len(df))
            # print('df_new',  df.head())

        labels = create_labels_3grps(df['age'])
        print('labels.shape', labels.shape)
        df['bin'] = labels
        print('df.shape', df.shape)
        print(df.head())

        # df = random_sampling(df)

    elif file_extension == '.csv':  # specifically for PAC
        print('in CSV')
        df2 = pd.read_csv(datafile, sep=",")
        df = pd.DataFrame(df2)
        df = df.drop(['subject'], axis=1)

        # data = df.as_matrix(columns=None)
        age = df['age']
        # age = age.astype(float).round()
        # age = age.astype(int)
        df['age'] = age

        print('initial data size', df.shape)

        # get data from single site
        # if args.which_site is not None:
        #     df = df[df.site == args.which_site]
        #     print('args.which_site', args.which_site)
        #     # print('df_new',  df.head())

        labels = create_labels_3grps(df['age'])
        print('labels.shape', labels.shape)
        df['bin'] = labels
        print('df.shape', df.shape)
        print(df.head())

    elif file_extension == '':
        df2 = pickle.load(open(datafile, "rb"))
        df = pd.DataFrame(df2)
        # df = df.drop(['subject'], axis=1)

        # data = df.as_matrix(columns=None)
        age = df['age']
        age = age.astype(float).round()
        age = age.astype(int)
        df['age'] = age

        print('initial data size', df.shape)
        labels = create_labels_3grps(df['age'])
        print('labels.shape', labels.shape)
        df['bin'] = labels
        print('df.shape', df.shape)
        print(df.head())

    else:  # for AgeAll.dat
        x = np.loadtxt(args.datfile)  # load data  in a numpy array
        age = x[:, 0]  # store Oth index of every row (which is age) in separate array
        age = age.astype(float).round()  # convert age into int (vector of age- OUTPUT)
        age = age.astype(int)
        x = np.delete(x, 0, 1)  # remove (0,1) element (age) from all the rows (X: vector of features-INPUT)
        labels = create_labels_3grps(age)

    return df, num_repeats, num_folds, seed_num, num_process


def create_labels_3grps(age):  # give dataframe as input
    df = pd.DataFrame({'age': age})

    # create 3 bins of ages of equal sizes
    qc = pd.cut(df.age.tolist(), bins=3, precision=1)
    print('age_bins', qc.categories)
    labels = qc.codes
    df['bin'] = qc.codes

    for bins in df.bin.unique():
        df_bins = df[df.bin == bins]
        print('samples per bin: ', bins, len(df_bins))
    # print(labels)
    return labels


def load_pca_files(args, input2, path):
    input_list = pickle.load(open(path + input2 + '.input_list', 'rb'))

    num_repeats = args.num_repeats
    master_seed = args.master_seed
    num_process = args.num_processes
    num_kfolds = args.num_folds

    data_file = pickle.load(open(path + input2 + '.data_dict', "rb"))
    X_train = data_file.get('X_train')
    # X_test = data_file.get('X_test')
    Y_train = data_file.get('Y_train')
    # Y_test = data_file.get('Y_test')
    # return num_repeats, master_seed, num_process, num_kfolds, input_list, X_train, X_test, Y_train, Y_test
    return num_repeats, master_seed, num_process, num_kfolds,input_list, X_train, Y_train

def load_pca_files2(args, input2, path):
    input_list = pickle.load(open(path + input2 + '.input_info_list', 'rb'))

    num_repeats = args.num_repeats
    master_seed = args.master_seed
    num_process = args.num_processes
    num_kfolds = args.num_folds

    data_file = pickle.load(open(path + input2 + '.data_dict', "rb"))
    X_train = data_file.get('X_train')
    X_test = data_file.get('X_test')
    Y_train = data_file.get('Y_train')
    Y_test = data_file.get('Y_test')

    return num_repeats, master_seed, num_process, num_kfolds,input_list, X_train, X_test, Y_train, Y_test
