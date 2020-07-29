#!/home/smore/.venvs/py3smore/bin/python3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np


def perform_pca(X_train, X_test):

    scaler = StandardScaler()
    print('before scaler')
    print(X_train.shape, type(X_train))
    scaler.fit(X_train) # Fit on training set only

#    print(scaler)
    print('after sacler')

    # Apply transform to both the training set and the test set.
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print('after transform')

    # if np.size(X_train, 0) < 600:
    #     num_comp = None # np.min(np.size(X_train, 0), np.size(X_train, 1))
    # else:
    #     num_comp = 600

    num_comp = min(np.size(X_train, 0), np.size(X_train, 1))
    print('Number of components PCA', num_comp)
    

    pca = PCA(n_components=num_comp) # pca = PCA(.99)
    pca.fit(X_train_scaled)

    print(pca)

    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

#    print("Original shape for train data: {}".format(str(X_train.shape)))
#    print("Reduced shape for train data: {}".format(str(X_train_pca.shape)))
#
#    print("Original shape for test data: {}".format(str(X_test.shape)))
#    print("Reduced shape for test data: {}".format(str(X_test_pca.shape)))

    # print('Explained Variance ratio:', pca.explained_variance_ratio_)

    return X_train_pca, X_test_pca, scaler, pca, num_comp


# train_data = pickle.load(open('/data/BnB_USER/smore/PAC_data_analysis/using_mni_template/pac_train_data', "rb"))
# train_data = np.array(train_data)
# print('train_data', train_data, len(train_data), type(train_data), train_data.shape)
#
# Y_train = train_data[:, 0]  # store Oth index of every row (which is age) in separate array 'y'
# Y_train = Y_train.astype(int)  # convert age into int (y: vector of age- OUTPUT)
# X_train = np.delete(train_data, 0, 1)  # remove (0,1) element (age) from all the rows (X: vector of features-INPUT)
# print('X_train.shape', X_train.shape, 'Y_train.shape', Y_train.shape, Y_train)
#
# test_data = pickle.load(open('/data/BnB_USER/smore/PAC_data_analysis/using_mni_template/pac_test_data', "rb"))
# test_data = np.array(test_data)
# print('test_data', test_data, len(test_data), type(test_data), test_data.shape)
#
# Y_test = test_data[:, 0]  # store Oth index of every row (which is age) in separate array 'y'
# Y_test = Y_test.astype(int)  # convert age into int (y: vector of age- OUTPUT)
# X_test = np.delete(test_data, 0, 1)  # remove (0,1) element (age) from all the rows (X: vector of features-INPUT)
# print('X_test.shape', X_test.shape, 'Y_test.shape', Y_test.shape, Y_test)
#
# X_train_pca, X_test_pac = perform_pca(X_train, X_test)

#pickle.dump(X_train_pca, open('X_train_pca', "wb"))
#pickle.dump(X_test_pac, open('X_test_pca', "wb"))
#pickle.dump(Y_train, open('Y_train_pca', "wb"))

#
# train_data = pickle.load(open('/data/BnB_USER/smore/PAC_data_analysis/X_train_pca', "rb"))
# print(train_data.shape)



