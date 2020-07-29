#!/home/smore/.venvs/py3smore/bin/python3
import sys
import numpy as np
from train_predict import train
from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression


# def confound_features_fun(site, gender):
#     print(site, type(site), site.shape, len(site))
#     print(gender, type(gender), gender.shape, len(gender))
#     # cat = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#
#     onehot_encoder1 = OneHotEncoder(sparse=False, categories='auto')
#     onehot_encoded_site = onehot_encoder1.fit_transform(site.reshape(-1, 1))
#
#     values = np.array(gender)
#     print('values', values)
#     label_encoder = LabelEncoder()
#     integer_encoded = label_encoder.fit_transform(values)
#     print('integer_encoded', integer_encoded)
#     onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
#     integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#     onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#     np.set_printoptions(threshold=sys.maxsize)
#     print('onehot_encoded', onehot_encoded)
#     inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
#     print('inverted', inverted)

def perform_confound_removal(site, gender,idx_train, idx_test, X_train, X_test, y):

    confound_features = confound_features_fun(site, gender)
    con_X_train, con_X_test, Y_train11, Y_test11 = confound_features[idx_train], confound_features[idx_test], y[idx_train], y[idx_test]
    print('Y_test', Y_test)
    print('Y_test11', Y_test11)

    # train confound models using confound features and actual features (one by one) as labels
    confound_models = confound_train_model(X_train, con_X_train)

    # calculate residual features by predicting from confound models
    X_train_features = residual_features_fun(con_X_train, confound_models, X_train)
    X_test_features = residual_features_fun(con_X_test, confound_models, X_test)

    confound_dict = {'X_train_features': X_train_features, 'X_test_features': X_test_features,
                     'counfound_models': confound_models, 'Y_train': Y_train, 'Y_test': Y_test}

    pickle.dump(confound_dict, open(args.outputfile + '.confound_dict', "wb"))

    return X_train_features, X_test_features


def confound_features_fun(site, gender):
    print('site_info', type(site), site.shape, len(site))
    print('gender_info', type(gender), gender.shape, len(gender))
    # cat = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    onehot_encoder1 = OneHotEncoder(sparse=False, categories='auto')
    onehot_encoded_site = onehot_encoder1.fit_transform(site.reshape(-1, 1))

    confound_features = onehot_encoded_site

    # onehot_encoder2 = OneHotEncoder(sparse=False, categories='auto')
    # onehot_encoded_gender = onehot_encoder2.fit_transform(gender.reshape(-1, 1))
    # np.set_printoptions(threshold=sys.maxsize)
    # confound_features = np.concatenate((onehot_encoded_site, onehot_encoded_gender), axis=1)
    # print('onehot_encoded_site', onehot_encoded_site.shape, 'onehot_encoded_gender', onehot_encoded_gender.shape)

    print('confound_features.shape', confound_features.shape)
    
    return confound_features


def confound_train_model(x, confound_features):
    confound_models = list()

    for i in range(0, x.shape[1]):
        labels = x[:, i]  # features as labels
        reg = LinearRegression()
        reg.fit(confound_features, labels)
        confound_models.append(reg)
        # print('Regression parameters for feature %d:' % i, reg.score(confound_features, labels), reg.coef_, reg.intercept_)
    print('len(confound_models)', len(confound_models))
    return confound_models


def residual_features_fun(confound_features, confound_models, x):
    print('x.shape', x.shape)
    residual_features = np.array([])

    for i in range(0, x.shape[1]):
        labels = x[:, i]
        # print('labels.shape', labels.shape)
        model = confound_models[i]
        y_pred = model.predict(confound_features)
        residuals = labels - y_pred
        # print('residuals', residuals, residuals.shape)
        residual_features = np.concatenate((residual_features, residuals), axis=0)
        # print('residual_features_every_loop', residual_features, residual_features, residual_features.shape)

    residual_features = residual_features.reshape(x.shape[1], x.shape[0])
    # print('residual_features_all', residual_features)
    print('residual_features.shape', residual_features.shape)
    residual_features = np.transpose(residual_features)
    print('residual_features.shape', residual_features.shape)
    return residual_features


def confound_models_fun(x, site, gender):
    confound_features = confound_features_fun(site, gender)
    confound_models = confound_train_model(x, confound_features)
    residual_features = residual_features_fun(confound_features, confound_models, x)
    return confound_models, residual_features





