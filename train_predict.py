#!/home/smore/.venvs/py3smore/bin/python3
import numpy as np
# from additional_funs import *
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
# from mlxtend.evaluate import bias_variance_decomp


def make_predictions(x, lad_model1, group_models):
    pred_all = np.array([])  # prediction from specific age model which includes broad model prediction

    for test_ind, test_item in enumerate(x):
        mod_pred_all = list()

        mod1_pred = lad_model1.predict(test_item.reshape(1, -1))
        # mod1_pred = np.round(mod1_pred)
        mod_pred_all.append(mod1_pred)

        for key_nm, mod in group_models.items():  # make predictions using age-specific group
            # lad = group_models[key_nm]['model']
            lad = group_models[key_nm]
            lad_pred = lad.predict(test_item.reshape(1, -1))
            # lad_pred = np.round(lad_pred)
            mod_pred_all.append(lad_pred)

        # xx = np.array(mod_pred_all)
        # xxt = np.array(mod_pred_all).transpose()
        # print('len(mod_pred_all)', len(mod_pred_all), xx.shape, xxt.shape)

        if pred_all.size == 0:
            pred_all = np.array(mod_pred_all).transpose()
        else:
            pred_all = np.concatenate((pred_all, np.array(mod_pred_all).transpose()), axis=0)
    print('size', pred_all.shape)
    return pred_all


def train(m, x_train, y_train,  x_test, y_test):
    print('training', m)
    model = []
    pred_var = {}

    if m == 'LAD':
        from mord import LAD
        lad = LAD(epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True,
              intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=10000)
        params = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
        model = GridSearchCV(lad, param_grid=params, cv=5, scoring='neg_mean_absolute_error', verbose=0)

        y_train = y_train.astype(float).round()
        y_train = y_train.astype(int)

        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)
        print("[INFO] LAD grid search best parameters: {}".format(model.best_params_))

    elif m =='MCLog': # this class is not avaialble
        from sklearn.linear_model import LogisticRegression
        mcl = LogisticRegression(multi_class='multinomial',max_iter=10000,
                    solver='newton-cg', fit_intercept=True)
        params = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
        model = GridSearchCV(mcl, param_grid=params, cv=5,
                scoring='neg_mean_absolute_error', verbose=0)
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)
        print("[INFO] MCLog grid search best parameters: {}".format(model.best_params_))

    elif m =='LogAT': # takes quite some time
        from mord import LogisticAT
        lat = LogisticAT()
        params = {"alpha": np.linspace(0,1,5)}
        model = GridSearchCV(lat, param_grid=params, cv=5,
                scoring='neg_mean_absolute_error', verbose=0)
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)
        print("[INFO] LogAT grid search best parameters: {}".format(model.best_params_))

    elif m =='LinearSVC':
        from sklearn.svm import LinearSVC
        svm = LinearSVC()
        params = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
        model = GridSearchCV(svm, param_grid=params, cv=5, verbose=0)
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)
        print("[INFO] LinearSVC grid search best parameters: {}".format(model.best_params_))

    elif m =='RFC':
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier()
        params = {"n_estimators": [10, 100, 500, 1000]}
        model = GridSearchCV(rfc, param_grid=params, cv=5, verbose=0)
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)
        print("[INFO] RFC grid search best parameters: {}".format(model.best_params_))

    elif m == 'Lasso':
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import LassoCV
        svm = Lasso()
        params = {"alpha": [10]}
        model = GridSearchCV(svm, param_grid=params, cv=5, verbose=0)
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)
        print("[INFO] RFR grid search best parameters: {}".format(model.best_params_))
        # model = LassoCV(n_alphas=10, cv=5, verbose=3)
        # model.fit(x_train, y_train)
        # print("[INFO] Lasso path search best parameter: {}".format(model.alpha_))

    elif m == 'RFR':
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(criterion='mse')
        params = {"n_estimators": [500]}
        model = GridSearchCV(rfr, param_grid=params, cv=5, verbose=0)
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)
        print("[INFO] RFR grid search best parameters: {}".format(model.best_params_))

    elif m == 'RR':
        from sklearn.linear_model import Ridge, RidgeCV
        ridge = Ridge()
        params = {'alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        model = GridSearchCV(ridge, param_grid=params, cv=5, verbose=0)
        model.fit(x_train, y_train)
        print("[INFO] Ridge Regression grid search best parameters: {}".format(model.best_params_))
        # model = RidgeCV(alphas=(0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), cv=5)
        # model.fit(x_train, y_train)
        # print("[INFO] Ridge Regression grid search best parameters: {}".format(model.alpha_))
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)

    elif m == 'PLSR':
        from sklearn.cross_decomposition import PLSRegression
        pls_reg = PLSRegression()
        params = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
        model = GridSearchCV(pls_reg, param_grid=params, cv=5, verbose=0)
        # pdb.set_trace()
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        print("[INFO] PLS Regression grid search best parameters: {}".format(model.best_params_))
        pred_var = predict(m, model, x_test, y_test)

    elif m == 'RVM':
        from skrvm import RVR
        print('in RVM')
        model = RVR(kernel='linear')
        # avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, x_train, y_train, x_test, y_test, loss='mse',
        #                                                             num_rounds=3, random_seed=123)
        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)

        # print('Average expected loss: %.3f' % avg_expected_loss)
        # print('Average bias: %.3f' % avg_bias)
        # print('Average variance: %.3f' % avg_var)

    elif m == 'DTR':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        # params = {"criterion": ["mse", "mae"], "min_samples_split": [10, 20, 40], "max_depth": [2],
        #           "min_samples_leaf": [20, 40, 100], "max_leaf_nodes": [5, 20, 100]}
        # params = {"max_depth": [2,4,6]}
        # model = GridSearchCV(dtr, param_grid=params, cv=5, verbose=0)

        model.fit(x_train, y_train)
        train_var = predict(m, model, x_train, y_train)
        pred_var = predict(m, model, x_test, y_test)

    elif m == 'COMB':
        from sklearn.ensemble import RandomForestRegressor
        from mord import LAD
        from group_pred import create_age_groups
        print('IN COMB')
        group_lad = dict()

        print('shapes', x_train.shape, y_train.shape)

        lad1 = LAD(epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True,
                  intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=10000)
        params = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
        broad_lad = GridSearchCV(lad1, param_grid=params, cv=5, scoring='neg_mean_absolute_error', verbose=0)

        y_train_r = y_train.astype(float).round()
        y_train_r = y_train_r.astype(int)

        broad_lad.fit(x_train, y_train_r)

        age_group_all = create_age_groups(y_train_r, 10, 5)

        for ages in age_group_all:
            # print('ages', ages)
            idx_grp = list()
            for item in ages:  # for every age in the age group collect the training data by getting the indices
                for idx, val in enumerate(y_train_r):
                    if val == item:
                        idx_grp.append(idx)

            print('group info', ages, len(idx_grp))
            if len(idx_grp) > 5:
                key_age_grp = str(np.min(ages)) + '_' + str(np.max(ages))
                x_samples_train = x_train[idx_grp]
                y_samples_train = y_train_r[idx_grp]
                # print('y_samples_train', y_samples_train)

                lad2 = LAD(epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True,
                           intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=10000)
                params2 = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
                specific_lad = GridSearchCV(lad2, param_grid=params2, cv=5, scoring='neg_mean_absolute_error', verbose=0)
                specific_lad.fit(x_samples_train, y_samples_train)
                group_lad[key_age_grp] = specific_lad

        print('len_groups', len(group_lad))
        pred_all = make_predictions(x_train, broad_lad, group_lad)

        rfr = RandomForestRegressor(criterion='mse')
        params = {"n_estimators": [500]}
        model_2 = GridSearchCV(rfr, param_grid=params, cv=5, verbose=0)
        model_2.fit(pred_all, y_train)

        # lad = LAD(epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True,
        #            intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=10000)
        # params = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
        # model_2 = GridSearchCV(lad, param_grid=params, cv=5, scoring='neg_mean_absolute_error', verbose=0)
        # model_2.fit(pred_all, y_train_r)

        train_var = predict(m, model_2, pred_all, y_train)
        print("[INFO] RFR grid search best parameters: {}".format(model_2.best_params_))

        pred_all_test = make_predictions(x_test, broad_lad, group_lad)
        pred_var = predict(m, model_2, pred_all_test, y_test)
        model = [broad_lad, group_lad, model_2]
    else:
        print('unknown model')

    if m == 'RVM' or 'DTR':
        return model, 0, 0, pred_var, train_var
    elif m == 'COMB':
        return model, model_2.best_score_, model_2.best_params_, pred_var, train_var
    else:
        return model, model.best_score_, model.best_params_, pred_var, train_var


def predict(modname, model, x_test, y_test):
    pred_var = dict()  # create an empty dictionary for saving prediction results

    if modname == 'Lasso' or 'RVM' or 'DTR':
        modbest = model
    else:
        modbest = model.best_estimator_

    y_pred = modbest.predict(x_test)
    y_pred = y_pred.astype(float).round()
    y_pred = y_pred.astype(int)

    if modname == "PLSR":
        y_pred = np.ravel(y_pred)

    MAE, CORR, MSE = performance_measures(y_pred, y_test)
    pred_var = {'True_values': y_test, 'Predicted_values': y_pred, 'MAE': MAE, 'CORR': CORR, 'MSE':MSE}

    return pred_var


def performance_measures(y_pred, y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    MAE = np.mean(np.abs(y_pred - y_test))
    # mae = mean_absolute_error(y_test,y_pred)
    MSE = mean_squared_error(y_test, y_pred)# squared=False will give RMSE
    # print(MSE, np.mean((y_pred - y_test)**2))
    # print('MAE', MAE, mae, MSE)
    CORR = np.corrcoef(y_pred, y_test)[1, 0]
    
    biased_squared  = np.mean((y_pred - y_test)**2)
    
    return MAE, CORR, MSE


def final_train(x, y, x_test, y_test, out_list, mn, age_group_all):
    model = []
    best_score = []

    if mn == 'LAD':
        print(out_list)
        [C_list, score_list] = zip(*[(item[6]['C'], item[5]) for item in out_list])
        C_final = np.median(C_list)
        best_score = np.mean(score_list)
        print('in final LAD')
        print('para', C_list, C_final, 'score', score_list, best_score)
        model = LAD(epsilon=0.0, tol=0.0001, C=C_final, loss='epsilon_insensitive', fit_intercept=True,
                  intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=10000)
        model.fit(x, y)
        pred_var = predict(mn, model, x_test, y_test)

    elif mn == 'RFR':
        [n_est_list, score_list] = zip(*[(item[6]['n_estimators'], item[5]) for item in out_list])
        n_est = int(np.median(n_est_list))
        best_score = np.mean(score_list)
        print('in final RFR')
        print('n_est_list', n_est_list, n_est, 'score', score_list, best_score)
        rfr = RandomForestRegressor(criterion='mse')
        params = {"n_estimators": [n_est]}
        model = GridSearchCV(rfr, param_grid=params, cv=5, verbose=0)
        model.fit(x, y)
        pred_var = predict(mn, model, x_test, y_test)

    elif mn == 'PLSR':
        [n_comp_list, score_list] = zip(*[(item[6]['n_components'], item[5]) for item in out_list])
        n_comp = int(np.median(n_comp_list))
        best_score = np.mean(score_list)
        print('in final PLSR')
        print('n_comp_list', n_comp_list, n_comp, 'score', score_list, best_score)
        pls_reg = PLSRegression()
        params = {'n_components': [n_comp]}
        model = GridSearchCV(pls_reg, param_grid=params, cv=5, verbose=0)
        model.fit(x, y)
        pred_var = predict(mn, model, x_test, y_test)

    elif mn == 'RR':
        from sklearn.linear_model import Ridge, RidgeCV
        [n_comp_list, score_list] = zip(*[(item[6]['alpha'], item[5]) for item in out_list])
        n_comp = int(np.median(n_comp_list))
        best_score = np.mean(score_list)
        print('in final RR')
        print('n_comp_list', n_comp_list, n_comp, 'score', score_list, best_score)
        ridge = Ridge()
        params = {'alpha': [n_comp]}
        model = GridSearchCV(ridge, param_grid=params, cv=5, verbose=0)
        model.fit(x, y)
        pred_var = predict(mn, model, x_test, y_test)

    elif mn == 'RVM':
        from skrvm import RVR
        print('in final RVM')
        model = RVR(kernel='linear')
        model.fit(x, y)
        best_score = 0
        pred_var = predict(mn, model, x_test, y_test)

    elif mn == 'COMB':
        print('IN COMB')
        group_lad = dict()
        from mord import LAD
        from sklearn.ensemble import RandomForestRegressor

        print('shapes', x.shape, y.shape)

        lad1 = LAD(epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True,
                  intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=10000)
        params = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
        broad_lad = GridSearchCV(lad1, param_grid=params, cv=5, scoring='neg_mean_absolute_error', verbose=0)
        broad_lad.fit(x, y)

        for ages in age_group_all:
            # print('ages', ages)
            idx_grp = list()
            for item in ages:  # for every age in the age group collect the training data by getting the indices
                for idx, val in enumerate(y):
                    if val == item:
                        idx_grp.append(idx)

            key_age_grp = str(np.min(ages)) + '_' + str(np.max(ages))
            x_samples_train = x[idx_grp]
            y_samples_train = y[idx_grp]
            # print('y_samples_train', y_samples_train)

            lad2 = LAD(epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True,
                       intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=10000)
            params2 = {"C": [0.001, 0.01, 1, 10, 100, 1000]}
            specific_lad = GridSearchCV(lad2, param_grid=params2, cv=5, scoring='neg_mean_absolute_error', verbose=0)
            specific_lad.fit(x_samples_train, y_samples_train)
            group_lad[key_age_grp] = specific_lad

        pred_all = make_predictions(x, broad_lad, group_lad)
        rfr = RandomForestRegressor(criterion='mse')
        params = {"n_estimators": [500]}
        model = GridSearchCV(rfr, param_grid=params, cv=5, verbose=0)
        model.fit(pred_all, y)
        print("[INFO] RFR grid search best parameters: {}".format(model.best_params_))

        best_score = model.best_score_
        pred_all_test = make_predictions(x_test, broad_lad, group_lad)
        pred_var = predict(mn, model, pred_all_test, y_test)

    return model, best_score, pred_var

