#!/home/smore/.venvs/py3smore/bin/python3
from additional_funs import *
from multiprocessing import Pool


def create_age_groups(y, n, m):
    age_range = list()
    max_age, min_age = np.max(y), np.min(y)

    for i in range(min_age, max_age+1): # get all ages in the range of min and max of the age
        age_range.append(i)

    age_groups = [age_range[i:i+n] for i in range(0, max_age-min_age, n-m) if len(age_range[i:i+n])==10]
    # print('age_groups', age_groups, 'len(age_groups)', len(age_groups))
    return age_groups


# def group_age_models(x, y, age_group, num_repeats, num_kfolds, num_process,master_seed, model_name):
#
#     group_model_dict = dict()
#
#     for ages in age_group:
#         idx_grp = list()
#         for item in ages: # for every age in the age group collect the training data by getting the indices
#             for idx, val in enumerate(y):
#                 if val == item:
#                     idx_grp.append(idx)
#
#         x_samples_train = x[idx_grp]
#         y_samples_train = y[idx_grp]
#         y_labels = create_labels_3grps(y_samples_train)
#
#         key_age_grp = str(np.min(ages)) + '_' + str(np.max(ages))
#         print('ages', ages)
#         print('y_samples_train', y_samples_train)
#         print('key_age_grp', key_age_grp)
#
#         num_kfolds = 3
#         input_list1, input_list2 = kfolds_fun_pred(x_samples_train, y_samples_train, y_labels, num_repeats, num_kfolds,
#                                                    master_seed, model_name, num_process)
#
#         p = Pool(num_process)  # to initiate multiple processes
#         out_list = p.map(train_test, input_list1)
#         p.close()
#         p.join()
#
#         # combining the model using the hyperparameter
#         lad, best_score = final_train(x_samples_train, y_samples_train, out_list, model_name[0])
#         group_model_dict[key_age_grp] = {'model': lad, 'best_score': best_score}
#
#     return group_model_dict


# def group_age_models(att):
#     mod_nm, x_train, y_train, x_test, y_test, item_rpt, split_no, seed_no = att
#
#     group_model_dict = dict()
#     age_group = create_age_groups(y_train, 10, 9)
#
#     print('age_group', age_group)
#     out_list_all = list()
#     for ages in age_group:
#         print('ages', ages)
#         idx_grp = list()
#         for item in ages: # for every age in the age group collect the training data by getting the indices
#             for idx, val in enumerate(y_train):
#                 if val == item:
#                     idx_grp.append(idx)
#
#         x_samples_train = x_train[idx_grp]
#         y_samples_train = y_train[idx_grp]
#
#
#         key_age_grp = str(np.min(ages)) + '_' + str(np.max(ages))
#         # print('ages', ages)
#         # print('y_samples_train', y_samples_train)
#         # print('key_age_grp', key_age_grp)
#
#         model, best_score, best_param = train(mod_nm, x_train, y_train)
#         pred_var = predict(mod_nm, model, x_test, y_test)
#         out_list = [key_age_grp, model, pred_var, item_rpt, split_no, best_score, best_param]
#         out_list_all.append(out_list)
#         # print('out_list', out_list)
#     return out_list_all



        # p = Pool(num_process)  # to initiate multiple processes
        # out_list = p.map(train_test, input_list1)
        # p.close()
        # p.join()
        #
        # # combining the model using the hyperparameter
        # lad, best_score = final_train(x_samples_train, y_samples_train, out_list, model_name[0])
        # group_model_dict[key_age_grp] = {'model': lad, 'best_score': best_score}

    # return group_model_dict





