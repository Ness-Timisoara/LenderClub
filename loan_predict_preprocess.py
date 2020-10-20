from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def run_dfmapper(data):

    print('DF MAPPER STARTED')

    X = data
    X_train = data
    X_valid = data

    print(X_train.columns)
    print(X_train.shape)

    validate = True
    onehot_cols = ['term', 'application_type', 'home_ownership', 'purpose']

    ordinal_cols = {'emp_length': [ '< 1 year',
                                      '1 year',
                                      '2 years',
                                      '3 years',
                                      '4 years',
                                      '5 years',
                                      '6 years',
                                      '7 years',
                                      '8 years',
                                      '9 years',
                                      '10+ years']}
    transformer = DataFrameMapper(
        [(onehot_cols, OneHotEncoder(drop="if_binary")),
         (list(ordinal_cols.keys()), OrdinalEncoder(categories=list(ordinal_cols.values())),),
         ], default=StandardScaler(),
    )

    X_train_New = X_train
    X_train_New = X_train_New.append({'term': '36 months'}, ignore_index=True)
    X_train_New = X_train_New.append({'term': '60 months'}, ignore_index=True)

    X_train_New = X_train_New.append({'application_type': 'Individual'}, ignore_index=True)
    X_train_New = X_train_New.append({'application_type': 'Joint App'}, ignore_index=True)

    X_train_New = X_train_New.append({'home_ownership': 'MORTGAGE'}, ignore_index=True)
    X_train_New = X_train_New.append({'home_ownership': 'RENT'}, ignore_index=True)
    X_train_New = X_train_New.append({'home_ownership': 'OWN'}, ignore_index=True)
    X_train_New = X_train_New.append({'home_ownership': 'ANY'}, ignore_index=True)
    X_train_New = X_train_New.append({'home_ownership': 'NONE'}, ignore_index=True)
    X_train_New = X_train_New.append({'home_ownership': 'OTHER'}, ignore_index=True)

    X_train_New = X_train_New.append({'purpose': 'debt_consolidation'	} , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'small_business'       } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'home_improvement'     } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'major_purchase'       } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'credit_card'          } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'other'                } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'house'                } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'car'                  } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'medical'              } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'vacation'             } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'moving'               } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'renewable_energy'     } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'wedding'              } , ignore_index=True)
    X_train_New = X_train_New.append({'purpose' : 'educational'          } , ignore_index=True)

    print('X_train_New.shape' ,X_train_New.shape)

    print('#######PRE')
    print('X_train.shape = ' ,X_train.shape)
    print('X_train.type = ' ,type(X_train))

    print('X_valid.shape = ' ,X_valid.shape)
    print('X_valid.type = ' ,type(X_valid))

    print('onehot_cols=' ,onehot_cols)
    print('list(ordinal_cols.keys())=' ,list(ordinal_cols.keys()))
    print('list(ordinal_cols.values())=' ,list(ordinal_cols.values()))
    print('X_train = ' ,X_train)
    print('#######')



    X_train_New.loan_amnt.fillna(0 ,inplace=True)
    X_train_New.term.fillna('36 months' ,inplace=True)
    X_train_New.emp_length.fillna('7 years' ,inplace=True)
    X_train_New.home_ownership.fillna('OWN' ,inplace=True)
    X_train_New.annual_inc.fillna(0 ,inplace=True)
    X_train_New.purpose.fillna('vacation' ,inplace=True)
    X_train_New.dti.fillna(0 ,inplace=True)
    X_train_New.delinq_2yrs.fillna(0 ,inplace=True)
    X_train_New.cr_hist_age_mths.fillna(0 ,inplace=True)
    X_train_New.fico_range_low.fillna(0 ,inplace=True)
    X_train_New.fico_range_high.fillna(0 ,inplace=True)
    X_train_New.inq_last_6mths.fillna(0 ,inplace=True)
    X_train_New.inv_mths_since_last_delinq.fillna(0 ,inplace=True)
    X_train_New.inv_mths_since_last_record.fillna(0 ,inplace=True)
    X_train_New.open_acc.fillna(0 ,inplace=True)
    X_train_New.pub_rec.fillna(0 ,inplace=True)
    X_train_New.revol_bal.fillna(0 ,inplace=True)
    X_train_New.revol_util.fillna(0 ,inplace=True)
    X_train_New.total_acc.fillna(0 ,inplace=True)
    X_train_New.collections_12_mths_ex_med.fillna(0 ,inplace=True)
    X_train_New.inv_mths_since_last_major_derog.fillna(0 ,inplace=True)
    X_train_New.application_type.fillna('Individual' ,inplace=True)
    X_train_New.annual_inc_joint.fillna(0 ,inplace=True)
    X_train_New.dti_joint.fillna(0 ,inplace=True)
    X_train_New.acc_now_delinq.fillna(0 ,inplace=True)
    X_train_New.tot_coll_amt.fillna(0 ,inplace=True)
    X_train_New.tot_cur_bal.fillna(0 ,inplace=True)
    X_train_New.total_rev_hi_lim.fillna(0 ,inplace=True)
    X_train_New.acc_open_past_24mths.fillna(0 ,inplace=True)
    X_train_New.avg_cur_bal.fillna(0 ,inplace=True)
    X_train_New.bc_open_to_buy.fillna(0 ,inplace=True)
    X_train_New.bc_util.fillna(0 ,inplace=True)
    X_train_New.chargeoff_within_12_mths.fillna(0 ,inplace=True)
    X_train_New.delinq_amnt.fillna(0 ,inplace=True)
    X_train_New.mo_sin_old_il_acct.fillna(0 ,inplace=True)
    X_train_New.mo_sin_old_rev_tl_op.fillna(0 ,inplace=True)
    X_train_New.inv_mo_sin_rcnt_rev_tl_op.fillna(0 ,inplace=True)
    X_train_New.inv_mo_sin_rcnt_tl.fillna(0 ,inplace=True)
    X_train_New.mort_acc.fillna(0 ,inplace=True)
    X_train_New.inv_mths_since_recent_bc.fillna(0 ,inplace=True)
    X_train_New.inv_mths_since_recent_bc_dlq.fillna(0 ,inplace=True)
    X_train_New.inv_mths_since_recent_inq.fillna(0 ,inplace=True)
    X_train_New.inv_mths_since_recent_revol_delinq.fillna(0 ,inplace=True)
    X_train_New.num_accts_ever_120_pd.fillna(0 ,inplace=True)
    X_train_New.num_actv_bc_tl.fillna(0 ,inplace=True)
    X_train_New.num_actv_rev_tl.fillna(0 ,inplace=True)
    X_train_New.num_bc_sats.fillna(0 ,inplace=True)
    X_train_New.num_bc_tl.fillna(0 ,inplace=True)
    X_train_New.num_il_tl.fillna(0 ,inplace=True)
    X_train_New.num_op_rev_tl.fillna(0 ,inplace=True)
    X_train_New.num_rev_accts.fillna(0 ,inplace=True)
    X_train_New.num_rev_tl_bal_gt_0.fillna(0 ,inplace=True)
    X_train_New.num_sats.fillna(0 ,inplace=True)
    X_train_New.num_tl_120dpd_2m.fillna(0 ,inplace=True)
    X_train_New.num_tl_30dpd.fillna(0 ,inplace=True)
    X_train_New.num_tl_90g_dpd_24m.fillna(0 ,inplace=True)
    X_train_New.num_tl_op_past_12m.fillna(0 ,inplace=True)
    X_train_New.pct_tl_nvr_dlq.fillna(0 ,inplace=True)
    X_train_New.percent_bc_gt_75.fillna(0 ,inplace=True)
    X_train_New.pub_rec_bankruptcies.fillna(0 ,inplace=True)
    X_train_New.tax_liens.fillna(0 ,inplace=True)
    X_train_New.tot_hi_cred_lim.fillna(0 ,inplace=True)
    X_train_New.total_bal_ex_mort.fillna(0 ,inplace=True)
    X_train_New.total_bc_limit.fillna(0 ,inplace=True)
    X_train_New.total_il_high_credit_limit.fillna(0 ,inplace=True)
    # X_train_New.fraction_recovered.fillna(0,inplace=True)





    print('X_train_New = ' ,X_train_New)
    X_train = transformer.fit_transform(X_train)
    X_valid = transformer.transform(X_valid) if validate else None

    X_train_New = transformer.fit_transform(X_train_New)
    # X_train_New2 = transformer.transform(X_train_New)

    print('#######POST')
    print('X_train.shape = ' ,X_train.shape)
    print('X_train.type = ' ,type(X_train))

    print('X_valid.shape = ' ,X_valid.shape)
    print('X_valid.type = ' ,type(X_valid))

    print('onehot_cols=' ,onehot_cols)
    print('list(ordinal_cols.keys())=' ,list(ordinal_cols.keys()))
    print('list(ordinal_cols.values())=' ,list(ordinal_cols.values()))
    print('X_train = ' ,X_train)
    print('#######DONE')

    print('X_train_New.shape = ' ,X_train_New.shape)
    print('X_train_New.type = ' ,type(X_train_New))

    print('X_train_New = ' ,X_train_New)

    X_predict= X_train_New[0].reshape(1, 83)

    return X_predict