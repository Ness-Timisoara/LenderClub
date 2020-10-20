import pandas as pd
import argparse
import os
import pickle
import numpy as np

def read_csv(data_dir: str,loadmode: str,output_dir:str):
    try:
        if loadmode == 'train':
            pkl_file = output_dir+'train_data.pickle'
        elif loadmode == 'predict':
            pkl_file = output_dir+'predict_data.pickle'
        else:
            pkl_file = output_dir+'test_data.pickle'
        loans = pd.read_csv(data_dir,low_memory=False,)

        print('data loaded')
        print('data shape===>',loans.shape)
        print('PKL FILE===>', pkl_file)
        try:
            pickle.dump(loans, open(pkl_file, "wb"))  # save it into a file named save.p
            print('data loaded and Saved')
            clean_data(pkl_file,loadmode,output_dir)
        except:
            print('NOT SAVED --- CANNOT PROGRESS')
            return 1


        return 0
    except:
        return 1
def clean_data(pfile:str,loadmode: str,output_dir:str):
    dictionary_df = pd.read_excel("data/LCDataDictionary.xlsx")
    dictionary_df.dropna(axis="index", inplace=True)
    dictionary_df = dictionary_df.applymap(lambda x: x.strip())
    dictionary_df.set_index("LoanStatNew", inplace=True)
    dictionary = dictionary_df["Description"].to_dict()
    dictionary["verification_status_joint"] = dictionary.pop("verified_status_joint")

    print('1')

    cols_for_output = ["term", "installment", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries","collection_recovery_fee"]
    cols_to_drop = ["id", "member_id", "funded_amnt", "funded_amnt_inv", "int_rate", "grade", "sub_grade", "emp_title",
                    "pymnt_plan", "url", "desc", "title", "zip_code", "addr_state", "initial_list_status", "out_prncp",
                    "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "last_pymnt_d", "last_pymnt_amnt",
                    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "policy_code",
                    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "deferral_term",
                    "hardship_amount", "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
                    "hardship_length", "hardship_dpd", "hardship_loan_status",
                    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
                    "hardship_last_payment_amount", "disbursement_method", "debt_settlement_flag",
                    "debt_settlement_flag_date", "settlement_status", "settlement_date", "settlement_amount",
                    "settlement_percentage", "settlement_term"]


    file = open(pfile, 'rb')
    loans = pickle.load(file)
    file.close()

    for col in loans.columns:
        print(f"•{col}: {dictionary[col]}")
    loans = loans.drop(columns=cols_to_drop)
    loans.groupby("loan_status")["loan_status"].count()

    credit_policy = "Does not meet the credit policy. Status:"
    len_credit_policy = len(credit_policy)
    remove_credit_policy = (
        lambda status: status[len_credit_policy:]
        if credit_policy in str(status)
        else status
    )
    loans["loan_status"] = loans["loan_status"].map(remove_credit_policy)

    rows_to_drop = loans[
        (loans["loan_status"] != "Charged Off") & (loans["loan_status"] != "Fully Paid")
        ].index
    loans.drop(index=rows_to_drop, inplace=True)

    loans.groupby("loan_status")["loan_status"].count()

    loans[cols_for_output].info()
    loans.groupby("term")["term"].count()

    onehot_cols = ["term"]

    loans["term"] = loans["term"].map(lambda term_str: term_str.strip())

    extract_num = lambda term_str: float(term_str[:2])
    loans["term_num"] = loans["term"].map(extract_num)
    cols_for_output.remove("term")
    cols_for_output.append("term_num")

    received = (
        loans["total_rec_prncp"]
        + loans["total_rec_int"]
        + loans["total_rec_late_fee"]
        + loans["recoveries"]
        - loans["collection_recovery_fee"]
    )
    expected = loans["installment"] * loans["term_num"]
    loans["fraction_recovered"] = received / expected

    loans.groupby("loan_status")["fraction_recovered"].describe()



    loans["fraction_recovered"] = np.where(
        (loans["loan_status"] == "Fully Paid") | (loans["fraction_recovered"] > 1.0),
        1.0,
        loans["fraction_recovered"],
    )
    loans.groupby("loan_status")["fraction_recovered"].describe()

    loans.drop(columns=cols_for_output, inplace=True)
    loans.info(verbose=True, null_counts=True)

    negative_mark_cols = ["mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
                          "mths_since_recent_bc_dlq", "mths_since_recent_inq", "mths_since_recent_revol_delinq",
                          "mths_since_recent_revol_delinq", "sec_app_mths_since_last_major_derog"]
    joint_cols = ["annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
                  "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
                  "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
                  "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
                  "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog"]
    confusing_cols = ["open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
                      "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl",
                      "inq_last_12m"]


    loans["issue_d"] = loans["issue_d"].astype("datetime64[ns]")
    loans[confusing_cols + ["issue_d"]].dropna(axis="index")["issue_d"].agg(["count", "min", "max"])

    # Compare to all entries from Dec 2015 onward
    loans["issue_d"][loans["issue_d"] >= np.datetime64("2015-12-01")].agg(["count", "min", "max"])

    new_metric_cols = confusing_cols

    mths_since_last_cols = [
        col_name
        for col_name in loans.columns
        if "mths_since" in col_name or "mo_sin_rcnt" in col_name
    ]
    mths_since_old_cols = [
        col_name for col_name in loans.columns if "mo_sin_old" in col_name
    ]

    for col_name in mths_since_last_cols:
        loans[col_name] = [
            0.0 if pd.isna(months) else 1 / 1 if months == 0 else 1 / months
            for months in loans[col_name]
        ]
    loans.loc[:, mths_since_old_cols].fillna(0, inplace=True)

    # Rename inverse columns
    rename_mapper = {}
    for col_name in mths_since_last_cols:
        rename_mapper[col_name] = col_name.replace("mths_since", "inv_mths_since").replace(
            "mo_sin_rcnt", "inv_mo_sin_rcnt"
        )
    loans.rename(columns=rename_mapper, inplace=True)

    def replace_list_value(l, old_value, new_value):
        i = l.index(old_value)
        l.pop(i)
        l.insert(i, new_value)

    replace_list_value(new_metric_cols, "mths_since_rcnt_il", "inv_mths_since_rcnt_il")
    replace_list_value(
        joint_cols,
        "sec_app_mths_since_last_major_derog",
        "sec_app_inv_mths_since_last_major_derog",
    )

    loans.groupby("application_type")["application_type"].count()

    joint_loans = loans[:][loans["application_type"] == "Joint App"]
    joint_loans[joint_cols].info()

    joint_new_metric_cols = ["revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
                             "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc",
                             "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
                             "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
                             "sec_app_inv_mths_since_last_major_derog"]
    joint_loans[joint_new_metric_cols + ["issue_d"]].dropna(axis="index")["issue_d"].agg(
        ["count", "min", "max"]
    )

    # Check without `sec_app_revol_util` column
    joint_new_metric_cols_2 = ["revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
                               "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc",
                               "sec_app_open_acc", "sec_app_open_act_il", "sec_app_num_rev_accts",
                               "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
                               "sec_app_inv_mths_since_last_major_derog"]
    joint_loans[joint_new_metric_cols_2 + ["issue_d"]].dropna(axis="index")["issue_d"].agg(
        ["count", "min", "max"]
    )

    joint_loans["issue_d"].agg(["count", "min", "max"])

    onehot_cols.append("application_type")

    # Fill joint columns in individual applications
    for joint_col, indiv_col in zip(
            ["annual_inc_joint", "dti_joint", "verification_status_joint"],
            ["annual_inc", "dti", "verification_status"],
    ):
        loans[joint_col] = [
            joint_val if app_type == "Joint App" else indiv_val
            for app_type, joint_val, indiv_val in zip(
                loans["application_type"], loans[joint_col], loans[indiv_col]
            )
        ]

    loans.info(verbose=True, null_counts=True)



    cols_to_search = [col for col in loans.columns if col not in new_metric_cols + joint_new_metric_cols]
    loans.dropna(axis="index", subset=cols_to_search, inplace=True)


    loans[["earliest_cr_line", "sec_app_earliest_cr_line"]]
    cr_hist_age_months = get_credit_history_age(loans,"earliest_cr_line")

    loans["earliest_cr_line"] = cr_hist_age_months
    loans["sec_app_earliest_cr_line"] = get_credit_history_age(loans,"sec_app_earliest_cr_line").astype("Int64")
    loans.rename(columns={"earliest_cr_line": "cr_hist_age_mths","sec_app_earliest_cr_line": "sec_app_cr_hist_age_mths",},inplace=True,)



    replace_list_value(joint_new_metric_cols, "sec_app_earliest_cr_line", "sec_app_cr_hist_age_mths")

    categorical_cols = ["term", "emp_length", "home_ownership", "verification_status", "purpose",
                        "verification_status_joint"]
    for i, col_name in enumerate(categorical_cols):
        print(loans.groupby(col_name)[col_name].count(),"\n" if i < len(categorical_cols) - 1 else "",)

    loans.drop(columns=["verification_status","verification_status_joint","issue_d","loan_status",],inplace=True,)

    onehot_cols += ["home_ownership", "purpose"]
    ordinal_cols = {"emp_length": ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years",]}

    loans_1 = loans.drop(columns=new_metric_cols + joint_new_metric_cols)
    loans_2 = loans.drop(columns=joint_new_metric_cols)
    loans_2.info(verbose=True, null_counts=True)

    loans_2["il_util"][loans_2["il_util"].notna()].describe()

    query_df = loans[["il_util", "total_bal_il", "total_il_high_credit_limit"]].dropna(axis="index", subset=["il_util"])
    query_df["il_util_compute"] = (query_df["total_bal_il"] / query_df["total_il_high_credit_limit"]).map(lambda x: float(round(x * 100)))
    # query_df[["il_util", "il_util_compute"]]
    # (query_df["il_util"] == query_df["il_util_compute"]).describe()
    # query_df["compute_diff"] = abs(query_df["il_util"] - query_df["il_util_compute"])
    # query_df["compute_diff"][query_df["compute_diff"] != 0].describe()

    loans["il_util_imputed"] = [
        True if pd.isna(util) & pd.notna(bal) & pd.notna(limit) else False
        for util, bal, limit in zip(
            loans["il_util"], loans["total_bal_il"], loans["total_il_high_credit_limit"]
        )
    ]
    new_metric_onehot_cols = ["il_util_imputed"]
    loans["il_util"] = [
        0.0
        if pd.isna(util) & pd.notna(bal) & (limit == 0)
        else float(round(bal / limit * 100))
        if pd.isna(util) & pd.notna(bal) & pd.notna(limit)
        else util
        for util, bal, limit in zip(
            loans["il_util"], loans["total_bal_il"], loans["total_il_high_credit_limit"]
        )
    ]

    loans_2 = loans.drop(columns=joint_new_metric_cols)
    loans_2.info(verbose=True, null_counts=True)

    loans_2.dropna(axis="index", inplace=True)
    loans_3 = loans.dropna(axis="index")
    loans_3.info(verbose=True, null_counts=True)

    print("Model 1:")
    print("loans_1--->", loans_1)
    print("onehot_cols--->", onehot_cols)
    print("ordinal_cols--->", ordinal_cols)

    print("\nModel 2:")
    print("loans_2 --->", loans_2)
    print("onehot_cols + new_metric_onehot_cols--->", onehot_cols + new_metric_onehot_cols)

    print("\nModel 3:")
    print("loans_3--->", loans_3)


    if loadmode == 'train':
        pkl_file_loan = output_dir+'train_load_1.pickle'
        pkl_file_onehot_cols = output_dir+'train_onehot_cols.pickle'
        pkl_file_ordinal_cols = output_dir+'train_ordinal_cols.pickle'
    elif loadmode == 'predict':
        pkl_file_loan = output_dir+'predict_load_1.pickle'
        pkl_file_onehot_cols = output_dir+'predict_onehot_cols.pickle'
        pkl_file_ordinal_cols = output_dir+'predict_ordinal_cols.pickle'
    else:
        pkl_file_loan = output_dir+'test_load_1.pickle'
        pkl_file_onehot_cols = output_dir+'test_onehot_cols.pickle'
        pkl_file_ordinal_cols = output_dir+'test_ordinal_cols.pickle'
    try:
        pickle.dump(loans_1, open(pkl_file_loan, "wb"))
        pickle.dump(onehot_cols, open(pkl_file_onehot_cols, "wb"))
        pickle.dump(ordinal_cols, open(pkl_file_ordinal_cols, "wb"))
        print('data for training Saved')
    except:
        print('data for training NOT SAVED --- CANNOT PROGRESS')
        return 1

    print('PTL')
    return 0






def get_credit_history_age(loans,col_name):
    earliest_cr_line_date = loans[col_name].astype("datetime64[ns]")
    cr_hist_age_delta = loans["issue_d"] - earliest_cr_line_date
    MINUTES_PER_MONTH = int(365.25 / 12 * 24 * 60)
    cr_hist_age_months = cr_hist_age_delta / np.timedelta64(MINUTES_PER_MONTH, "m")
    return cr_hist_age_months.map(
        lambda value: np.nan if pd.isna(value) else round(value)
    )


def add_trail_slash(s):
    # if not s.startswith('/'):
    #     s = '/'+s
    if not s.endswith('/'):
        s = s+'/'
    return s

def remove_trail_slash(s):
    # if not s.startswith('/'):
    #     s = '/'+s
    if s.endswith('/'):
        s = s[:-1]
    return s



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow training script')
    parser.add_argument('--data_dir',help='path to csv and labels.')
    parser.add_argument('--loadmode', help='train predict test')
    parser.add_argument('--output_dir', help='path to output preprocessed pickled files and labels.')
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.output_dir is None:
        output_dir='data/'

    if not os.path.exists(remove_trail_slash(output_dir)):
        os.makedirs(remove_trail_slash(output_dir))

    output_dir=add_trail_slash(output_dir)
    print('output_dir=',output_dir)
    print('---------->>>',args.loadmode)
    if args.loadmode != 'noload':
        print('calling load')
        read_csv(data_dir=args.data_dir,loadmode=args.loadmode,output_dir=output_dir)
        print('load called')
    if args.loadmode == 'noload':
        clean_data(args.loadmode)
    print('Data Loaded')
