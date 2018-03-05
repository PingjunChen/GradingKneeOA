# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import pandas as pd
import collections


# Check ID consistency
def check_id_consistency():
    Enrollees = pd.read_csv('../data/Clinicals/Enrollees_SAS/Enrollees.txt', sep='|')
    ids_en = Enrollees['ID'].tolist()
    AllClinical00 = pd.read_csv('../data/Clinicals/AllClinical00_SAS/AllClinical00.txt', sep='|')
    ids_00 = AllClinical00['ID'].tolist()
    AllClinical03 = pd.read_csv('../data/Clinicals/AllClinical03_SAS/AllClinical03.txt', sep='|')
    ids_03 = AllClinical03['ID'].tolist()

    flag = True
    if len(ids_en) != len(ids_00) or len(ids_en) != len(ids_03):
        flag = False
        return flag

    for ind in range(len(ids_en)):
        if ids_en[ind] != ids_00[ind] or ids_en[ind] != ids_03[ind]:
            flag = False
            return flag

    return True


def save_clinical_info():
    Enrollees = pd.read_csv('../data/Clinicals/Enrollees_SAS/Enrollees.txt', sep='|')
    AllClinical00 = pd.read_csv('../data/Clinicals/AllClinical00_SAS/AllClinical00.txt', sep='|')
    AllClinical03 = pd.read_csv('../data/Clinicals/AllClinical03_SAS/AllClinical03.txt', sep='|')

    info_dict = collections.OrderedDict()

    # Info from Enrollees
    info_dict["ID"] = Enrollees['ID'].tolist()
    any_id_nan = np.isnan(info_dict["ID"]).any()
    if any_id_nan:
        print("{} NAN exist in ID.".format(np.sum(np.isnan(info_dict["ID"]))))
    info_dict["Gender"] = Enrollees['P02SEX'].tolist()
    any_gender_nan = np.isnan(info_dict["Gender"]).any()
    if any_gender_nan:
        print("{} NAN exist in Gender".format(np.sum(np.isnan(info_dict["Gender"]))))

    # Info from AllClinical00
    info_dict["Age"] = AllClinical00['V00AGE'].tolist()
    any_age_nan = np.isnan(info_dict["Age"]).any()
    if any_age_nan:
        print("{} NAN exist in Age.".format(np.sum(np.isnan(info_dict["Age"]))))
    info_dict["BMI"] = AllClinical00['P01BMI'].tolist()
    any_bmi_nan = np.isnan(info_dict["BMI"]).any()
    if any_bmi_nan:
        print("{} NAN exist in BMI.".format(np.sum(np.isnan(info_dict["BMI"]))))


    # info_dict["KLL00"] = AllClinical00['V00XRKLL'].tolist()
    # any_kll00_nan = np.isnan(info_dict["KLL00"]).any()
    # if any_kll00_nan:
    #     print("{} NAN exist in KLL00".format(np.sum(np.isnan(info_dict["KLL00"]))))

    # info_dict["KLR00"] = AllClinical00['V00XRKLR'].tolist()
    # any_klr00_nan = np.isnan(info_dict["KLR00"]).any()
    # if any_klr00_nan:
    #     print("{} NAN exist in KLR00".format(np.sum(np.isnan(info_dict["KLR00"]))))

    # info_dict["WOMKPL00"] = AllClinical00['V00WOMKPL'].tolist()
    # any_womkpl00_nan = np.isnan(info_dict["WOMKPL00"]).any()
    # if any_womkpl00_nan:
    #     print("{} NAN exist in WOMKPL00".format(np.sum(np.isnan(info_dict["WOMKPL00"]))))
    # info_dict["WOMKPR00"] = AllClinical00['V00WOMKPR'].tolist()
    # any_womkpr00_nan = np.isnan(info_dict["WOMKPR00"]).any()
    # if any_womkpr00_nan:
    #     print("{} NAN exist in WOMKPR00".format(np.sum(np.isnan(info_dict["WOMKPR00"]))))

    # # Info from AllClinical03
    # info_dict["WOMKPL03"] = AllClinical03['V03WOMKPL'].tolist()
    # any_womkpl03_nan = np.isnan(info_dict["WOMKPL03"]).any()
    # if any_womkpl03_nan:
    #     print("{} NAN exist in WOMKPL03".format(np.sum(np.isnan(info_dict["WOMKPL03"]))))
    # info_dict["WOMKPR03"] = AllClinical03['V03WOMKPR'].tolist()
    # any_womkpr03_nan = np.isnan(info_dict["WOMKPR03"]).any()
    # if any_womkpr03_nan:
    #     print("{} NAN exist in WOMKPR03".format(np.sum(np.isnan(info_dict["WOMKPR03"]))))


    info_filename = "clinical_summary.xlsx"
    info_df = pd.DataFrame(info_dict)
    info_df = info_df.dropna(axis=0, how='any')
    info_writer = pd.ExcelWriter(info_filename, engine='xlsxwriter')
    info_df.to_excel(info_writer, sheet_name='Sheet1', index=False)
    info_writer.save()


if __name__ == '__main__':
    save_clinical_info()
