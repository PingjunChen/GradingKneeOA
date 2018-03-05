import os, sys, pdb
import pandas as pd
import json

def filter_kl_info(summary_kl_file):
    dfs = pd.read_excel(summary_kl_file, sheet_name="Sheet1")
    # Remove row with kl grade of nan value
    dfs = dfs.dropna(axis=0, how='any')
    # Only keep project 15 for KL grade evaluation
    dfs = dfs.loc[dfs['READPRJ'] == 15]
    dfs = dfs.drop('READPRJ', 1)

    # Save to filtered excels
    filename = os.path.basename(summary_kl_file)
    filtered_name = os.path.splitext(filename)[0] + "_filtered.xlsx"
    filtered_path = os.path.join(os.path.dirname(summary_kl_file), filtered_name)

    writer = pd.ExcelWriter(filtered_path, engine='xlsxwriter')
    dfs.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()

    # Save info to json
    kl_dict = {}
    for index, row in dfs.iterrows():
        pid = str(int(row["ID"]))
        side = str(int(row["SIDE"]))
        if side == "1":
            key = pid + "R"
        else:
            key = pid + "L"
        klg = float(row["KLG"])
        kl_dict[key] = klg
    kl_json_name = os.path.splitext(filename)[0] + "_filtered.json"
    kl_json_path = os.path.join(os.path.dirname(summary_kl_file), kl_json_name)
    with open(kl_json_path, 'w') as outfile:
        json.dump(kl_dict, outfile)



if __name__ == '__main__':
    # kl00_info_summary = "../data/organized/clinical/KL00_summary.xlsx"
    # filter_kl_info(kl00_info_summary)

    kl03_info_summary = "../data/organized/clinical/KL03_summary.xlsx"
    filter_kl_info(kl03_info_summary)
