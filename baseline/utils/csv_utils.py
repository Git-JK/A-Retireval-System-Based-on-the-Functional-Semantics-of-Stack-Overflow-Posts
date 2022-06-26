import pandas as pd
import csv

def write_list_to_csv(list_tmp, csv_fpath, header):
    df = pd.DataFrame(data=list_tmp, columns=header)
    df.to_csv(csv_fpath, index=False)
    # with open(csv_fpath, 'w') as myfile:
    #     wr = csv.writer(myfile)
    #     wr.writerow(header)
    #     for x in list_tmp:
    #         try:
    #             print(type(x))
    #             wr.writerow(x)
    #         except Exception as e:
    #             print("Error %s" % e)
    print("Write %s successfully!" % csv_fpath)
