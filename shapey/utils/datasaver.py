import csv


def check_and_retrieve_stored_feature(imgname, imgname_list, feat_list):
    try:
        idx = imgname_list.index(imgname)
        feat = feat_list[idx]
    except ValueError:
        feat = "Not Found"

    return feat


def write_row_to_hdf(df, hdfstore, data_key):
    hdfstore.append(data_key, df, format="table")


def write_row_to_csv(row, csvfile):
    with open(csvfile, "a") as f:
        wrtr = csv.writer(f, delimiter=",", quotechar='"')
        wrtr.writerow(row)


def get_row_count(csvfile):
    with open(csvfile, "r") as f:
        reader = csv.reader(f)
        row_count = sum(1 for r in reader)

    return row_count
