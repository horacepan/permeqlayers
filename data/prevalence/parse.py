import numpy as np
import pdb
from tqdm import tqdm
import pandas as pd
import pickle

def ncols(page):
    max_cols = 0
    for _, row in page.iterrows():
        ncols = len(row[~pd.isna(row)])
        max_cols = max(max_cols, ncols)
    return ncols

def get_max_cols(lst):
    return max([ncols(page) for page in tqdm(lst)])

def parse_doses(row):
    '''
    Return dict mapping drug -> doseage
    '''
    def canonical_str(col, doses_dict):
        '''
        Col name: {DRUG}+{DRUG}+...+{DRUG}
        Return: {DRUG}{DOSE}{DRUG}{DOSE}...
        '''
        parts = col.split('+')
        return '-'.join([p + str(doses_dict[p]) for p in parts])

    doses = {}
    s = row[0]
    parts = [s[i:i+4] for i in range(0, len(s), 4)]

    for p in parts:
        doses[p[:3]] = int(p[-1])

    cols = []
    for cstr in row[1:]:
        if type(cstr) == str:
            for c in cstr.split(' '):
                cols.append(canonical_str(c, doses))
        else:
            continue
    assert len(cols) == 31
    return cols

def clean_row(row):
    values = []
    for v in row.values:
        if type(v) == str:
            for vs in v.split(' '):
                try:
                    values.append(float(vs))
                except:
                    if 'TMP' in vs:
                        values.append(float(vs[:6]))
                    else:
                        print(f"Cant split string in clean row at page {pid}, row {i}")
                        pdb.set_trace()
        else:
            if not pd.isna(v):
                values.append(v)
    return np.array(values)

def parse(lst):
    '''
    page: DataFrame of current page
    prev_page: previous page
    '''
    # loop over a page
    cols = None
    index = []
    batched_values = []
    col_vals = []

    for pid, pg in tqdm(enumerate(lst)):
        # need access to last true row of lst
        for i, row in pg.iterrows():
            cleaned_row = row[~pd.isna(row)]
            try:
                if (type(cleaned_row.values[0]) != float) and cleaned_row.values[0][0].isalpha(): # start of a label row
                    x = 22
            except:
                print(f"Cant parse doses at page {pid}, row {i}")
                pdb.set_trace()
            if (type(cleaned_row.values[0]) != float) and cleaned_row.values[0][0].isalpha(): # start of a label row
                # evict an l thing
                if len(col_vals) > 0:
                    index.extend(cols)
                    batch = np.stack(col_vals).T
                    assert batch.shape[0] == 31, "Incorrect batch shape! {}".format(batch.shape)
                    res = np.zeros((batch.shape[0], 4))
                    res[:, :batch.shape[1]] = batch
                    batched_values.append(res)
                    #batched_values.append(batch)
                    col_vals = []

                try:
                    cols = parse_doses(cleaned_row)
                except:
                    print(f"Cant parse doses at page {pid}, row {i}")
                    pdb.set_trace()
            else:
                try:
                    col_vals.append(clean_row(cleaned_row)) #cleaned_row.values.astype(float))
                except:
                    print(f"Cant clean row at page {pid}, row {i}")
                    pdb.set_trace()

    if len(col_vals) > 0:
            index.extend(cols)
            batch = np.stack(col_vals).T
            assert (batch.shape[0] == 31), "Incorrect batch shape!"
            res = np.zeros((batch.shape[0], 4))
            res[:, :batch.shape[1]] = batch
            batched_values.append(res)
            col_vals = []

    values = np.vstack(batched_values)
    df = pd.DataFrame(values, index=index)
    return df


def main():
    f = open('./area_pdf.pkl', 'rb')
    #f = open('./small.pkl', 'rb')
    lst = pickle.load(f)
    parsed_df = parse(lst)
    parsed_df.to_csv('./parsed.csv')
if __name__ == '__main__':
    main()
