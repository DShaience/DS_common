import sys
import warnings
import numpy as np
import pandas as pd


def create_transition_df(s1: pd.Series) -> pd.DataFrame:
    """
    :param s1: gets a pandas series with values
    :return: returns a dataframe containing counts of the transitions between values
    i.e., A->B, B->C, etc.
    """
    s = s1.copy(deep=True).reset_index(drop=True)
    diffs = s.diff()[s.diff() != 0].index.values
    diffs = diffs[1:]

    # return empty dataframe if there are no transitions
    if len(diffs) == 0:
        return pd.DataFrame()
    df_transitions = pd.DataFrame()
    df_transitions['from'] = s[diffs - 1].values
    df_transitions['to'] = s[diffs].values
    return df_transitions


def feature_rescaling_minmax(df: pd.DataFrame, col_name: str):
    # Function applies minmax scaling: x' = (x-min(x))/(max(x)-mix(x))
    # input: data-frame, and target column to rescale (assumes column is numeric)
    # returns: scaled column, cast to float
    x = (pd.to_numeric(df[col_name]) - float(min(df[col_name]))) / (float(max(df[col_name])) - float(min(df[col_name])))
    return x


def feature_standardization(df: pd.DataFrame, col_name: str):
    # Function applies minmax scaling: x' = (x-min(x))/(max(x)-mix(x))
    # input: data-frame, and target column to rescale (assumes column is numeric)
    # returns: standardized column, cast to float
    std = float(df[col_name].std())
    mean = float(df[col_name].mean())
    x = (df[col_name] - mean) / (std)
    return x


def value_pairs_to_df(colname_A, colname_B, colvalues_A, colvalues_B, toabs=False):
    # expects: (array, array, str, str)
    # returns: sorted df by colvalues_B, descending 
    # toabs==True: sort by absolute value 
    # converts two lists, A and B, to dataframe
    df = pd.DataFrame(data=[colvalues_A, colvalues_B]).transpose()
    df.columns = [colname_A, colname_B]

    # df.sort_values(colname_B, ascending=[0], inplace=True)
    # df.reset_index(inplace=True, drop=True)
    if toabs:  # sort by absolute value
        # df = df.reindex(df[colname_B].abs().sort_values(inplace=False, ascending=0).index)
        df = df.reindex(df[colname_B].abs().sort_values(inplace=False, ascending=False).index)
    else:  # sort by value
        df.sort_values(colname_B, ascending=[0], inplace=True)
        df.reset_index(inplace=True, drop=True)

    return df


def train_test_unbalanced_dataset(df, col_class, frac_train_class0, frac_train_class1, frac_test_class0,
                                  frac_test_class1, seed, DEBUG=False):
    # This function samples randomly train and test sets. 
    # The function allows for:
    # * UNBALANCED datasets (according to fractions specified)
    # * Datasets can be only a part of the dataset (frac_train_class0 + frac_train_class0 <= 1.0, 
    # frac_train_class1+frac_test_class1 <= 1.0)
    # This is usedful for example when wanting to train multiple SVMs on a smaller dataset, but still have 
    # enough class1 examples
    # SVM is computaionally expensive O(n^2), so cutting on class0, can be useful
    # df = expects dataframe
    # col_class = the column containing the class (target) variable
    # frac_* = fraction of records to include in train [0-1]. Must be float. 
    #          Summary of same-class fraction must be <= 1
    #          frac_train_class0 + frac_test_class0 <= 1
    #          frac_train_class1 + frac_test_class1 <= 1
    # seed = random seed, to ensure randomness and repeatability of experiments
    # returns: indices for train, and test (both classes), (returns 2 index types)

    if (frac_train_class0 + frac_test_class0 > 1.0):
        sys.exit("Training sample fractions are larger than 1.0. Can't sample more than the complete dataset.")
    elif (frac_train_class1 + frac_test_class1 > 1.0):
        sys.exit("Test sample fractions are larger than 1.0. Can't sample more than the complete dataset.")

    train_class0 = df[df[col_class] == 0].sample(frac=frac_train_class0, replace=False, random_state=seed).index
    train_class1 = df[df[col_class] == 1].sample(frac=frac_train_class1, replace=False, random_state=seed).index

    ntest0 = int(frac_test_class0 * len(df[df[col_class] == 0]))
    ntest1 = int(frac_test_class1 * len(df[df[col_class] == 1]))
    test_class0 = df[df[col_class] == 0].loc[~df[df[col_class] == 0]
        .index.isin(train_class0)].sample(n=ntest0, replace=False,
                                          random_state=seed).index
    test_class1 = df[df[col_class] == 1].loc[~df[df[col_class] == 1]
        .index.isin(train_class1)].sample(n=ntest1, replace=False,
                                          random_state=seed).index

    if DEBUG:
        num_tr0 = float(len(train_class0))
        num_tr1 = float(len(train_class1))
        num_ts0 = float(len(test_class0))
        num_ts1 = float(len(test_class1))
        total_tr = num_tr0 + num_tr1
        total_ts = num_ts0 + num_ts1
        print("TRAINSET:")
        print("\tClass0: %d records (%.2f%% of total records in train)" % (num_tr0, 100 * num_tr0 / total_tr))
        print("\tClass1: %d records (%.2f%% of total records in train)" % (num_tr1, 100 * num_tr1 / total_tr))
        print("\tTotal records: %d" % total_tr)
        print("TESTSET:")
        print("\tClass0: %d records (%.2f%% of total records in test)" % (num_ts0, 100 * num_ts0 / total_ts))
        print("\tClass1: %d records (%.2f%% of total records in test)" % (num_ts1, 100 * num_ts1 / total_ts))
        print("\tTotal: " + str(total_ts))
        print("Total rows used in datasets: " + str(total_tr + total_ts))
        print("Total rows in input file: " + str(len(df)))

        check = list(train_class0) + list(train_class1) + list(test_class0) + list(test_class1)
        if len(check) != len(set(check)):
            warnings.warn(
                "WARNING: Some indices overlap in the train/testsets. Please re-evaluate what you're trying to do")
        else:
            print("Total unique records equal to the number of records (" + str(len(check)) + ")")

    return train_class0.union(train_class1), test_class0.union(test_class1)


def train_test_partial_dataset(df, col_class, frac_train, frac_test, seed, DEBUG=False):
    # This function samples randomly train and test sets, so that both sets have similar percent of each class
    # Diffrence from train_test_dataset() is that this has frac_train and frac_test so that fract_train+frac_test can be <= 1.
    # This is usedful for example when wanting to train multiple SVMs on a smaller dataset, but still maintain the same size 
    # of testset.
    # df = expects dataframe
    # col_class = the column containing the class (target) variable
    # frac_train, frac_train = fraction of records to include in train [0-1]. Must be float, and must add-up to 
    # <= 1.0 (i.e., the complete dataset)
    # seed = random seed, to ensure randomness and repeatability of experiments
    # returns: indices for train, and test (both classes), (returns 2 index types)

    if (frac_train + frac_test > 1):
        sys.exit("Sample fractions are larger than 1.0. Can't sample more than the complete dataset.")

    train_class0 = df[df[col_class] == 0].sample(frac=frac_train, replace=False, random_state=seed).index
    train_class1 = df[df[col_class] == 1].sample(frac=frac_train, replace=False, random_state=seed).index

    # number of records to sample
    ntest0 = int(frac_test * len(df[df[col_class] == 0]))
    ntest1 = int(frac_test * len(df[df[col_class] == 1]))
    test_class0 = df[df[col_class] == 0].loc[~df[df[col_class] == 0]
        .index.isin(train_class0)].sample(n=ntest0, replace=False,
                                          random_state=seed).index
    test_class1 = df[df[col_class] == 1].loc[~df[df[col_class] == 1]
        .index.isin(train_class1)].sample(n=ntest1, replace=False,
                                          random_state=seed).index

    if DEBUG:
        num_tr0 = float(len(train_class0))
        num_tr1 = float(len(train_class1))
        num_ts0 = float(len(test_class0))
        num_ts1 = float(len(test_class1))
        total_tr = num_tr0 + num_tr1
        total_ts = num_ts0 + num_ts1

        print("TRAINSET:")
        print("\tClass0: " + str(100 * num_tr0 / total_tr), )
        print("\tClass1: " + str(100 * num_tr1 / total_tr))
        print("\tTotal: " + str(total_tr))
        print("TESTSET:")
        print("\tClass0: " + str(100 * num_ts0 / total_ts), )
        print("\tClass1: " + str(100 * num_ts1 / total_ts))
        print("\tTotal: " + str(total_ts))
        print("Total rows used in datasets: " + str(total_tr + total_ts))
        print("Total rows in input file: " + str(len(df)))

        check = list(train_class0) + list(train_class1) + list(test_class0) + list(test_class1)
        if len(check) != len(set(check)):
            warnings.warn(
                "WARNING: Some indices overlap in the train/testsets. Please re-evaluate what you're trying to do")

    return train_class0.union(train_class1), test_class0.union(test_class1)


def train_test_dataset_indices(df, col_class, frac_train, seed):
    # This function samples randomly train and test sets, so that both sets have similar percent of each class
    # df = expects dataframe
    # col_class = the column containing the class (target) variable
    # frac_train = fraction of records to include in train [0-1]
    # seed = random seed, to ensure randomness and repeatability of experiments
    # returns: indices for train, and test (both classes), (returns 2 index types)

    train_class0 = df[df[col_class] == 0].sample(frac=frac_train, replace=False, random_state=seed).index
    train_class1 = df[df[col_class] == 1].sample(frac=frac_train, replace=False, random_state=seed).index

    test_class0 = df[df[col_class] == 0].loc[~df[df[col_class] == 0].index.isin(train_class0)].index
    test_class1 = df[df[col_class] == 1].loc[~df[df[col_class] == 1].index.isin(train_class1)].index

    return train_class0.union(train_class1), test_class0.union(test_class1)


def train_test_datasets(df, df_class, frac_train, seed):
    # returns x_train, y_train, x_test, y_test
    train_class0 = df_class[df_class == 0].sample(frac=frac_train, replace=False, random_state=seed).index
    train_class1 = df_class[df_class == 1].sample(frac=frac_train, replace=False, random_state=seed).index

    test_class0 = df_class[df_class == 0].loc[~df_class[df_class == 0].index.isin(train_class0)].index
    test_class1 = df_class[df_class == 1].loc[~df_class[df_class == 1].index.isin(train_class1)].index

    idx_train = train_class0.union(train_class1)
    idx_test = test_class0.union(test_class1)

    return df.iloc[idx_train], df_class.iloc[idx_train], df.iloc[idx_test], df_class.iloc[idx_test]


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def mutual_information(y, x):
    res = entropy(y)
    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])
    return res


def dataset_classes(df, col_class):
    classes = list(set(df[col_class]))
    records_total = len(df)
    print("The dataset contains " + str(records_total))
    for i in classes:
        records_class = len(df[df[col_class] == i])
        records_ratio = (float(records_class) / float(records_total)) * 100
        print("\tClass " + str(i) + " has " + str(records_class) + " (" + str(records_ratio) + "%)")


def pct_rank_qcut(series, n):
    # qcut variant to avoid the "unique-edges" error
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)


def dummify_variables(df):
    """Create dummy variables for all columns in df (assumes all are categorical)"""
    """Returns df of dummy variables"""
    cols = df.columns.values
    print("Number of columns before dummy-variables:\t%s" % (len(cols)))
    for col in cols:
        dummy_ranks = pd.get_dummies(df[col], prefix=col)
        df = df.join(dummy_ranks)
        # dropping the original categoric column
        df = df.drop(col, 1)

    cols = df.columns.values
    print("Number of columns after dummy-variables:\t%s" % (len(cols)))
    return df


def stripRN(unicodeTxt: str):
    '''
    :param unicodeTxt: a unicode string
    :return: the same string, without special characters, such as \n \r, etc.
    '''

    newStr = unicodeTxt.replace(u'\n', '')
    newStr = newStr.replace(u'\r', '')

    return newStr


def sanitizeText(txt: str) -> str:
    '''
    :param txt: any text
    :return: offers basic text sanitization against special characters and strips trailing/heading whitespace
    '''
    if (txt is None) or (txt is np.nan) or (txt == ""):
        return txt
    removeSpecialChars = txt.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+'\""})
    removeSpecialChars = removeSpecialChars.replace('rlm', '')
    removeSpecialChars = removeSpecialChars.strip()
    return removeSpecialChars

