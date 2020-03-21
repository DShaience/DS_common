import itertools
from typing import List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def correlationMatrix(df: pd.DataFrame, fontsz: int = 16, toShow: bool = True):
    """
    :param df: input dataframe. Correlation matrix calculated for all columns
    :param fontsz: font size
    :param toShow: True - plots the figure
    :return:
    """
    # Correlation between numeric variables
    cols_numeric = list(df)
    data_numeric = df[cols_numeric].copy(deep=True)
    corr_mat = data_numeric.corr(method='pearson')
    cbar_ticks = [round(num, 1) for num in np.linspace(-1, 1, 11, dtype=np.float)]  # rounding corrects for floating point imprecision

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=[8, 8])
    plt.xticks(fontsize=fontsz)
    plt.yticks(fontsize=fontsz)
    ax = sns.heatmap(corr_mat, cmap=cmap, vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticks)
    if toShow:
        plt.show()


def plot_confusion_matrix(cm: np.array, label_names: List[str], title: str='Confusion matrix', cmaptype=0):
    '''
    :param cm:
    :param label_names:
    :param title:
    :param cmaptype:
    :return: nicely plots confusion matrices
    '''
    # Pretty plotting for confusion matrices
    if (cmaptype == 0):
        cmap = plt.cm.Purples
    elif (cmaptype == 1):
        cmap = plt.cm.Greens
    else:
        cmap = plt.cm.Reds

    fontsz = 12
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsz + 4)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, fontsize=fontsz + 2)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks()).astype(str))
    ax.grid(False)
    plt.yticks(tick_marks, fontsize=fontsz + 2)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], weight='bold',
                 horizontalalignment="center", fontsize=fontsz + 2,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=fontsz + 2)
    plt.xlabel('Predicted label', fontsize=fontsz + 2)


def plotKDE_1d(ds: pd.Series, title: str, fontsz: int = 14):
    '''
    :param ds: data series
    :param fontsz:
    :return: plots kdeplot for  one series
    '''
    plt.figure()
    plt.xticks(fontsize=fontsz)
    plt.yticks(fontsize=fontsz)
    plt.title(title, fontsize=fontsz)
    x = ds.dropna()
    sns.kdeplot(x, shade=True, label=title, cut=0)
    plt.legend(fontsize=fontsz)
    plt.show()


def scatterPlot2d(df: pd.DataFrame, featureName1: str, featureName2: str):
    ax = sns.scatterplot(x=featureName1, y=featureName2, data=df)
    plt.show()