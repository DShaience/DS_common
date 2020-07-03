import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import List, Union
from scipy.stats import ks_2samp, gaussian_kde
sns.set(color_codes=True)


def plotMatrix(mat: Union[pd.DataFrame, np.ndarray], fontsz: int, cbar_ticks: List[float] = None):
    """
    :param mat: matrix to plot. If using dataframe, the columns are automatically used as labels. Othereise, matrix is anonymous
    :param fontsz: font size
    :param cbar_ticks: the spacing between cbar ticks. If None, this is set automatically.
    :return:
    """
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=[8, 8])
    if cbar_ticks is not None:
        ax = sns.heatmap(mat, cmap=cmap, vmin=min(cbar_ticks), vmax=max(cbar_ticks), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
    else:
        ax = sns.heatmap(mat, cmap=cmap, vmin=np.min(np.array(mat).ravel()), vmax=np.max(np.array(mat).ravel()), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsz)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsz)
    plt.show()


def correlationMatrix(df: pd.DataFrame, fontsz: int = 16, corrThr: float = None):
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
    if corrThr is not None:
        assert corr_mat > 0.0, "corrThr must be a float between [0, 1]"
        corr_mat[corr_mat >= corrThr] = 1.0
        corr_mat[corr_mat <= -corrThr] = -1.0

    cbar_ticks = [round(num, 1) for num in np.linspace(-1, 1, 11, dtype=np.float)]  # rounding corrects for floating point imprecision
    plotMatrix(corr_mat, fontsz=fontsz, cbar_ticks=cbar_ticks)
    # def plotMatrix(mat: np.ndarray, fontsz: int, cbar_ticks: np.ndarray = None):


    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # plt.figure(figsize=[8, 8])
    # plt.xticks(fontsize=fontsz)
    # plt.yticks(fontsize=fontsz)
    # ax = sns.heatmap(corr_mat, cmap=cmap, vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks(cbar_ticks)
    # cbar.set_ticklabels(cbar_ticks)
    # plt.show()


def plotConfusionMatrix(cm: np.array, label_names: List[str], title: str='Confusion matrix', cmaptype=0):
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


def plotFeaturesCoefficients(featuresCoeffsDf: pd.DataFrame, fontSize: int=10):
    """
    :param featuresCoeffsDf: dataframe of features and their coefficients
    :param fontSize:
    :return:
    """
    featuresCoeffs = featuresCoeffsDf.apply(pd.to_numeric)
    featuresCoeffs.reset_index(drop=True, inplace=True)
    featuresCoeffs = featuresCoeffs.transpose()
    asArray = np.array(featuresCoeffs)
    minVal = np.min(asArray)
    maxVal = np.max(asArray)
    fromVal = round(minVal - 0.005, 2)
    toVal = round(maxVal + 0.005, 2)
    cbar_ticks = np.linspace(fromVal, toVal, 11, dtype=np.float16)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=[12, 12])
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    ax = sns.heatmap(featuresCoeffs, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticks)
    plt.show()


def calcShowTwoDfHistCDF(df1: pd.DataFrame, df2: pd.DataFrame, df1Label: str, df2Label: str, featureCol: str, density: bool = True):
    fontsz = 14
    df1Color = '#33CC33'  # green
    df2Color = '#bc0033'  # red
    alpha = 0.5

    featureData1 = df1.loc[~df1[featureCol].isna(), featureCol]
    featureData2 = df2.loc[~df2[featureCol].isna(), featureCol]
    if len(featureData1) == 0 or len(featureData2) == 0:
        print("One of the dataframes is empty. Lengths:\t%s, %s" % (len(featureData1), len(featureData2)))
        return

    bins = 100
    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.xaxis.set_tick_params(labelsize=fontsz)
    ax.yaxis.set_tick_params(labelsize=fontsz)

    try:
        D, p_value = ks_2samp(featureData1, featureData2)
    except Exception as e:
        print("ERROR: Exception while trying to calculate Kolmogorov Smirnov test over the two samples. Aborting with exception")
        raise ValueError(e)

    ax.set_title(featureCol + " (normalized)", fontsize=fontsz)
    counts_df1, bin_edges_df1, _ = ax.hist(featureData1, bins=bins, density=density, color=df1Color, alpha=alpha, label=df1Label)
    counts_df2, bin_edges_df2, _ = ax.hist(featureData2, bins=bins, density=density, color=df2Color, alpha=alpha, label=df2Label)
    ax.plot([], [], ' ', label="\nKS D: " + str(round(D, 4)))

    ax.legend(fontsize=fontsz)

    samples = featureData1
    xmax, xmin = max(samples), min(samples)
    positions = np.linspace(xmin, xmax, 1000)
    gaussian_kernel = gaussian_kde(samples)
    values = gaussian_kernel.evaluate(positions)
    ax.plot(positions, values, 'b')

    samples = featureData2
    xmax, xmin = max(samples), min(samples)
    positions = np.linspace(xmin, xmax, 1000)
    gaussian_kernel = gaussian_kde(samples)
    values = gaussian_kernel.evaluate(positions)
    ax.plot(positions, values, 'k')

    ax2.xaxis.set_tick_params(labelsize=fontsz)
    ax2.yaxis.set_tick_params(labelsize=fontsz)
    ax2.set_title(featureCol + " CDF ", fontsize=fontsz)
    cdf1 = np.cumsum(counts_df1)
    cdf2 = np.cumsum(counts_df2)

    ax2.plot(bin_edges_df1[1:], cdf1 / cdf1[-1], label=df1Label, color=df1Color)
    ax2.plot(bin_edges_df2[1:], cdf2 / cdf2[-1], label=df2Label, color=df2Color)
    ax2.plot([], [], ' ', label="\nKS D: " + str(round(D, 4)))
    ax2.legend(fontsize=fontsz)

    print("Feature: %s\t K-S D: %s" % (featureCol, str(round(D, 4))))
    plt.show()


