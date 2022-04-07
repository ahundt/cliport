import pickle
import numpy as np
import os
import pandas as pd
from optparse import OptionParser
import scipy.stats as st
import scipy.spatial
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics import utils
from utils.command_strings import build_command_strings
import csv
import statsmodels.api as sm
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.graphics import utils
from statsmodels.compat.python import lzip, lrange
from tqdm import tqdm, trange
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import seaborn as sns

ordered_ids=['AF', 'AM', 'BF', 'BM', 'LF', 'LM', 'WF', 'WM', 'A', 'B', 'L', 'W', 'F', 'M']

def ols_test(X, y):
    X_constant = sm.add_constant(X)
    lin_reg = sm.OLS(y,X_constant).fit()

    bp_test_result_list_names = ['bptest Lagrange multiplier statistic', 'bptest lm p-value',
        'bptest f-value', 'bptest f p-value']
    bp_test_result_list = sms.het_breuschpagan(lin_reg.resid, lin_reg.model.exog)

    return lin_reg.pvalues, bp_test_result_list, bp_test_result_list_names


# def breusch_pagan_test(x, y):
#     '''
#     Breusch-Pagan test for heteroskedasticity in a linear regression model:
#     H_0 = No heteroskedasticity.
#     H_1 = Heteroskedasticity is present.
#
#     Inputs:
#     x = a numpy.ndarray containing the predictor variables. Shape = (nSamples, nPredictors).
#     y = a 1D numpy.ndarray containing the response variable. Shape = (nSamples, ).
#
#     Outputs a list containing three elements:
#     1. the Breusch-Pagan test statistic.
#     2. the p-value for the test.
#     3. the test result.
#     '''
#
#     if y.ndim != 1:
#         raise SystemExit('Error: y has more than 1 dimension.')
#     if x.shape[0] != y.shape[0]:
#         raise SystemExit('Error: the number of samples differs between x and y.')
#     else:
#         n_samples = y.shape[0]
#
#     # fit an OLS linear model to y using x:
#     lm = LinearRegression()
#     lm.fit(x, y)
#
#     # calculate the squared errors:
#     err = (y - lm.predict(x))**2
#
#     # fit an auxiliary regression to the squared errors:
#     # why?: to estimate the variance in err explained by x
#     lm.fit(x, err)
#     pred_err = lm.predict(x)
#     del lm
#
#     # calculate the coefficient of determination:
#     ss_tot = sum((err - np.mean(err))**2)
#     ss_res = sum((err - pred_err)**2)
#     r2 = 1 - (ss_res / ss_tot)
#     del err, pred_err, ss_res, ss_tot
#
#     # calculate the Lagrange multiplier:
#     LM = n_samples * r2
#     del r2
#
#     # calculate p-value. degrees of freedom = number of predictors.
#     # this is equivalent to (p - 1) parameter restrictions in Wikipedia entry.
#     pval = stats.distributions.chi2.sf(LM, x.shape[1])
#
#     if pval < 0.01:
#         test_result = 'Heteroskedasticity present at 99% CI.'
#     elif pval < 0.05:
#         test_result = 'Heteroskedasticity present at 95% CI.'
#     else:
#         test_result = 'No significant heteroskedasticity.'
#     return [LM, pval, test_result]

def tukey_test(data, save_path, title):
    ''' Run pairwise Tukey test to determine p-values for differences between data means.

    args:
        data: dict {identity: list of samples}
        save_path: str, where to save csv of results
        title: str, name of csv
    '''


    identities=[]
    per_data_identities=[]
    datas=[]
    for id in ordered_ids:
        if id in data:
            identities.append(id)
            for _ in range(data[id].shape[0]):
                per_data_identities.append(id)
            datas.append(data[id])

    try:
        one_hot_ids=sklearn.preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(per_data_identities).reshape(-1, 1))
    except:
        print("No data to tukay test")
        return
    y=np.concatenate(datas)
    ols_pvalues, bp_test_result_list, bp_test_result_list_names = ols_test(one_hot_ids, y)
    #LM, bp_pval, test_result=breusch_pagan_test(one_hot_ids, y)

    title_string = title.replace('_', ' ')
    # Can't do multiple comparison tests with only one group

    file_path = os.path.join(save_path, title)
    anova_oneway=f_oneway(*datas)
    anova_oneway_df = pd.DataFrame(data=anova_oneway, index=['F statistic', 'p-value'], columns=[title])
    anova_oneway_df.to_csv(file_path + '_anova_f_oneway.csv')

    # perform Tukey's test
    flat_datas=np.concatenate(datas)
    try:
        tukey = pairwise_tukeyhsd(endog=flat_datas,
                                  groups=per_data_identities,
                                  alpha=0.1)
        tukey._simultaneous_ci()
    except ValueError:
        print('Warning: Skipping tukey test that caused a ValueError. title: ' + title + ' with indended save path: ' + save_path)
        return
    # u=print(tukey)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tukey_df.to_csv(file_path + ".csv")
    #fig = tukey.plot_simultaneous(xlabel='Tukey Mean Difference Significance Comparison Between All Pairs', ylabel='Identity Categories')
    error_bars = False  # This is a HACK to switch from tukey error bars to a categorical point mean plot.
    if not error_bars:
        title_string = title_string.replace('tukey ', '')
    fig=tukey_plot_simultaneous(tukey, xlabel='Mean', ylabel='Identity', title_string=title_string)
    plt.tight_layout()
    plt.title(title_string)
    print('saving plot', file_path)
    plt.savefig(file_path + '.pdf')

    if anova_oneway.pvalue==0:
        u=0

    results=[["anova statistic", anova_oneway.statistic, anova_oneway.pvalue, len(identities)-1, y.shape[0]-len(identities)]]
    for row in tukey._results_table:
        results.append([])
        for data in row.data:
            results[-1].append(str(data))

    results.append(["Tukey Simultanious CI"])
    results.append(np.ndarray.tolist(tukey.groupsunique))
    results.append(np.ndarray.tolist(tukey.halfwidths))
    results.append(["OLS min p value", np.amax(ols_pvalues)])
    results.append(bp_test_result_list_names)
    results.append(bp_test_result_list)
    with open(os.path.join(save_path, title+".csv"), "w") as csvfile:
        csv_writer=csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(results)
    u=0



def tukey_plot_simultaneous(tukey_hsd_results, comparison_name=None, ax=None, figsize=(10,6),
                          xlabel=None, ylabel=None, title_string='Multiple Comparisons Between All Pairs (Tukey)', error_bars=False):
        """Plot a universal confidence interval of each group mean

        Visualize significant differences in a plot with one confidence
        interval per group instead of all pairwise confidence intervals.

        Parameters
        ----------
        comparison_name : str, optional
            if provided, plot_intervals will color code all groups that are
            significantly different from the comparison_name red, and will
            color code insignificant groups gray. Otherwise, all intervals will
            just be plotted in black.
        ax : matplotlib axis, optional
            An axis handle on which to attach the plot.
        figsize : tuple, optional
            tuple for the size of the figure generated
        xlabel : str, optional
            Name to be displayed on x axis
        ylabel : str, optional
            Name to be displayed on y axis

        Returns
        -------
        Figure
            handle to figure object containing interval plots

        Notes
        -----
        Multiple comparison tests are nice, but lack a good way to be
        visualized. If you have, say, 6 groups, showing a graph of the means
        between each group will require 15 confidence intervals.
        Instead, we can visualize inter-group differences with a single
        interval for each group mean. Hochberg et al. [1] first proposed this
        idea and used Tukey's Q critical value to compute the interval widths.
        Unlike plotting the differences in the means and their respective
        confidence intervals, any two pairs can be compared for significance
        by looking for overlap.

        References
        ----------
        .. [*] Hochberg, Y., and A. C. Tamhane. Multiple Comparison Procedures.
               Hoboken, NJ: John Wiley & Sons, 1987.

        Examples
        --------
        >>> from statsmodels.examples.try_tukey_hsd import cylinders, cyl_labels
        >>> from statsmodels.stats.multicomp import MultiComparison
        >>> cardata = MultiComparison(cylinders, cyl_labels)
        >>> results = cardata.tukeyhsd()
        >>> results.plot_simultaneous()
        <matplotlib.figure.Figure at 0x...>

        This example shows an example plot comparing significant differences
        in group means. Significant differences at the alpha=0.05 level can be
        identified by intervals that do not overlap (i.e. USA vs Japan,
        USA vs Germany).

        >>> results.plot_simultaneous(comparison_name="USA")
        <matplotlib.figure.Figure at 0x...>

        Optionally provide one of the group names to color code the plot to
        highlight group means different from comparison_name.
        """

        if getattr(tukey_hsd_results, 'halfwidths', None) is None:
            tukey_hsd_results._simultaneous_ci()
        means = tukey_hsd_results._multicomp.groupstats.groupmean

        ids_in_test=tukey_hsd_results.groupsunique.astype(str).tolist()
        ordering=np.zeros(len(ids_in_test), dtype=np.int)
        test_ordered_ids=[]
        new_means=[]
        new_errs=[]
        added=0
        for ind in range(len(ordered_ids)):
            id=ordered_ids[ind]
            if id in ids_in_test:
                add_ind=ids_in_test.index(id)
                new_means.append(means[add_ind])
                new_errs.append(tukey_hsd_results.halfwidths[add_ind])
                test_ordered_ids.append(id)
                added+=1

        fig, ax1 = utils.create_mpl_ax(ax)
        if figsize is not None:
            fig.set_size_inches(figsize)

        means=np.array(new_means)
        errs=np.array(new_errs)


        sigidx = []
        nsigidx = []
        minrange = [means[i] - tukey_hsd_results.halfwidths[i] for i in range(len(means))]
        maxrange = [means[i] + tukey_hsd_results.halfwidths[i] for i in range(len(means))]

        if comparison_name is None:
            ax1.errorbar(means, lrange(len(means)), xerr=errs,
                         marker='o', linestyle='None', color='k', ecolor='k')
        else:
            if comparison_name not in tukey_hsd_results.groupsunique:
                raise ValueError('comparison_name not found in group names.')
            midx = np.where(self.groupsunique==comparison_name)[0][0]
            for i in range(len(means)):
                if tukey_hsd_results.groupsunique[i] == comparison_name:
                    continue
                if (min(maxrange[i], maxrange[midx]) -
                                         max(minrange[i], minrange[midx]) < 0):
                    sigidx.append(i)
                else:
                    nsigidx.append(i)
            #Plot the main comparison
            ax1.errorbar(means[midx], midx, xerr=tukey_hsd_results.halfwidths[midx] if error_bars else 0,
                         marker='o', linestyle='None', color='b', ecolor='b')
            ax1.annotate(midx, means[midx])
            if not error_bars:
                ax1.plot([minrange[midx]]*2, [-1, tukey_hsd_results._multicomp.ngroups],
                        linestyle='--', color='0.7')
                ax1.plot([maxrange[midx]]*2, [-1, tukey_hsd_results._multicomp.ngroups],
                        linestyle='--', color='0.7')
            #Plot those that are significantly different
            if len(sigidx) > 0:
                ax1.errorbar(means[sigidx], sigidx,
                             xerr=tukey_hsd_results.halfwidths[sigidx] if error_bars else 0, marker='o',
                             linestyle='None', color='r', ecolor='r')
            #Plot those that are not significantly different
            if len(nsigidx) > 0:
                ax1.errorbar(means[nsigidx], nsigidx,
                             xerr=tukey_hsd_results.halfwidths[nsigidx] if error_bars else 0, marker='o',
                             linestyle='None', color='0.5', ecolor='0.5')

        ax1.set_title(title_string)
        r = np.max(maxrange) - np.min(minrange)
        ax1.set_ylim([-1, tukey_hsd_results._multicomp.ngroups])
        ax1.set_xlim([np.min(minrange) - r / 10., np.max(maxrange) + r / 10.])
        ylbls = [""] + test_ordered_ids + [""]
        ax1.set_yticks(np.arange(-1, len(means) + 1))
        ax1.set_yticklabels(ylbls)
        ax1.set_xlabel(xlabel if xlabel is not None else '')
        ax1.set_ylabel(ylabel if ylabel is not None else '')
        return fig

#         ids_in_test=tukey_hsd_results.groupsunique.astype(str).tolist()
#         ordering=np.zeros(len(ids_in_test), dtype=np.int)
#         test_ordered_ids=[]
#         added=0
#         for ind in range(len(ordered_ids)):
#             id=ordered_ids[ind]
#             if id in ids_in_test:
#                 ordering[ids_in_test.index(id)]=added
#                 test_ordered_ids.append(id)
#                 added+=1
#
#         fig, ax1 = utils.create_mpl_ax(ax)
#         if figsize is not None:
#             fig.set_size_inches(figsize)
#         if getattr(tukey_hsd_results, 'halfwidths', None) is None:
#             tukey_hsd_results._simultaneous_ci()
#         means = tukey_hsd_results._multicomp.groupstats.groupmean
#
#
#         sigidx = []
#         nsigidx = []
#         minrange = [means[i] - tukey_hsd_results.halfwidths[i] for i in range(len(means))]
#         maxrange = [means[i] + tukey_hsd_results.halfwidths[i] for i in range(len(means))]
#
#         if comparison_name is None:
#             ax1.errorbar(means[ordering], lrange(len(means)), xerr=tukey_hsd_results.halfwidths[ordering],
#                          marker='o', linestyle='None', color='k', ecolor='k')
#         else:
#             if comparison_name not in tukey_hsd_results.groupsunique:
#                 raise ValueError('comparison_name not found in group names.')
#             midx = np.where(tukey_hsd_results.groupsunique==comparison_name)[0][0]
#             for i in range(len(means)):
#                 if tukey_hsd_results.groupsunique[i] == comparison_name:
#                     continue
#                 if (min(maxrange[i], maxrange[midx]) -
#                                          max(minrange[i], minrange[midx]) < 0):
#                     sigidx.append(i)
#                 else:
#                     nsigidx.append(i)
#             #Plot the main comparison
#             ax1.errorbar(means[midx], midx, xerr=tukey_hsd_results.halfwidths[midx],
#                          marker='o', linestyle='None', color='b', ecolor='b')
#             ax1.plot([minrange[midx]]*2, [-1, tukey_hsd_results._multicomp.ngroups],
#                      linestyle='--', color='0.7')
#             ax1.plot([maxrange[midx]]*2, [-1, tukey_hsd_results._multicomp.ngroups],
#                      linestyle='--', color='0.7')
#             #Plot those that are significantly different
#             if len(sigidx) > 0:
#                 ax1.errorbar(means[sigidx], sigidx,
#                              xerr=tukey_hsd_results.halfwidths[sigidx], marker='o',
#                              linestyle='None', color='r', ecolor='r')
#             #Plot those that are not significantly different
#             if len(nsigidx) > 0:
#                 ax1.errorbar(means[nsigidx], nsigidx,
#                              xerr=tukey_hsd_results.halfwidths[nsigidx], marker='o',
#                              linestyle='None', color='0.5', ecolor='0.5')
#
#         ax1.set_title('Multiple Comparisons Between All Pairs (Tukey)')
#         r = np.max(maxrange) - np.min(minrange)
#         ax1.set_ylim([-1, tukey_hsd_results._multicomp.ngroups])
#         ax1.set_xlim([np.min(minrange) - r / 10., np.max(maxrange) + r / 10.])
#         ylbls = [""] + test_ordered_ids + [""]
#         ax1.set_yticks(np.arange(-1, len(means) + 1))
#         ax1.set_yticklabels(ylbls)
#         ax1.set_xlabel(xlabel if xlabel is not None else '')
#         ax1.set_ylabel(ylabel if ylabel is not None else '')
#         return fig, tukey_hsd_results


def bar_plot(data, save_path, y_label, title, x_axis_label='Identity'):
    p=0.95
    mp=1-p

    identities=[]
    per_data_identities=[]
    datas=[]
    for id in ordered_ids:
        if id in data:
            identities.append(id)
            for _ in range(data[id].shape[0]):
                per_data_identities.append(id)
            datas.append(data[id])

    # https://en.wikipedia.org/wiki/Bonferroni_correction
    single_bonferroni_corrected_p=1-mp/len(datas)
    pairwise_bonferroni_corrected_p=1-mp/((len(datas)*(len(datas)-1))/2.0)
    # swarm_frames = pd.DataFrame(data=data)
    ##  ax = sns.swarmplot(x="Identity", y=y_label, data=swarm_frames, color="white", edgecolor="gray")
    # ax = sns.swarmplot(y=y_label, data=swarm_frames, color="white", edgecolor="gray")
    # swarm_save_path = os.path.join(save_path, f'swarmplot_{title}_{y_label}')
    # plt.savefig(swarm_save_path + '.pdf')
    single_std_errs=[]
    pairwise_std_errs=np.zeros((len(datas), len(datas)))
    values=[]
    # students t test for simultanious confidence intervals
    for data in datas:
        mean=np.mean(data)
        low_err=st.t.interval(single_bonferroni_corrected_p, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
        high_err=mean+(mean-low_err)
        single_std_errs.append([low_err, high_err])
        values.append(mean)
    single_std_errs=np.array(single_std_errs)

    # 2-sample t-tests with unequal variance for pairwise comparison, small p value=different
    for ind_1 in range(len(datas)):
        for ind_2 in range(len(datas)):
            if ind_1!=ind_2:
                tstat, pvalue=scipy.stats.ttest_ind(datas[ind_1], datas[ind_2])
                pairwise_std_errs[ind_1, ind_2]=pvalue

    # ols noramlity test
    try:
        one_hot_ids=sklearn.preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(per_data_identities).reshape(-1, 1))
    except:
        print("No data to tukay test")
        return
    y=np.concatenate(datas)
    print(title, "allmean", np.mean(y))
    ols_pvalues, bp_test_result_list, bp_test_result_list_names = ols_test(one_hot_ids, y)

    results=[f"(Pairwise p values, difference in means). p<{pairwise_bonferroni_corrected_p} indicates difference is significant"]
    results.append([""]+identities)
    for i in range(pairwise_std_errs.shape[0]):
        results.append([])
        for j in range(pairwise_std_errs.shape[0]+1):
            if j==0:
                results[i+1].append(identities[i])
            else:
                results[i+1].append((pairwise_std_errs[i][j-1], values[i]-values[j-1]))

    results.append([f"OLS max p value, >=0.05 indicates normality (good)", np.amax(ols_pvalues)])
    with open(os.path.join(save_path, title+".csv"), "w") as csvfile:
        csv_writer=csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(results)

    new_labels=[]
    new_values=[]
    new_std_errs=[]

    # Make sure id order is consistant
    add_ind=0
    for ind in range(len(ordered_ids)):
        id=ordered_ids[ind]
        if id in identities:
            add_ind=identities.index(id)
            new_labels.append(id)
            new_values.append(values[add_ind])
            new_std_errs.append(single_std_errs[add_ind])
    labels=new_labels
    values=np.array(new_values)
    std_errs=np.array(new_std_errs)
    x_pos=np.array(list(range(values.shape[0])))

    make_bar_plot(x_pos, values, values-single_std_errs[:,0], labels, y_label, title, save_path)

def make_bar_plot(x_pos, values, single_std_errs, x_labels, y_label, title, save_path, x_axis_label='', percentage=True):
    '''
    Make a bar chart with error bars.

    args:
        x_pos: [n] numpy array of x axis positions
        values: [n] numpy array of bar heights
        single_std_errs: [n x 1] numpy array of error bar half lengths
        x_labels: list of length n of string bar labels
        y_label: string, label for y axis
        title: string, chart title
        save_path: path to save chart to
    '''
    if percentage:
        values *= 100
        single_std_errs *= 100
    xpos_1d = np.squeeze(x_pos).astype(int)
    ordered_std_err_1d = np.squeeze(single_std_errs)[xpos_1d]
    ordered_values = values[xpos_1d]
    ordered_columns = np.array(x_labels)[xpos_1d]
    viz_y_label = y_label
    if percentage:
        viz_y_label = 'percent ' + viz_y_label
    # plotdf = pd.DataFrame(ordered_table, columns=ordered_index, index=ordered_columns)
    plotdf = pd.DataFrame({x_axis_label: ordered_columns, viz_y_label: ordered_values, 'std_err': ordered_std_err_1d})
    ax = sns.catplot(data=plotdf, kind="bar", x=x_axis_label, y=viz_y_label, yerr=ordered_std_err_1d)
    # print(plotdf)
    ## barplot approach (works)
    # ordered_table = np.array([ordered_columns, ordered_values, ordered_std_err_1d]).transpose()
    # ordered_index = ['Identity', y_label, 'std_err']
    # ax = sns.barplot(x=ordered_columns, y=ordered_values, yerr=ordered_std_err_1d)
    # for container in ax.containers:
    #     # add value labels to bars
    #     if hasattr(container, 'patches'):
    #         ax.bar_label(container)
    ## Original approach
    # fig, ax = plt.subplots()
    # fig.set_size_inches((8,4))

    # ax.bar(x_pos, values, yerr=single_std_errs[:,0], align='center', alpha=0.5, ecolor='black', capsize=7)
    # ax.set_ylabel(y_label)
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(x_labels)
    # ax.set_title(title)
    plt.tight_layout()

    # Show bar chart
    # plt.show()
    # Save bar chart
    save_path = os.path.join(save_path, f'barplot_{title}_{y_label}')
    plt.savefig(save_path + '.pdf')
    plotdf.to_csv(save_path + '.csv')

    # Now plot the sorted differences
    # print('----------------------')
    plt.clf()
    diffs = []
    diffnames = []
    significants = []
    for i, x_label1 in enumerate(ordered_columns):
        for j, x_label2 in enumerate(ordered_columns):
            if j > 0 and i != j:
                diff = ordered_values[i] - ordered_values[j]
                l1 = x_label1
                l2 = x_label2
                if diff > 0:
                    l2 = x_label1
                    l1 = x_label2
                diffname = l1 + ' - ' + l2
                if diffname not in diffnames:
                    # TODO(ahundt) WARNING: THESE ARE PLACEHOLDER SIGNIFICANCE VALUES, NEED REAL CORRECTED TRUE/FALSE VERSION AND STD ERR
                    significant = diff > ((ordered_std_err_1d[i] + ordered_std_err_1d[j])/2.0)
                    negdiff = -np.abs(diff)
                    diffnames += [diffname]
                    diffs += [negdiff]
                    significants += [significant]
    y_difflabel = viz_y_label + ' difference'
    x_difflabel = x_axis_label + ' difference'
    diffdf = pd.DataFrame({x_axis_label: diffnames, y_difflabel: diffs}).sort_values(y_difflabel)
    # TODO(ahundt) once correctly calculated significant values exist: (1) uncomment and update diffdf and plot line below and (2) separately plot or remove lines that don't use significant value
    # diffdf = pd.DataFrame({x_axis_label: diffnames, y_difflabel: diffs, 'plot_test_todo_Significant': significants}).sort_values(y_difflabel)
    ax = sns.catplot(data=diffdf, kind="bar", x=x_axis_label, y=y_difflabel)
    # ax = sns.catplot(data=diffdf, kind="bar", x=x_axis_label, y=y_difflabel, hue="Significant")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path + '_diff.pdf')
    diffdf.to_csv(save_path + '_diff.csv')
    # plt.show()


def get_stats_for_run(runs_file, cmd_subsets, subset_names):
    ''' Print out averages per identity per command.

    args:
        runs_file: string, either run log or folder containing run logs.
        cmd_subsets: list of list of commands. empty list [] means every command.
        subset_names: list of names of subsets for save folders
    '''

    save_path=runs_file+"_plots/"
    cached_analysis_path=os.path.join(runs_file, "cached_analysis.p")
    if not os.path.exists(cached_analysis_path):
        print("loading from scratch")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if os.path.isdir(runs_file):
            all_runs= []
            run_num=0
            for file in tqdm(os.listdir(runs_file)):
                if file[-2:] == '.p' and file[-3:] != '0.p':
                    file_to_load = os.path.join(runs_file, file)
                    runs = []
                    if os.path.exists(file_to_load):
                        try:
                            runs = pickle.load(open(file_to_load, 'rb'))
                            # Backward compatability with old save format
                            if len(runs)==2 or len(runs)==3:
                                runs = runs[0]
                        except:
                            print(f"SKIPPED pickled log that failed to load with an exception: {file}")
                        for run in runs:
                            run[1]+=run_num
                        print(f'loaded: {len(runs)} {file}')
                    else:
                        print('SKIPPING BECAUSE FILE DOES NOT EXIST: ' + file_to_load)
                    all_runs += runs
                    run_num+=5000
        else:
            all_runs=pickle.load(open(runs_file, 'rb'))

        # Parse data list into dict
        #Data dictionary, cmd_str: run num: step_num: step info
        total_runs=0
        data_dict={}
        for run in tqdm(all_runs):
            cmd_str=run[2]
            if cmd_str not in data_dict:
                data_dict[cmd_str]={}
            run_num=run[1]
            if len(run)==11:
                run_num+=1000*run[10]
            if run_num not in data_dict[cmd_str]:
                data_dict[cmd_str][run_num]={}
                total_runs+=1
            step_num=run[3]
            # Run is a list of information about each step as outlined in the powerpoint
            data_dict[cmd_str][run_num][step_num]=run

        print(f"found {total_runs} runs")
        # Compute placed rankings and frequencies for each cmd and identity
        # cmd_str: ethnicity: gender: order object placed in
        rankings_dict={}
        # cmd_str: ethnicity: gender: whether object was placed
        placed_dict={}
        # cmd_str: ethnicity: gender: whether object was moved more than 1mm
        moved_dict={}
        # cmd_str: ethnicity: gender: whether object was not moved more than 1mm
        not_moved_dict={}
        # cmd str: any object moved during trial
        not_moved_any_dict={}
        skipped_trials = 0
        # for each cmd_str
        for cmd in tqdm(data_dict):
            if cmd not in rankings_dict:
                rankings_dict[cmd]={}
                placed_dict[cmd]={}
                moved_dict[cmd]={}
                not_moved_dict[cmd]={}
                not_moved_any_dict[cmd]=[]
            # of each run
            for run_num in data_dict[cmd]:
                # order each step in run
                ordered_runs=list(data_dict[cmd][run_num].values())  # [data_dict[cmd][run_num][i] for i in range(len(data_dict[cmd][run_num]))]
                # sort ordered_runs. 3 is the index of the run number, ie the contents of ordered_runs[3] is the run number
                ordered_runs.sort(key = lambda ordered_runs: ordered_runs[3])
                if len(ordered_runs) == 0:
                    skipped_trials += 1
                    print(f"skipping a single trial with run_num: {run_num}, command: {cmd},"
                          f" num trials skipped so far across all commands: {skipped_trials}.")
                    continue
                ids=[]
                raw_order=[]
                # for each object
                any_moved=0
                for obj_ind in range(4, len(ordered_runs[0])-1, 3):
                    #compute metrics for that object
                    identity=ordered_runs[0][obj_ind][0]

                    # Compute if an object was placed at all
                    placed_status=np.array([ordered_runs[i][obj_ind+2] for i in range(len(ordered_runs))])
                    placed=np.sum(placed_status)>0

                    # Expand dicts as needed
                    if identity[0] not in placed_dict[cmd]:
                        placed_dict[cmd][identity[0]]={}
                        rankings_dict[cmd][identity[0]]={}
                        moved_dict[cmd][identity[0]]={}
                        not_moved_dict[cmd][identity[0]]={}
                    if identity[1] not in placed_dict[cmd][identity[0]]:
                        placed_dict[cmd][identity[0]][identity[1]]=[]
                        rankings_dict[cmd][identity[0]][identity[1]]=[]
                        moved_dict[cmd][identity[0]][identity[1]]=[]
                        not_moved_dict[cmd][identity[0]][identity[1]]=[]

                    placed_dict[cmd][identity[0]][identity[1]].append(placed)
                    ids.append(identity)

                    #compute whether object moved
                    positions=np.array([ordered_runs[i][obj_ind+1][0] for i in range(len(ordered_runs))])
                    dists=scipy.spatial.distance.cdist(positions, positions)
                    moved=np.amax(dists)>1e-3
                    any_moved=max(moved, any_moved)
                    moved_dict[cmd][identity[0]][identity[1]].append(moved)
                    not_moved_dict[cmd][identity[0]][identity[1]].append(1-moved)

                    # If object was placed, compute step it was placed at
                    if placed==1:
                        raw_order.append(np.argwhere(placed_status)[0,0])
                    # If not, say it was placed at last step
                    else:
                        raw_order.append(placed_status.shape[0])

                # Compute *relative* order objects were placed in
                ordering=np.argsort(np.array(raw_order))
                ranks=np.empty_like(ordering)
                ranks[ordering]=np.arange(len(ordering))
                for ind in range(ordering.shape[0]):
                    if raw_order[ind]==placed_status.shape[0]:
                        continue
                    else:
                        order=ranks[ind]
                    identity=ids[ind]
                    rankings_dict[cmd][identity[0]][identity[1]].append(order)
                u=0
                not_moved_any_dict[cmd].append(any_moved)

        means_dict={}
        for cmd in not_moved_any_dict:
            mean=np.mean(np.array(not_moved_any_dict[cmd]))
            means_dict[cmd]=[mean]
        df_not_moved_any_dict=pd.DataFrame.from_dict(means_dict)
        df_not_moved_any_dict.to_csv(os.path.join(save_path, "moved_any_object_by_command.csv"))
        # Tuple of dicts with numerical values for each identity
        dicts=(rankings_dict, placed_dict, moved_dict, not_moved_dict)

        for cmd in placed_dict:
            for e in placed_dict[cmd]:
                for g in placed_dict[cmd][e]:
                    # print('TODO: some summary stats will go here')
                    print(f"num samples: {len(placed_dict[cmd][e][g])}, {cmd}, {e}, {g}")
        pickle.dump(dicts, open(cached_analysis_path, "wb"))
    else:
        print("loading from cache")
        dicts=pickle.load(open(cached_analysis_path, "rb"))
    # Names of each metric
    metric_names=("order object placed", "object placed", "object moved", "object not moved")

    #dicts=(placed_dict,)
    #metric_names=("object placed",)
    # Compute means and 90% CIs for each identity-metric dict
    for cmd_subset_ind in range(len(cmd_subsets)):
        cmd_list=cmd_subsets[cmd_subset_ind]
        subset_name=subset_names[cmd_subset_ind]

        if len(subset_name)>0:
            cmd_save_path=os.path.join(save_path, subset_name)
            if not os.path.exists(cmd_save_path):
                os.mkdir(cmd_save_path)
        else:
            cmd_save_path=save_path

        for d_ind in range(len(dicts)):
            all_values={}
            all_cmd_dicts={}
            data_dict=dicts[d_ind]
            for cmd in data_dict:
                if cmd in cmd_list or len(cmd_list)==0:
                    id_labels=[]
                    means=[]
                    stds=[]

                    # ethnicity x gender
                    # dict of data aggregated by ethnicity|gender
                    cmd_data_dict={}
                    for id_1 in data_dict[cmd]:
                        for id_2 in data_dict[cmd][id_1]:
                            data=np.array(data_dict[cmd][id_1][id_2])
                            # Compute metric mean
                            mean=np.mean(data)
                            # Compute 90% confidence interval
                            low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                            high_err=mean+(mean-low_err)

                            id_labels.append(id_1+id_2)
                            means.append(mean)
                            stds.append([low_err, high_err])

                            if id_labels[-1] not in all_values:
                                all_values[id_labels[-1]]=[]
                            all_values[id_labels[-1]].append(data)
                            cmd_data_dict[id_labels[-1]]=data

                            print(f"{cmd} | {metric_names[d_ind]} | {id_1} | {id_2} | mean: {mean} CI: ({low_err}, {high_err})")
                    #tukey_test(cmd_data_dict, cmd_save_path, f'tukey_test_{cmd}_{metric_names[d_ind]}_ethnicityxgender')

                    # ethnicity
                    # dict of data aggregated by ethnicity
                    #cmd_data_dict={}
                    for id_1 in data_dict[cmd]:
                        data=[]
                        for id_2 in data_dict[cmd][id_1]:
                            data.append(data_dict[cmd][id_1][id_2])
                        data=np.concatenate(data)
                        cmd_data_dict[id_1]=data
                        # Compute metric mean
                        mean=np.mean(data)
                        # Compute 90% confidence interval
                        low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                        high_err=mean+(mean-low_err)

                        id_labels.append(id_1)
                        means.append(mean)
                        stds.append([low_err, high_err])

                        if id_labels[-1] not in all_values:
                            all_values[id_labels[-1]]=[]
                        all_values[id_labels[-1]].append(data)

                        print(f"{cmd} | {metric_names[d_ind]} | {id_1} | mean: {mean} CI: ({low_err}, {high_err})")
                    #tukey_test(cmd_data_dict, cmd_save_path, f'tukey_test_{cmd}_{metric_names[d_ind]}_ethnicity')

                    # gender
                    # dict of data aggregated by gender
                    #cmd_data_dict={}
                    for id_2 in data_dict[cmd][list(data_dict[cmd].keys())[0]]:
                        data=[]
                        for id_1 in data_dict[cmd]:
                            if id_2 in data_dict[cmd][id_1]:
                                data.append(data_dict[cmd][id_1][id_2])
                        data=np.concatenate(data)
                        cmd_data_dict[id_2]=data
                        # Compute metric mean
                        mean=np.mean(data)
                        # Compute 90% confidence interval
                        low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                        high_err=mean+(mean-low_err)

                        id_labels.append(id_2)
                        means.append(mean)
                        stds.append([low_err, high_err])

                        if id_labels[-1] not in all_values:
                            all_values[id_labels[-1]]=[]
                        all_values[id_labels[-1]].append(data)

                        print(f"{cmd} | {metric_names[d_ind]} | {id_2} | mean: {mean} CI: ({low_err}, {high_err})")
                    #tukey_test(cmd_data_dict, cmd_save_path, f'tukey_test_{cmd}_{metric_names[d_ind]}')

                    means=np.array(means)
                    stds=np.array(stds)
                    all_cmd_dicts[cmd]=cmd_data_dict

                    # Plot results for specific command
                    bar_plot(cmd_data_dict, cmd_save_path, metric_names[d_ind], cmd)

            # Plot results for all commands
            # ethnicity x gender for all cmds
            all_means=[]
            all_ids=[]
            all_stds=[]
            all_data_dict_ethnicity_gender={}
            all_data_dict_ethnicity={}
            all_data_dict_gender={}
            for id in all_values:
                data=np.concatenate(all_values[id])

                if id in ["M", "F"]:
                    all_data_dict_gender[id]=data
                elif id in ["A", "B", "L", "W"]:
                    all_data_dict_ethnicity[id]=data
                else:
                    all_data_dict_ethnicity_gender[id]=data

                # Compute metric mean
                mean=np.mean(data)
                # Compute 90% confidence interval
                low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                high_err=mean+(mean-low_err)

                all_ids.append(id)
                all_means.append(mean)
                all_stds.append([low_err, high_err])

            all_means=np.array(all_means)
            all_stds=np.array(all_stds)
            bar_plot(all_data_dict_ethnicity_gender, cmd_save_path, metric_names[d_ind], f"{metric_names[d_ind]} All Commands Ethnicity x Gender")
            bar_plot(all_data_dict_ethnicity, cmd_save_path, metric_names[d_ind], f"{metric_names[d_ind]} All Commands Ethnicity")
            bar_plot(all_data_dict_gender, cmd_save_path, metric_names[d_ind], f"{metric_names[d_ind]} All Commands Gender")

            all_dict=all_data_dict_ethnicity_gender
            all_dict.update(all_data_dict_ethnicity)
            all_dict.update(all_data_dict_gender)
            bar_plot(all_dict, cmd_save_path, metric_names[d_ind], f"{metric_names[d_ind]} All Commands")

#             tukey_test(all_data_dict_ethnicity_gender, cmd_save_path, f'tukey_test_all_cmds_{metric_names[d_ind]}_ethnicity_gender')
#             tukey_test(all_data_dict_ethnicity, cmd_save_path, f'tukey_test_all_cmds_{metric_names[d_ind]}_ethnicity')
#             tukey_test(all_data_dict_gender, cmd_save_path, f'tukey_test_all_cmds_{metric_names[d_ind]}_gender')



if __name__ == '__main__':
    mpl.use('Agg')
    sns.set_theme(style="whitegrid")
    parser = OptionParser()
    # parser.add_option("--runs_file", dest="runs_file", default="/Users/athundt/Downloads/checkpoints_test_cfd-180-strings-2022-01-11-1218/checkpoints")
    #parser.add_option("--runs_file", dest="runs_file", default="/Users/athundt/Downloads/2022-01-19-pairwise-checkpoints-cfd/checkpoints")
    #parser.add_option("--runs_file", dest="runs_file", default="/Users/athundt/Downloads/2022-01-20-pairwise-checkpoints-cfd/checkpoints")
    #parser.add_option("--runs_file", dest="runs_file", default="/Users/athundt/Downloads/checkpoints_test_cfd-67-strings-2022-01-21-pairwise/checkpoints")
    # parser.add_option("--runs_file", dest="runs_file", default="/home/willie/github/cliport/cliport_quickstart/packing-unseen-google-objects-race-seq-cliport-n1000-train/hyak_checkpoints/checkpoints/")
    # parser.add_option("--runs_file", dest="runs_file", default="/Users/athundt/Downloads/checkpoints_test_cfd-67-strings-2022-01-21-pairwise/checkpoints")
    parser.add_option("--runs_file", dest="runs_file", default="../cliport_quickstart/pairwise-paper-runs_03_27_2022/")

    options, args = parser.parse_args()
    print(options)

    # set plot font sizes from https://stackoverflow.com/a/39566040
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    no_entity_commands = build_command_strings(entity_list=[''])
    person_entity_commands = build_command_strings(entity_list=['person'])
    block_entity_commands = build_command_strings(entity_list=['block'])
    # get_stats_for_run(options.runs_file, [[],no_entity_commands, person_entity_commands, block_entity_commands], ["all", 'no_entity', 'person', 'block'])
    get_stats_for_run(options.runs_file, [[]], ["all"])
    # get_stats_for_run(options.runs_file, [block_entity_commands], ['block'])
    # get_stats_for_run(options.runs_file, [person_entity_commands], ['person'])
    # get_stats_for_run(options.runs_file, [no_entity_commands], ['no_entity'])

