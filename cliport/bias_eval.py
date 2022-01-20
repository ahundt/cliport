import pickle
import numpy as np
import os
import pandas as pd
from optparse import OptionParser
import scipy.stats as st
import scipy.spatial
import matplotlib as mpl
mpl.use('Agg')
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

ordered_ids=['AF', 'AM', 'BF', 'BM', 'LF', 'LM', 'WF', 'WM', 'A', 'B', 'L', 'W', 'F', 'M']

def ols_test(X, y):
    X_constant = sm.add_constant(X)
    lin_reg = sm.OLS(y,X_constant).fit()
    return lin_reg.pvalues

def breusch_pagan_test(x, y):
    '''
    Breusch-Pagan test for heteroskedasticity in a linear regression model:
    H_0 = No heteroskedasticity.
    H_1 = Heteroskedasticity is present.

    Inputs:
    x = a numpy.ndarray containing the predictor variables. Shape = (nSamples, nPredictors).
    y = a 1D numpy.ndarray containing the response variable. Shape = (nSamples, ).

    Outputs a list containing three elements:
    1. the Breusch-Pagan test statistic.
    2. the p-value for the test.
    3. the test result.
    '''

    if y.ndim != 1:
        raise SystemExit('Error: y has more than 1 dimension.')
    if x.shape[0] != y.shape[0]:
        raise SystemExit('Error: the number of samples differs between x and y.')
    else:
        n_samples = y.shape[0]

    # fit an OLS linear model to y using x:
    lm = LinearRegression()
    lm.fit(x, y)

    # calculate the squared errors:
    err = (y - lm.predict(x))**2

    # fit an auxiliary regression to the squared errors:
    # why?: to estimate the variance in err explained by x
    lm.fit(x, err)
    pred_err = lm.predict(x)
    del lm

    # calculate the coefficient of determination:
    ss_tot = sum((err - np.mean(err))**2)
    ss_res = sum((err - pred_err)**2)
    r2 = 1 - (ss_res / ss_tot)
    del err, pred_err, ss_res, ss_tot

    # calculate the Lagrange multiplier:
    LM = n_samples * r2
    del r2

    # calculate p-value. degrees of freedom = number of predictors.
    # this is equivalent to (p - 1) parameter restrictions in Wikipedia entry.
    pval = stats.distributions.chi2.sf(LM, x.shape[1])

    if pval < 0.01:
        test_result = 'Heteroskedasticity present at 99% CI.'
    elif pval < 0.05:
        test_result = 'Heteroskedasticity present at 95% CI.'
    else:
        test_result = 'No significant heteroskedasticity.'
    return [LM, pval, test_result]

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
    ols_pvalues=ols_test(one_hot_ids, y)
    LM, bp_pval, test_result=breusch_pagan_test(one_hot_ids, y)

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
    fig, _=tukey_plot_simultaneous(tukey, xlabel='Tukey Mean Difference Significance Comparison Between All Pairs', ylabel='Identity Categories')
    plt.tight_layout()
    plt.title(title_string)
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
    results.append(["OLS min p value", np.amax(ols_pvalues), "bptest p value", bp_pval])
    with open(os.path.join(save_path, title+".csv"), "w") as csvfile:
        csv_writer=csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(results)
    u=0


def tukey_plot_simultaneous(tukey_hsd_results, comparison_name=None, ax=None, figsize=(10,6),
                          xlabel=None, ylabel=None):
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
        
        ids_in_test=tukey_hsd_results.groupsunique.astype(str).tolist()
        ordering=np.zeros(len(ids_in_test), dtype=np.int)
        test_ordered_ids=[]
        added=0
        for ind in range(len(ordered_ids)):
            id=ordered_ids[ind]
            if id in ids_in_test:
                ordering[ids_in_test.index(id)]=added
                test_ordered_ids.append(id)
                added+=1
        
        fig, ax1 = utils.create_mpl_ax(ax)
        if figsize is not None:
            fig.set_size_inches(figsize)
        if getattr(tukey_hsd_results, 'halfwidths', None) is None:
            tukey_hsd_results._simultaneous_ci()
        means = tukey_hsd_results._multicomp.groupstats.groupmean


        sigidx = []
        nsigidx = []
        minrange = [means[i] - tukey_hsd_results.halfwidths[i] for i in range(len(means))]
        maxrange = [means[i] + tukey_hsd_results.halfwidths[i] for i in range(len(means))]

        if comparison_name is None:
            ax1.errorbar(means[ordering], lrange(len(means)), xerr=tukey_hsd_results.halfwidths[ordering],
                         marker='o', linestyle='None', color='k', ecolor='k')
        else:
            if comparison_name not in tukey_hsd_results.groupsunique:
                raise ValueError('comparison_name not found in group names.')
            midx = np.where(tukey_hsd_results.groupsunique==comparison_name)[0][0]
            for i in range(len(means)):
                if tukey_hsd_results.groupsunique[i] == comparison_name:
                    continue
                if (min(maxrange[i], maxrange[midx]) -
                                         max(minrange[i], minrange[midx]) < 0):
                    sigidx.append(i)
                else:
                    nsigidx.append(i)
            #Plot the main comparison
            ax1.errorbar(means[midx], midx, xerr=tukey_hsd_results.halfwidths[midx],
                         marker='o', linestyle='None', color='b', ecolor='b')
            ax1.plot([minrange[midx]]*2, [-1, tukey_hsd_results._multicomp.ngroups],
                     linestyle='--', color='0.7')
            ax1.plot([maxrange[midx]]*2, [-1, tukey_hsd_results._multicomp.ngroups],
                     linestyle='--', color='0.7')
            #Plot those that are significantly different
            if len(sigidx) > 0:
                ax1.errorbar(means[sigidx], sigidx,
                             xerr=tukey_hsd_results.halfwidths[sigidx], marker='o',
                             linestyle='None', color='r', ecolor='r')
            #Plot those that are not significantly different
            if len(nsigidx) > 0:
                ax1.errorbar(means[nsigidx], nsigidx,
                             xerr=tukey_hsd_results.halfwidths[nsigidx], marker='o',
                             linestyle='None', color='0.5', ecolor='0.5')

        ax1.set_title('Multiple Comparisons Between All Pairs (Tukey)')
        r = np.max(maxrange) - np.min(minrange)
        ax1.set_ylim([-1, tukey_hsd_results._multicomp.ngroups])
        ax1.set_xlim([np.min(minrange) - r / 10., np.max(maxrange) + r / 10.])
        ylbls = [""] + test_ordered_ids + [""]
        ax1.set_yticks(np.arange(-1, len(means) + 1))
        ax1.set_yticklabels(ylbls)
        ax1.set_xlabel(xlabel if xlabel is not None else '')
        ax1.set_ylabel(ylabel if ylabel is not None else '')
        return fig, tukey_hsd_results


def bar_plot(labels, values, std_errs, save_path, y_label, title):
    
    new_labels=[]
    new_values=[]
    
    # Make sure id order is consistant
    for ind in range(len(ordered_ids)):
        id=ordered_ids[ind]
        if id in labels:
            new_labels.append(id)
            new_values.append(values[ind])
    labels=new_labels
    values=np.array(new_values)  
    
    fig, ax = plt.subplots()

    x_pos=np.array(list(range(values.shape[0])))

    ax.bar(x_pos, values, yerr=values-std_errs[:,0], align='center', alpha=0.5, ecolor='black', capsize=7)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
#     ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    save_path = os.path.join(save_path, f'barplot_{title}_{y_label}')
    plt.savefig(save_path + '.pdf')

    table = [values, std_errs, [title]*len(values), [y_label]*len(values)]
    # creating the dataframe
    # df = pd.DataFrame(data = table,
    #                   index = [y_label, y_label + ' standard error', 'command string', 'object state type'],
    #                   columns = labels)
    # df.to_csv(save_path + '.csv')
#     plt.show()

def get_stats_for_run(runs_file, cmd_subsets, subset_names):
    ''' Print out averages per identity per command.

    args:
        runs_file: string, either run log or folder containing run logs.
        cmd_subsets: list of list of commands. empty list [] means every command.
        subset_names: list of names of subsets for save folders
    '''

    save_path=runs_file+"_plots/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.isdir(runs_file):
        all_runs= []
        run_num=0
        for file in os.listdir(runs_file):
            if file[-2:] == '.p':
                file_to_load = os.path.join(runs_file, file)
                
                try:
                    runs=pickle.load(open(file_to_load, 'rb'))
                except:
                    print(f"SKIPPED {file}")
                for run in runs:
                    run[1]+=run_num
                print(f'loaded: {len(runs)} {file}')
                all_runs += runs
                run_num+=5000

    # Parse data list into dict
    #Data dictionary, cmd_str: run num: step_num: step info
    data_dict={}
    for run in all_runs:
        cmd_str=run[2]
        if cmd_str not in data_dict:
            data_dict[cmd_str]={}
        run_num=run[1]
        if run_num not in data_dict[cmd_str]:
            data_dict[cmd_str][run_num]={}
        step_num=run[3]
        # Run is a list of information about each step as outlined in the powerpoint
        data_dict[cmd_str][run_num][step_num]=run

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
    # for each cmd_str
    for cmd in data_dict:
        if cmd not in rankings_dict:
            rankings_dict[cmd]={}
            placed_dict[cmd]={}
            moved_dict[cmd]={}
            not_moved_dict[cmd]={}
            not_moved_any_dict[cmd]=[]
        # of each run
        for run_num in data_dict[cmd]:
            # order each step in run
            try:
                ordered_runs=[data_dict[cmd][run_num][i] for i in range(len(data_dict[cmd][run_num]))]
            except KeyError:
                print(f"key not found {cmd} {run_num} {len(data_dict[cmd][run_num])} runs found")
                continue
            ids=[]
            raw_order=[]
            # for each object
            any_moved=0
            for obj_ind in range(4, len(ordered_runs[0]), 3):
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
    # Names of each metric
    metric_names=("order object placed", "object placed", "object moved", "object not moved")
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
                    tukey_test(cmd_data_dict, cmd_save_path, f'tukey_test_{cmd}_{metric_names[d_ind]}')

                    means=np.array(means)
                    stds=np.array(stds)

                    # Plot results for specific command
                    bar_plot(id_labels, means, stds, cmd_save_path, metric_names[d_ind], cmd)

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
            bar_plot(all_ids, all_means, all_stds, cmd_save_path, metric_names[d_ind], "All Commands")

            tukey_test(all_data_dict_ethnicity_gender, cmd_save_path, f'tukey_test_all_cmds_{metric_names[d_ind]}_ethnicity_gender')
            tukey_test(all_data_dict_ethnicity, cmd_save_path, f'tukey_test_all_cmds_{metric_names[d_ind]}_ethnicity')
            tukey_test(all_data_dict_gender, cmd_save_path, f'tukey_test_all_cmds_{metric_names[d_ind]}_gender')



if __name__ == '__main__':
    parser = OptionParser()
    #parser.add_option("--runs_file", dest="runs_file", default="/Users/athundt/Downloads/checkpoints_test_cfd-180-strings-2022-01-11-1218/checkpoints")
    #parser.add_option("--runs_file", dest="runs_file", default="/Users/athundt/Downloads/2022-01-19-pairwise-checkpoints-cfd/checkpoints")
    parser.add_option("--runs_file", dest="runs_file", default="/home/willie/github/cliport/cliport_quickstart/packing-unseen-google-objects-race-seq-cliport-n1000-train/checkpoints/checkpoints_test_cfd-180-strings-2022-01-11-1218 2/checkpoints_test_cfd-180-strings-2022-01-11-1218/checkpoints")
    options, args = parser.parse_args()
    print(options)

    # set plot font sizes from https://stackoverflow.com/a/39566040
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
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

