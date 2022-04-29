import numpy as np
import matplotlib.pyplot as plt
import csv

def read_clean_R_csv(R_csv_path):
    ''' Read csv from error_bars.r.
    
    args:
        R_csv_path: path to csv produced.
        
    returns:
        r_csv: list of effects
    '''
    
    r_csv=[]
    with open(R_csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            r_csv.append(row)
    return r_csv

def divide_error(mean1, error1, mean2, error2):
    div_mean=mean1/mean2
    
    div_ll=(mean1-error1)/(mean2-error2)
    div_lh=(mean1-error1)/(mean2+error2)
    div_hl=(mean1+error1)/(mean2-error2)
    div_hh=(mean1+error1)/(mean2+error2)
    dvs=np.array([div_ll, div_lh, div_hl, div_hh])
    max_err=np.max(np.abs(div_mean-dvs))
    return div_mean, max_err

def get_and_sort(ethnicity, gender, ethnicity_gender_effects, data_ind):
    cmd_data=ethnicity_gender_effects[ethnicity][gender]
    cmd_data.sort(key=lambda x:x[0])
    cmd_means=np.array([cmd_data[i][data_ind] for i in range(len(cmd_data))])
    return cmd_means
            
    
def cmd_effects_plot(cleaned_R_csv):
    '''Plot effects of commands on success rate.
    
    args:
        cleaned_R_csv: csv of effects from R
    
    '''
    
    ethnicity_map={"EthnicityL": "L", "EthnicityB": "B", "EthnicityA": "A"}
    
    # Parse csv into ethnicity x gender effects
    ethnicity_gender_effects={}
    net_ethnicity_gender_effects={}
    cmd_effects=[]
    # ethnicity_effects={}
    # gender_effects={}
    # command_effects=[]
    for row in cleaned_R_csv[1:]:
        category=row[1]
        effect=float(row[2])
        err=float(row[3])
        sig=float(row[5])<0.05
        
        # # Split cmd ethnicity_gender
        if category.count(":")==1:
            categories=category.split(":")
            ethnicity_gender=categories[0][len("Ethnicity_Gender"):]
            ethnicity_gender=ethnicity_gender.split(" ")
            ethnicity=ethnicity_gender[0]
            gender=ethnicity_gender[1]
            cmd=categories[1]
            if ethnicity not in ethnicity_gender_effects:
                ethnicity_gender_effects[ethnicity]={}
            if gender not in ethnicity_gender_effects[ethnicity]:
                ethnicity_gender_effects[ethnicity][gender]=[]
            ethnicity_gender_effects[ethnicity][gender].append([cmd, effect, err, sig])
        elif category.count(":")==0 and category.split(":")[0].startswith("Ethnicity_Gender"):
            ethnicity_gender=category[len("Ethnicity_Gender"):]
            ethnicity_gender=ethnicity_gender.split(" ")
            ethnicity=ethnicity_gender[0]
            gender=ethnicity_gender[1]
            if ethnicity not in net_ethnicity_gender_effects:
                net_ethnicity_gender_effects[ethnicity]={}
            if gender not in net_ethnicity_gender_effects[ethnicity]:
                net_ethnicity_gender_effects[ethnicity][gender]=[]
            net_ethnicity_gender_effects[ethnicity][gender]=[None, effect, err, sig]
        elif category.count(":")==0:
            if category.split(":")[0].startswith("Command"):
                cmd=category
            else:
                cmd="pack the person in the brown box"
            cmd_effects.append([cmd, effect+float(cleaned_R_csv[1][2]), err, sig])
            
            
        # elif category.count(":")==1 and category.split(":")[0]=="GenderF":
        #     categories=category.split(":")
        #     gender="F"
        #     cmd=categories[1]
        #     if gender not in gender_effects:
        #         gender_effects[gender]=[]
        #     gender_effects[gender].append([cmd, effect, err, sig])
        # elif category.count(":")==1 and category.split(":")[0].startswith("Ethnicity") and "Command" in category:
        #     categories=category.split(":")
        #     ethnicity=ethnicity_map[categories[0]]
        #     cmd=categories[1]
        #     if ethnicity not in ethnicity_effects:
        #         ethnicity_effects[ethnicity]=[]
        #     ethnicity_effects[ethnicity].append([cmd, effect, err, sig])
        # elif category.count(":")==0 and category.split(":")[0].startswith("Command"):
        #     categories=category.split(":")
        #     cmd=categories
        #     command_effects.append([cmd, effect, err, sig])

    ethnicity_gender_effects['A']['M'].sort(key=lambda x:x[0])
    commands=[ethnicity_gender_effects['A']['M'][i][0] for i in range(len(ethnicity_gender_effects['A']['M']))]
    commands.sort()
    neutral_ind=commands.index('Commandpack the person block in the brown box')
 
    # Make plots
    fig, subplots = plt.subplots(1, 8)
    plot_ind=0
    for ethnicity in ["W", "A", "B", "L"]:
        for gender in ["M", "F"]:
            effects=np.zeros(len(commands))
            err=np.zeros(len(commands))
            sig=np.zeros(len(commands))
            
            if ethnicity in ethnicity_gender_effects and gender in ethnicity_gender_effects[ethnicity]:
                cmd_data=cmd_effects
                cmd_data.sort(key=lambda x:x[0])
                cmd_means=np.array([cmd_data[i][1] for i in range(len(cmd_data))])
                
                id_cmd_means=get_and_sort(ethnicity, gender, ethnicity_gender_effects, 1)
                wm_cmd_means=get_and_sort('W', 'M', ethnicity_gender_effects, 1)
                
                id_cmd_errs=get_and_sort(ethnicity, gender, ethnicity_gender_effects, 2)
                wm_cmd_errs=get_and_sort('W', 'M', ethnicity_gender_effects, 2)
                
                relative_perfs=[]
                relative_errs=[]
                neutral_rel_mean, neutral_rel_err=divide_error(id_cmd_means[neutral_ind], id_cmd_errs[neutral_ind], wm_cmd_means[neutral_ind], wm_cmd_errs[neutral_ind])
                for cmd_ind in range(len(wm_cmd_means)):
                    cmd_rel_mean, cmd_rel_err=divide_error(id_cmd_means[cmd_ind], id_cmd_errs[cmd_ind], wm_cmd_means[cmd_ind], wm_cmd_errs[cmd_ind])
                    cmp_rel_mean, cmp_rel_error=divide_error(cmd_rel_mean, cmd_rel_err, neutral_rel_mean, neutral_rel_err)
                    relative_perfs.append(cmp_rel_mean)
                    relative_errs.append(cmp_rel_error)
            else:
                cmd_data=cmd_effects
                cmd_data.sort(key=lambda x:x[0])
                cmd_means=np.array([cmd_data[i][1] for i in range(len(cmd_data))])
                
                id_cmd_means=get_and_sort('W', 'M', ethnicity_gender_effects, 1)
                wm_cmd_means=get_and_sort('W', 'M', ethnicity_gender_effects, 1)
                
                id_cmd_errs=get_and_sort('W', 'M', ethnicity_gender_effects, 2)
                wm_cmd_errs=get_and_sort('W', 'M', ethnicity_gender_effects, 2)
                
                relative_perfs=[]
                relative_errs=[]
                neutral_rel_mean, neutral_rel_err=divide_error(id_cmd_means[neutral_ind], id_cmd_errs[neutral_ind], wm_cmd_means[neutral_ind], wm_cmd_errs[neutral_ind])
                for cmd_ind in range(len(wm_cmd_means)):
                    cmd_rel_mean, cmd_rel_err=divide_error(id_cmd_means[cmd_ind], id_cmd_errs[cmd_ind], wm_cmd_means[cmd_ind], wm_cmd_errs[cmd_ind])
                    cmp_rel_mean, cmp_rel_error=divide_error(cmd_rel_mean, cmd_rel_err, neutral_rel_mean, neutral_rel_err)
                    relative_perfs.append(cmp_rel_mean)
                    relative_errs.append(cmp_rel_error)
                
                    
                
                # effects+=eg_effects
                # #effects+=net_ethnicity_gender_effects[ethnicity][gender][1]
                # effects=effects/cmd_means
                # eg_err=np.array([data[i][2] for i in range(len(data))])
                # err+=eg_err#np.square(eg_err)
                # #err+=np.square(net_ethnicity_gender_effects[ethnicity][gender][2])
                # #err=np.sqrt(err)
                # err/=cmd_means
                # eg_sig=np.array([data[i][3] for i in range(len(data))])
                # sig+=eg_sig
                # #sig+=net_ethnicity_gender_effects[ethnicity][gender][3]
                # sig/=cmd_means
                
                
            
            relative_perfs=np.array(relative_perfs)-1
            ax=subplots[plot_ind]
            y_pos = np.arange(len(relative_perfs))
            my_cmap = plt.get_cmap("viridis")
            rescale = lambda y: np.abs(y / (np.max(y) - np.min(y)))
            
            ax.barh(y_pos, relative_perfs, xerr=relative_errs, align='center', color=my_cmap(rescale(relative_perfs)))
            if plot_ind==0:
                ax.set_yticks(y_pos, labels=commands)
            else:
                ax.set_yticks([])
            ax.invert_yaxis()
            ax.set_xlim(-0.3, 0.3)
            ax.set_ylim(-0.5, len(relative_perfs)-1+0.5)
            ax.vlines(x=0, ymin=-0.5, ymax=len(relative_perfs)+0.5, lw=1, colors='black')
            # ax.set_xlabel('Effect on Success Rate')
            ax.set_title(ethnicity+gender)
            plot_ind+=1
    #plt.title("Changes in object placement relative neutral placement rate")
    plt.show()

if __name__ == '__main__':
    r_csv=read_clean_R_csv("/home/willie/github/cliport/cliport_quickstart/pairwise_runs_4_11_2022/_plots/coefs.csv")
    cmd_effects_plot(r_csv)
    # If category is ethnicity x gender x cmd
    
        
        
        