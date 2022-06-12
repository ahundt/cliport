"""Ravens main training script."""

import os
import pickle
import json
import logging
import traceback

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.utils import command_strings
from cliport.environments.environment import Environment
from optparse import OptionParser
import pandas as pd
from tqdm import tqdm, trange
import torch

# parser = OptionParser()
# # Random seed
# parser.add_option("--seed", type="int", dest="seed", default=0)
# # Description of target item to be used in language prompt
# parser.add_option("--target_item_desc", dest="target_item_desc", default="cube")

identity_groups=[['W', "M"], ['W', "F"],
                       ['A', "M"], ['A', "F"],
                       ['L', "M"], ['L', "F"],
                       ['B', "M"], ['B', "F"]]

def call_main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    # options, args = parser.parse_args()

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    start_seed = vcfg['seed']
    target_item_desc = vcfg['target_item_desc']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=eval_task,
                                            mode=mode,
                                            n_demos=100,
                                            augment=False)
    # elif mode is not 'test':
    else:
        # TODO HACK test mode disabled for race experiments, make code work properly
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=100,
                                   augment=False)

    all_results = {}
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])
    
    
    command_strs = command_strings.build_command_strings('reduced')
    if vcfg['eval_task']=="packing-unseen-google-objects-race-seq-pairwise":
        command_strs=[command_strs[vcfg['adjective_sentance_num']]]
        id_exp_name=f"one-{command_strs[0]}_{identity_groups[vcfg['identity_group_num_0']][0]}_{identity_groups[vcfg['identity_group_num_0']][1]}_{identity_groups[vcfg['identity_group_num_1']][0]}_{identity_groups[vcfg['identity_group_num_1']][1]}"
        wm_id_exp_name=f"one-{command_strs[0]}_W_M_W_M"
    else:
        id_exp_name=""
        

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    folds = int(vcfg['folds'])
    fold = int(vcfg['fold'])
    json_str = f'{name}-{json_name}-{id_exp_name}'
    wm_json_str = f'{name}-{json_name}-{wm_id_exp_name}'
    # save separate log files when running multiple processes that complete different parts of the same experiment
    if folds > 0:
        json_str = f'fold{fold}-' + json_str
    save_json = os.path.join(save_path, json_str)
    wm_save_json = os.path.join(save_path, wm_json_str)

    # Load existing results.
    existing_results = {}
    object_infos = []
#     if os.path.exists(save_json):
#         print('Found exiting results in file: ' + str(save_json))
#         with open(save_json, 'r') as f:
#             existing_results = json.load(f)

    # Load existing log.
    pickle_path = save_json+f"_{id_exp_name}"+".p"
    csv_path = save_json+f"_{id_exp_name}"+".csv"
    if os.path.exists(pickle_path):
        object_infos, cmd_reward = pickle.load(open(pickle_path, "rb"))

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg['model_path'], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg['update_results'] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

            # Load checkpoint
            agent.load(model_file)
            print(f"Loaded: {model_file}", flush=True)

            record = vcfg['record']['save_video']
            n_demos = int(vcfg['n_demos'])
            # HACK TODO clean up command string iteration, re-enable other tasks
            num_command_strs = len(command_strs)
            # n_demos_per_command = n_demos
            # n_demos = n_demos * num_command_strs
            command_string_min = 0
            command_string_max = max(num_command_strs, 1)
            num_strings_in_fold = num_command_strs
            if folds > 0:
                num_strings_in_fold = int(np.ceil(float(num_command_strs) / float(folds)))
                command_string_min = int(num_strings_in_fold * fold)
                command_string_max = int(min(num_strings_in_fold * (fold + 1), num_command_strs))

            # j indexes command_string_max, i indexes n_demos
            completed_experiments = np.zeros((command_string_max, n_demos))
            if object_infos:
                for log in object_infos:
                    completed_experiments[log[0], log[1]] = 1

            # Run testing and save total rewards with last transition info.        
            for j in trange(command_string_min, command_string_max):
                cmd_reward=0
                num_runs_save_path=wm_save_json+str(j)+".p"
#                 if vcfg['eval_task']=="packing-unseen-google-objects-race-seq-pairwise" and vcfg['identity_group_num_0']!=0:
#                     n_demos=pickle.load(open(num_runs_save_path, "rb"))
#                     n_demos=min(n_demos, 2500)
                print(f"running for {n_demos} trials", flush=True)
                for i in trange(0, n_demos):
                    try:
                        k = (j+1) * (i+1) + i
                        episode, seed = ds.load(0)
                        seed = k
    #                     if vcfg['eval_task']=="packing-unseen-google-objects-race-seq-pairwise":
    #                         seed=k+n_demos*((vcfg['adjective_sentance_num']+1)*(vcfg['identity_group_num']+1)+vcfg['identity_group_num'])
                        
                        np.random.seed(seed)
                        current_command_string = command_strs[j]
                        if completed_experiments[j, i]:
                            print(f"skipping saved exp {j} {i}", flush=True)
                            continue
                        
                        print(f"cmd str {current_command_string}", flush=True)
    
                        goal = episode[-1]
                        total_reward = 0
    
                        # set task
                        if 'multi' in dataset_type:
                            task_name = ds.get_curr_task()
                            task = tasks.names[task_name](current_command_string)
                            print(f'Evaluating on {task_name}', flush=True)
                        else:
                            task_name = vcfg['eval_task']
                            task = tasks.names[task_name]()
    
                        task.mode = mode
                        if vcfg['eval_task']=="packing-unseen-google-objects-race-seq-pairwise":
                            task.identity_1 = identity_groups[vcfg['identity_group_num_0']]
                            task.identity_2 = identity_groups[vcfg['identity_group_num_1']]
                            
                        num_command_strs = len(task.command_strs)
                        if num_command_strs:
                            task.lang_template = task.command_strs[j]
                        if vcfg['eval_task']=="packing-unseen-google-objects-race-seq-pairwise":
                            task.lang_template = current_command_string
    
                        agent=agent.float()
                        # env.seed(np.random.randint(2**32-1))
                        env.seed(seed)
                        env.set_task(task)
                        obs = env.reset()
                        info = env.info
                        reward = 0
    
                        # Start recording video (NOTE: super slow)
                        if record:
                            video_name = f'{task_name}-{k+1:06d}_desc-{current_command_string}'
                            if 'multi' in vcfg['model_task']:
                                video_name = f"{vcfg['model_task']}-{video_name}"
    
                            env.start_rec(video_name)
    
                        for step in range(task.max_steps):
                            with torch.no_grad():
                                act = agent.act(obs, info, goal)
                            lang_goal = info['lang_goal']
                            print(f'Lang Goal: {lang_goal}', flush=True)
                            obs, reward, done, info = env.step(act)
    
                            object_info = []
                            object_info.append(j)
                            object_info.append(i)
                            object_info.append(current_command_string)
                            object_info.append(step)
    
                            if "pose" in info:
                                for obj_id in info['pose']:
                                    object_info.append(task.object_log_info[obj_id])
                                    object_info.append(info['pose'][obj_id])
                                    object_info.append(info['placed'][obj_id])
                                object_infos.append(object_info)
                            else:
                                print("pose not in info", info, flush=True)
                                object_info.append(task.object_log_info[obj_id])
                                object_info.append({})
                                object_info.append({6:0})
#                                 obs, reward, done, info = env.step(act)
    
                            total_reward += reward
                            cmd_reward+=(reward>0)
                            print(f'Total Reward: {total_reward:.3f} | Done: {done}\n', flush=True)
                            if done:
                                break
    
                        
                        if i%50==0:
                            df = pd.DataFrame(data=object_infos)
                            df.to_csv(csv_path)
                            pickle.dump((object_infos, cmd_reward), open(pickle_path, "wb"))
    
                        results.append((total_reward, info))
                        mean_reward = np.mean([r for r, i in results])
                        print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}', flush=True)
    
                        # End recording video
                        if record:
                            env.end_rec()
                        completed_experiments[j, i] = 1
                        
                        print("cumulative reward: "+str(cmd_reward), flush=True)
                        # Break after certain # WM places
#                         if cmd_reward>=500:# and vcfg['identity_group_num_0']==0:
#                             print("cumulative reward exceeded, cumulative reward: "+str(cmd_reward), flush=True)
# #                             pickle.dump(i, open(num_runs_save_path, "wb"))
#
#                             df = pd.DataFrame(data=object_infos)
#                             df.to_csv(csv_path)
#                             pickle.dump((object_infos, cmd_reward), open(pickle_path, "wb"))
#
#                             break
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        

                all_results[ckpt] = {
                    'episodes': results,
                    'mean_reward': mean_reward,
                }

            df = pd.DataFrame(data=object_infos)
            df.to_csv(csv_path)
            pickle.dump((object_infos, cmd_reward), open(pickle_path, "wb"))
        # Save results in a json file.
        if vcfg['save_results']:

            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, 'w') as f:
                json.dump(all_results, f, indent=4)


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        print('looking for checkpoints excluding those found in the existing results json in the directory: ' + vcfg['model_path'])
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        print('looking for checkpoints in: ' + vcfg['model_path'])
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    if not ckpts_to_eval:
        raise FileNotFoundError('Could not find any model checkpoint files to load, check the folders specified and the eval.py function list_ckpts_to_eval().')

    return ckpts_to_eval

@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):
    call_main(vcfg)

if __name__ == '__main__':
    main()
