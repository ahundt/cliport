"""Ravens main training script."""

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle
import json
import logging
import traceback
from queue import Empty
import torch

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
import multiprocessing as mp
import itertools
from pebble import ProcessPool
import time

# parser = OptionParser()
# # Random seed
# parser.add_option("--seed", type="int", dest="seed", default=0)
# # Description of target item to be used in language prompt
# parser.add_option("--target_item_desc", dest="target_item_desc", default="cube")

# identity_groups=[['W', "M"], ['W', "F"],
#                        ['A', "M"], ['A', "F"],
#                        ['L', "M"], ['L', "F"],
#                        ['B', "M"], ['B', "F"]]

identity_groups=[['W', "M"], ['B', "M"]]

def do_run(seed, command_strs, completed_experiments, i, j, episode, ds, vcfg, dataset_type, mode, record, agent_queue, agent_output_queue, k, id_group_pair_ind):
    if completed_experiments[j, id_group_pair_ind, i]:
        print(f"skipping saved exp {j} {id_group_pair_ind} {i}", flush=True)
        return None
    
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )
    
    current_command_string = command_strs[j]
    
    
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
        id_group_pair=list(itertools.combinations(identity_groups, 2))[id_group_pair_ind]
        task.identity_1 = id_group_pair[0]
        task.identity_2 = id_group_pair[1]
        
    num_command_strs = len(task.command_strs)
    if num_command_strs:
        task.lang_template = task.command_strs[j]
    if vcfg['eval_task']=="packing-unseen-google-objects-race-seq-pairwise":
        task.lang_template = current_command_string

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
    #print("a")
    object_infos = []
    cmd_reward = 0
    for step in range(task.max_steps):
        #act = agent.act(obs, info, goal)
        act_inputs={"obs": obs, "info": info, "goal": goal, "k": k}
        
        agent_queue.put(act_inputs)
        act=agent_output_queue.get()
        lang_goal = info['lang_goal']
#         print(f'Lang Goal: {lang_goal}', flush=True)
        #print("b")
        obs, reward, done, info = env.step(act)
        #print("c")
        object_info = []
        object_info.append(j)
        object_info.append(i)
        object_info.append(current_command_string)
        object_info.append(step)

        if "pose" in info:
            #print("pose in info", info, flush=True)
            for obj_id in info['pose']:
                object_info.append(task.object_log_info[obj_id])
                object_info.append(info['pose'][obj_id])
                object_info.append(info['placed'][obj_id])
                if info['placed'][obj_id]:
                    print("------------PLACED----------------------------------------------------------------------------------------")
            object_infos.append(object_info)
        else:
            #print("pose not in info", info, flush=True)
            object_info.append(None)
            object_info.append({})
            object_info.append({6:0})
#                                 obs, reward, done, info = env.step(act)
        object_info.append(id_group_pair_ind)
        total_reward += reward
        cmd_reward+=(reward>0)
        #print(f'Total Reward: {total_reward:.3f} | Done: {done}\n', flush=True)
        #print("d")
        if done:
            break
    
    return object_infos, cmd_reward, total_reward, info, j, id_group_pair_ind, i

def call_main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    # options, args = parser.parse_args()



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
        #command_strs=[command_strs[vcfg['adjective_sentance_num']]]
        id_exp_name=f"two_{vcfg['command_string_min']}_{vcfg['command_string_max']}"
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
        object_infos, cmd_reward, _ = pickle.load(open(pickle_path, "rb"))

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # Set up inference server queues and thread pool
    print(f"using {vcfg['num_processes']} threads")
    pool = mp.Pool(processes=vcfg['num_processes'], maxtasksperchild=1)
    m = mp.Manager()
    agent_queue=m.Queue()
    agent_output_queues={}

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    model_file = os.path.join(vcfg['model_path'], ckpts_to_eval[0])

    results = []
    mean_reward = 0.0

    # Initialize agent.
    agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

    # Load checkpoint
    agent.load(model_file)
    print(f"Loaded: {model_file}", flush=True)

    record = vcfg['record']['save_video']
    n_demos = int(vcfg['n_demos'])
    # HACK TODO clean up command string iteration, re-enable other tasks
    num_command_strs = len(command_strs)
    num_id_pairs = len(list(itertools.combinations(identity_groups, 2)))
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
    completed_experiments = np.zeros((num_command_strs, len(list(itertools.combinations(identity_groups, 2))), n_demos))
    if object_infos:
        for log in object_infos:
            completed_experiments[log[0], log[-1], log[1]] = 1

    cumulative_rewards={}
    # Run testing and save total rewards with last transition info.        
    for j in trange(vcfg['command_string_min'], vcfg['command_string_max']):
        cmd_reward=0
        num_runs_save_path=wm_save_json+str(j)+".p"
        print(f"running for {n_demos} trials", flush=True)
        for id_group_pair_ind in range(num_id_pairs):
            all_parallel_runs=[]
            for i in trange(0, n_demos):
                k = num_id_pairs*n_demos*j+n_demos*id_group_pair_ind+i
                if not completed_experiments[j, id_group_pair_ind, i]:
                    episode, seed = ds.load(0)
                    seed = k
                    agent_output_queues[k]=m.Queue()
                    print("k", k)
                    run=pool.apply_async(do_run, args=(seed, command_strs, completed_experiments, i, j, episode, ds, vcfg, dataset_type, mode, record, agent_queue, agent_output_queues[k], k, id_group_pair_ind))
                    all_parallel_runs.append(run)

            print("start act loop")
            all_done=False
            total_num_runs=len(all_parallel_runs)
            s_time=time.time()
            all_reward=0
            while not all_done:  
                all_done=True
                new_parallel_runs=[]
                for p_ind in range(len(all_parallel_runs)):
                    p=all_parallel_runs[p_ind]
                    if p.ready():
                        res=p.get()
                        if res!=None:
                            run_object_infos, run_cmd_reward, total_reward, info, j, id_group_pair_ind, i=res
                            all_reward+=total_reward
                            object_infos+=run_object_infos
                            #cumulative_rewards[j][id_group_pair_ind]+=run_cmd_reward
                            results.append((total_reward, info))
                            completed_experiments[j, id_group_pair_ind, i] = 1
                            if len(results)%250==0:
                                print(f"len(results) {len(results)}")
                                df = pd.DataFrame(data=object_infos)
                                df.to_csv(csv_path)
                                pickle.dump((object_infos, cmd_reward, completed_experiments), open(pickle_path, "wb"))                    
                    else:
                        new_parallel_runs.append(p)
                        all_done=False
                all_parallel_runs=new_parallel_runs
                num_sent=0
                while num_sent<20:
                    try:
                        act_inputs=agent_queue.get(timeout=0.02)
                        act = agent.act(act_inputs['obs'], act_inputs['info'], act_inputs['goal'])
                        k=act_inputs['k']
                        print("k", k)
                        agent_output_queues[k].put(act)
                        c_time=time.time()
                        avg_per_it=(c_time-s_time)/max(total_num_runs-len(all_parallel_runs), 1)
                        print(f"{id_group_pair_ind} {len(all_parallel_runs)} of {total_num_runs} {avg_per_it} s/it all_reward {all_reward}", flush=True)
                        num_sent+=1
                    except Empty:
                        break
                
                

    df = pd.DataFrame(data=object_infos)
    df.to_csv(csv_path)
    pickle.dump((object_infos, cmd_reward, completed_experiments), open(pickle_path, "wb"))

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
