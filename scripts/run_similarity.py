import os
from multiprocessing import Pool, current_process, Queue
# glue_tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']
# model_names = ["bert-base-uncased","bert-large-uncased"]
# glue_tasks = ['qnli', 'qqp', 'sst2']
glue_tasks = ['mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
task_to_keys = {
    "cola": 3,
    "mnli": 3,
    "mrpc": 5,
    "qnli": 3,
    "qqp": 3,
    "rte": 3,
    "sst2": 3,
    "stsb": 3
}
queue = Queue()

def distribute(process_command):
    gpu_id = queue.get()
    try:
        # run processing on GPU <gpu_id>
        ident = current_process().ident
        # ... process filename
        command = "CUDA_VISIBLE_DEVICES=" + str(gpu_id) + " " + process_command
        print('{}: starting process on GPU {}, {}'.format(ident, gpu_id, command))
        #print(command)
        os.system(command)
        print('{}: finished'.format(ident))
    finally:
        queue.put(gpu_id)

output_dir = "output/caillen"
# task = 'mnli'
flow_model_state = 'train'
process_list = []
# mode = 1
example_num = 50
for task in ['mnli']:
    for mode in [6]:
        for seed_1 in range(25):
            for seed_2 in range(seed_1 + 1):
                if seed_1 == seed_2:
                    continue
                log_dir = os.path.join(output_dir, f"log/{task}/{flow_model_state}/mode_{mode}/seed_{seed_1}_to_{seed_2}")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{task}.txt")

                temp = f"python3 " \
                    f"run_mutual_similarity.py " \
                    f"--model_name_or_path google/multiberts-seed_0 " \
                    f"--task_name {task} " \
                    f"--do_train " \
                    f"--learning_rate 2e-5 " \
                    f"--max_seq_length 128 "\
                    f"--per_device_train_batch_size 32 " \
                    f"--num_train_epochs {task_to_keys[task]} " \
                    f"--output_dir {output_dir}/{task}/seed_{seed_1}_to_{seed_2} " \
                    f"--method {mode} " \
                    f"--example_num {example_num} " \
                    f"--overwrite_output_dir " \
                    f"--seed_original {seed_1} " \
                    f"--seed_compare {seed_2} " \
                    f"> {log_file} 2>&1"
                print(temp)
                process_list.append(temp)

PROC_PER_GPU = 1
NUM_GPUS = 8
# initialize the queue with the GPU ids
for _ in range(PROC_PER_GPU):
  for gpu_ids in range(NUM_GPUS):
    queue.put(gpu_ids)

pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)

for _ in pool.imap_unordered(distribute, process_list):
    pass
pool.close()
pool.join()