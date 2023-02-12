import os
from multiprocessing import Pool, current_process, Queue
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

output_dir = "output"
task = 'sst2'
process_list = []
for seed in range(25):
    log_dir = os.path.join(output_dir, f"log/{task}/seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task}.txt")

    temp = f"python3 " \
        f"run_glue.py " \
        f"--model_name_or_path google/multiberts-seed_{seed} " \
        f"--task_name {task} " \
        f"--do_train " \
        f"--do_eval " \
        f"--learning_rate 2e-5 " \
        f"--max_seq_length 128 "\
        f"--num_train_epochs {task_to_keys[task]} " \
        f"--output_dir {output_dir}/{task}/seed_{seed} " \
        f"--load_best_model_at_end " \
        f"--save_total_limit 1 " \
        f"--save_strategy \"no\" " \
        f"--fp16" \
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