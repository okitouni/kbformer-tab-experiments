
import subprocess
from queue import Queue
from threading import Thread

def runner(device, cmd_base, queue):
    while True:
        data = queue.get()
        if data is None:
            break

        cmd = cmd_base.format(data=data, device=device)
        print("Running on {device}: {cmd}".format(device=device, cmd=cmd))
        subprocess.run(cmd.split())
        queue.task_done()


# Datasets
datasets = [
    "abalone", "adult", "buddy", "california", "cardio",
    "churn2", "default", "diabetes", "fb-comments", "gesture",
    "higgs-small", "house", "insurance", "king", "miniboone", "wilt",
][::-1]

cmd_base = ("python scripts/tune_ddpm.py {data} synthetic catboost kbformer "
            "--device {device} --eval_seeds")

# Fill the queue with datasets
queue = Queue()
for dataset in datasets:
    queue.put(dataset)

# Set up and start threads for both GPUs
threads = []
for i in range(2):
    gpu_id = "cuda:{}".format(i)

    t = Thread(target=runner, args=(gpu_id, cmd_base, queue))
    t.start()
    threads.append(t)

# Wait for the queue to be empty
queue.join()

# Terminate the threads
for _ in range(len(threads)):
    queue.put(None)
for t in threads:
    t.join()