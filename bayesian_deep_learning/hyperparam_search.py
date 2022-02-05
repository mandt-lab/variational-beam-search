import os
import sys
from datetime import datetime
import time
import pickle
import abc
import multiprocessing
from multiprocessing import Pool
from itertools import product
from pprint import pprint

def print_log(*args):
    print("[{}]".format(datetime.now()), *args)
    sys.stdout.flush()

# example
def train_and_eval(param, save_path):
    """Accept a specific parameter setting, and run the training and evaluation
    procedure. 

    The training and evaluation procedure are specified by users.
    
    The evaluation procedure should return a `scalar` value to indicate the 
    performance of this parameter setting.

    Args:
    param -- A dict object specifying the parameters.
    save_path -- A string object of intermediate result saving path.

    Returns:
    A scalar value of evaluation.
    """
    pass


class HyperparameterSearch:
    """Multiprocessing hyperparameter search. It also enables multiple GPU
    usage.

    One worker process is assigned to one GPU. Worker-GPU mapping can be 
    many-to-many.
    """
    def __init__(self, num_worker=1, first_gpu=0, num_gpu=1,
                 save_dir=""):
        """
        Args:
        num_worker -- The number of subprocesses in use. An integer larger than 
            one.
        first_gpu -- The id of the first gpu in a consequtive sequence of gpus 
            in use.
        num_gpu -- The number of a consequtive sequence of gpus 
            in use. Specify 0 to indicate not using gpus.
        save_dir -- Save root path.
        """
        assert num_worker >= 1 and isinstance(num_worker, int)
        self.num_worker = num_worker
        assert num_gpu >= 0 and isinstance(num_gpu, int)
        self.num_gpu = num_gpu
        if self.num_gpu > 0:
            assert first_gpu >= 0 and isinstance(first_gpu, int) # gpu id
            self.first_gpu = first_gpu
        dt_now_str = datetime.now().strftime("d%Y%m%dt%H%M%S")
        if save_dir:
            self.save_dir = save_dir + '/' + "HyperSearch" + dt_now_str + '/'
        else:
            self.save_dir = "./" + "HyperSearch" + dt_now_str + '/'
        os.mkdir(self.save_dir)
        print_log(
            f"{self.num_worker} worker processes. "
            f"{self.num_gpu} GPUs in use. "
            f"Logs will be saved to {self.save_dir}.")

    def param_to_foldername(self, param):
        """Transfer a dict parameter setting to a string.
        """
        assert isinstance(param, dict)
        to_str = ""
        for k, v in param.items():
            to_str += (str(k) + str(v) + '_')
        to_str = to_str.strip('_')
        return to_str

    def _worker(self, param):
        """Assocaite one gpu to a worker if possible.
        """
        sys.stdout = sys.__stdout__
        if self.num_gpu > 0:
            # set visible gpus
            proc = multiprocessing.current_process()
            proc_id = int(proc.name.split('-')[-1]) # id in the proc group
            # each process assumes one specific gpu
            gpu_id = self.first_gpu + proc_id % self.num_gpu 
            os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id
            print(f"Process-{proc_id} GPU-{gpu_id} contains:")
            pprint(param)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print(f"Process-{proc_id} contains:")
            pprint(param)

        param_str = self.param_to_foldername(param)
        folder_name = self.save_dir + param_str + '/'
        os.mkdir(folder_name)
        sys.stdout = open(
            folder_name + "log.txt", 'w') # log file
        pprint(param)
        sys.stdout.flush()

        if hasattr(self, "train_and_eval"):
            start_time = time.time()
            eval_res = self.train_and_eval(param, folder_name)
            end_time = time.time()
            assert isinstance(eval_res, float)
        else:
            raise NameError(
                "train_and_eval() is not specified. "
                "Run hyperparameter_optimization() to add in.")
        return eval_res, end_time - start_time

    def hyperparameter_optimization(self, train_and_eval, params):
        """Perform hyperparameter searching in the space defined by `params`. 
        
        Args:
        train_and_eval -- A function taking a specific parameter setting, and 
            run the training and evaluation procedure. 

            The training and evaluation procedure are specified by users.
            
            The evaluation procedure should return a `scalar` value to indicate 
            the performance of this parameter setting.

            Args:
            param -- A dict object specifying the parameters.
            save_path -- A string object of intermediate result saving path.

            Returns:
            A scalar value of evaluation.

        params -- A dict object defining the parameter space to search the
            optimal value. Each item must be iteratable and has the format:
                {
                 "param_1": [N, ...], 
                 "param_2": [N, ...],
                 ...
                }

        Returns:
        A list of dict object that contains the best configuration (note that 
            a repetition of the same values might happen);
        A list of dict object that contains the parameter setting;
        A list of tuples (evluation result, elapsed time) that corresponds to
        the parameter setting.
        """
        self.train_and_eval = train_and_eval # add for later usage

        pid = os.getpid()
        print(f"Ancestor process: {pid}")
        print("Parameter settings to be searched:")
        pprint(params, width=40)
        print_log("Search starts.")
        possible_combs = product(*params.values())
        # this format is required for multiprocessing
        params_list = [
            dict(zip(params.keys(), c))
                for c in possible_combs]

        with Pool(self.num_worker) as p:
            res_list = p.map(self._worker, params_list)
            p.close()

        eval_res_list = [i for (i, j) in res_list]
        elapsed_time_list = [j for (i, j) in res_list]
        
        best_eval_res = max(eval_res_list)
        best_args = [
            i for (i, res) in enumerate(eval_res_list) 
                if best_eval_res == res]
        best_param_list = [params_list[i] for i in best_args]
        # summary
        print_log(
            "Search finishes. "
            f"Best parameter setting with evaluation result {best_eval_res}:")
        pprint(best_param_list, width=40)
        print("\nOther settings")
        for p, r, t in zip(params_list, eval_res_list, elapsed_time_list):
            print('-'*78)
            pprint(p, width=80)
            print(f"Evaluation result: {r}")
            print(f"Elapsed time {t}")
        return best_param_list, params_list, res_list


# tools for finding the folder name
def product_params(params):
    possible_combs = product(*params.values())
    # this format is required for multiprocessing
    params_list = [
        dict(zip(params.keys(), c))
            for c in possible_combs]
    return params_list

def params_to_str(params):
    params_list = product_params(params)
    param_strs = []
    for param in params_list:
        assert isinstance(param, dict)
        to_str = ""
        for k, v in param.items():
            to_str += (str(k) + str(v) + '_')
        to_str = to_str.strip('_')
        param_strs.append(to_str)
    return param_strs


if __name__ == '__main__':
    # example
    params = {
        "lr": [0.001, 0.01, 0.1],
        "dim": [20, 50, 100],
    }

    def train_and_eval(param, save_path):
        print(param["lr"], param["dim"])
        return param["lr"]

    hs = HyperparameterSearch(num_worker=1, first_gpu=0, num_gpu=8)
    best_param_list, params_list, res_list = hs.hyperparameter_optimization(
        train_and_eval, params)
    