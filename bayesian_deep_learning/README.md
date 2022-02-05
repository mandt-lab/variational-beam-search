# Variational-Beam-Search-for-Transformed-CIFAR10
Artificial covariate shifts by elastic transformations applies to CIFAR10 and SVHN pixels. Our goal is to detect and adapt the covariate shifts quickly. Baselines include Variational Continual Learning and Laplace Propagation

## Usage
The code is written in `python3.7.5`, `tensorflow1.15`, and `tensorflow_probability0.8.0`.

All model entrance function is in the script named starting with "runscript". To run different models, please refer to the following bullet points.

The runscripts are encoded in a hyperparameter search modules. Before running, check the hyperparameter searching space and the output directory, which is the argument `save_dir` of the function `HyperparameterSearch()`, and make sure it exists. 

The searching process utilizes multiprocessing and GPUs. After it finishes, the script will aggregate information from all hyperparameter settings, and print the best setting and all other settings. The evaluated performance is the average test accuracy.

Note that if you want to use customized `params` and `train_and_eval()`, the following format requirements must meet:

```python
# examples
# `params` is a dictionary of (STRING, LIST) pairs, where STRING is the parameter name and LIST is a list of searched values.
params = {
    "param_name1": [...],
    "param_name2": [...],
		...
}

# examples
def train_and_eval(param, save_path):
    """Accept a specific parameter setting, and run the training and evaluation
    procedure. 

    The training and evaluation procedure are specified by users.
    
    The evaluation procedure should return a `scalar` value to indicate the 
    performance of this parameter setting.

    Args:
    param -- A dict object specifying the parameters. It shares the same keys with `params`.
    save_path -- A string object of intermediate result saving path.

    Returns:
    A scalar value for evaluation.
    """
    pass
```

* Run Variational Beam Search:

  ```shell
  # please make sure the `save_dir` exists
  python runscript.py
  ```

* Run variational continual learning: 

  ```shell
  mkdir vcl_res
  python runscript_vcl.py
  ```

* Run Laplace Propagation:
  ```shell
  mkdir lp_res
  python runscript_laplace_propagation.py
  ```

# Plot results

Please refer to the jupyter notebook in this directory.