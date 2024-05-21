# Nevergrad tutorial 
## Intro
Nevergrad is a gradient-free optimization toolbox created by Facebook Research, and written in Python. At the moment it contains 269 optimizers, and is being actively updated with new ones.

Check out their [github](https://github.com/facebookresearch/nevergrad) and [documentation](https://facebookresearch.github.io/nevergrad/).

\- Lawrence
## Installation
You can use nevergrad with any Python 3.6+ kernel.
`pip install nevergrad` or `conda install -c conda-forge nevergrad`
I haven't noticed any peculiarities between Windows or Linux installations.

## How to use
I'm going to separate the templates from the simplest use case to the most complicated. Going through them should give you a good idea how to customize the code to your needs. 
  
<details>
<summary>One objective function</summary>
Import block is self-explanatory.
  
```
import numpy as np                            
import matplotlib.pyplot as plt               
import nevergrad as ng                        
```
Next we define our objective/loss/cost function to minimize. You have a lot of freedom here, in this case its just the sum of squared errors between a `target` vector and the optimizer's solution.

```
def objFun(x: list[float]) -> float:
    return np.sum((x - target) ** 2)
```
**Note:** nevergrad makes use of Python's typing syntax in function definitions. You DO NOT have to do this, but it's good practice. If you haven't used it before, the syntax is: `def funName(varName: varType) -> funOutputType` and the types are as expected (e.g. int, bool, float, str). If you have a list the syntax is simply `list[varType]`. For more on typing see [this documentatio](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

**Note:** you may wish to use the distance to target distribution (e.g. experimental data) as your objective. In this case consider using a call to Kullback-Leibler divergence (`scipy.special.kl_div`) or Earth-movers/Wasserstein distance (`scipy.stats.wasserstein_distance`).

**Note:** you may need to make a scalar function that accepts vectors as inputs. You can do this with numpy by writing `objFunVectorized = np.vectorize(objFun)`.

Next we create the optimizer object.

`opt = ng.optimizers.optimizerName(parametrization, budget)`

 * optimizerName: a good general purpose optimizer is NGOpt, an adaptive algorithm. You can find the whole list of optimizers by running `ng.optimizers.registry.keys()`.
 * Parametrization: the size of your objective function input
 * Budget: the number of interation of the optimizer algorithm

Then we use the automated logger to record all parameter sets tested and their loss values to a file, which we will read later.

```
logger = ng.callbacks.ParametersLogger("./log", append=False)     # create logger object
opt.register_callback("tell",  logger)                            # set it to obtain 'tell's of details on every iteration
```
Then we run the minimization, feeding it the objective function as an argument 
`results = opt.minimize(objFun)`


</details>

<details>
<summary> Objective function with multiple parameter types </summary>
m
</details>

<details>
<summary> Parallelization </summary>
m
</details>

<details>
<summary> Multiple objective functions </summary>
m
</details>
