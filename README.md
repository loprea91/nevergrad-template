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
## One objective function, no parallelization 
import numpy as np                            # Usually necessary 
import matplotlib.pyplot as plt               # For visualization 
import nevergrad as ng                        # Canonical import name
```
Next we define our objective/loss/cost function to minimize. You have a lot of freedom here, in this case its just the sum of squared errors between a `target` vector [0.707, 0.707] and the optimizer's solution.

```
target = np.array([0.707, 0.707])
def objFun(x: list[float]) -> float:
    return np.sum((x - target) ** 2)
```
**Note:** nevergrad makes use of Python's typing syntax in function definitions. You DO NOT have to do this, but it's good practice. If you haven't used it before, the syntax is: `def funName(varName: varType) -> funOutputType` and the types are as expected (e.g. int, bool, float, str). If you have a list the syntax is simply `list[varType]`. For more on typing see [this documentatio](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

You may wish to use the distance to target distribution (e.g. experimental data) as your objective. In this case consider using a call to Kullback-Leibler divergence (`scipy.special.kl_div`) or Earth-movers/Wasserstein distance (`scipy.stats.wasserstein_distance`).

Next we create the optimizer object.
`opt = ng.optimizers.DE(parametrization=4, budget=500)`
 * DE: the name of the optimizer, DE is differential evolution. A good 
 *Parametrization: the size of your objective function input, in our case 2.
 *Budget: the number of interation of the optimizer algorithm

</details>
<details>
<summary>One objective function + Parallel computation</summary>
m
</details>
<details>
<summary>Multiple objective functions + Parallel computation</summary>
m
</details>
