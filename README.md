# Nevergrad tutorial 
## Intro
Nevergrad is a gradient-free optimization toolbox created by Facebook Research, and written in Python. At the moment it contains 269 optimizers, and is being actively updated with new ones.

The largest advantage is that you can very easily add it to any existing code!

Check out their [github](https://github.com/facebookresearch/nevergrad) and [documentation](https://facebookresearch.github.io/nevergrad/).

\- Lawrence

## Installation
You can use nevergrad with any Python 3.6+ kernel.

`pip install nevergrad` or `conda install -c conda-forge nevergrad`

I haven't noticed any peculiarities between Windows or Linux installations.

## How to use
I'm going to separate the templates from the simplest use case to the most complicated. Going through them should give you a good idea how to customize the code to your needs. The templates are fully runable, containing test examples.
  
<details>
<summary>One objective function</summary>
Import block is self-explanatory.
  
```
import numpy as np                            
import matplotlib.pyplot as plt               
import nevergrad as ng                        
```
Next we define our objective/loss/cost function to minimize. You have **a lot** of freedom here, in this case its just the sum of squared errors between a `target` vector and the optimizer's solution. If you need to set upper/lower bounds, or have more complicated parameter needs, please see the next section of the tutorial after reading this one.

```
def objFun(x: list[float]) -> float:
    return np.sum((x - target) ** 2)
```

<details>
<summary> Notes on function definition </summary>
  
* Nevergrad makes use of Python's typing syntax in function definitions. You DO NOT have to do this, but it's good practice. If you haven't used it before, the syntax is: `def funName(varName: varType) -> funOutputType` and the types are as expected (e.g. int, bool, float, str). If you have a list the syntax is simply `list[varType]`. For more on typing see [this documentatio](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

* You may wish to use the distance to target distribution (e.g. experimental data) as your objective. In this case consider using a call to Kullback-Leibler divergence (`scipy.special.kl_div`) or Earth-movers/Wasserstein distance (`scipy.stats.wasserstein_distance`).

* You may need to turn a scalar function into one that accepts vectors as inputs. You can do this with numpy using `objFunVectorized = np.vectorize(objFun)`, for example. You can also prevent arguments of your choice from being vectorized (see [docs](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html)).

</details>

Next we create the optimizer object.

`opt = ng.optimizers.optimizerName(parametrization, budget)`

 * optimizerName: a good general purpose optimizer is NGOpt, an adaptive algorithm. You can find the whole list of optimizers by running `ng.optimizers.registry.keys()`.
 * Parametrization: the size of your objective function input. For example, a vector with two elements has parametrization = 2.
 * Budget: the total number of iterations that the optimizer algorithm will run

Then we use the automated logger to record all parameter sets tested and their loss values to a file.

```
logger = ng.callbacks.ParametersLogger("./log", append=False)     # create logger object
opt.register_callback("tell",  logger)                            # set it to obtain 'tell's of details on every iteration
```
Then we run the minimization, feeding it the objective function as an argument.

`results = opt.minimize(objFun)`

The field results.value is our recommended parameter set!

Finally, we will plot the loss as a function of budget using the code block. Note that the loss file we saved above is a list of dictionaries, each containing info about one iteration.

```
log = logger.load()
losses = [dict['#loss'] for dict in log]
plt.figure()
plt.plot(range(500), losses)
plt.xlabel("Budget")
plt.ylabel("Loss")
```
![Loss](/Assets/loss_1.png)
</details>

<details>
<summary> Complex parameters </summary>
  
Typically our objective function takes more than one parameter, and/or the parameters are of different types. Note this is **NOT** the same thing as multiple objective functions (which must be optimized at the same time), we deal with this case in the last section of the tutorial.

We have several parameter types to choose from:

|Parameter type|Description|
| --- | --- |
|nevergrad.p.Instrumentation|A container to ease the use of multi-parameter functions (example below).|
|nevergrad.p.Array|An array of any value type.|
|nevergrad.p.Scalar|Parameter representing a scalar.|
|nevergrad.p.Log|Positive log-scaled variable.|
|nevergrad.p.Dict|A parameter that holds other parameters. The keys are the parameter names.|
|nevergrad.p.Tuple|A tuple, that may contain other parameters as elements.|
|nevergrad.p.Choice|Random choice of list of categorical options.|
|nevergrad.p.TransitionChoice|Basically a discrete-time Markov chain, a list of values and transition weights.|

In general the parameters have arguments like:
* init (optional float) – initial value
* lower (optional float) – minimum value
* upper (optional float) – maximum value

You can force integers by using `.set_integer_casting()` on the parameter object. Note: internally, all nevergrad parameters are centered and scaled to $\mathcal{N}(0, 1)$.

An example of a parameter definition of a 2D array of possible integer values from 0 to 10:
```
param = ng.p.Array(init=(5,5), lower=(0,0), upper=(10,10)).set_integer_casting()
```

We can combine parameters together using instrumentations (useful when we have keywords, tuples, or dicts), to produce multi-parameter functions. Example from the template:

```
a = ng.p.Scalar(50, lower=1, upper=100).set_integer_casting()
b = ng.p.Scalar(50, lower=1, upper=100).set_integer_casting()
instrum = ng.p.Instrumentation(a, b, c="not_used")
opt = ng.optimizers.DE(parametrization=instrum, budget=500)
```

The output of which will be a tuple, where the first element is a tuple of regular arguments, and the second is a dictionary of keyword arguments, e.g.

```
((a_value, b_value), {'c': 'not_used'})
```

</details>

<details>
<summary> Parallelization </summary>
To implement concurrent computing we make a few simple changes to the basic script. First, we need a parallel computing package: we can either use `multiprocessing` or `concurrent`, the latter being a bit simpler to use.

```
from concurrent import futures
```
Second, we have to encapsulate the code in an `main()` function, and add the common boilerplate at the bottom:

```
if __name__ == '__main__':
    main()
```
Next, we add the number of workers we want as an argument to the optimizer object creation. Note: you can find your cpu core count by running `print(os.cpu_count())`.

```
opt = ng.optimizers.DE(parametrization=2, budget=500, num_workers=20)
```

Finally, we run the minimization using a ProcessPoolExecutor. This was chosen because our code is cpu-bound; it's possible that your algorithm may be IO (input/output bound), if it needs to perform costly read or write operations. In this latter case, replace `ProcessPoolExecutor` with `ThreadPoolExecutor`.

```
 with futures.ProcessPoolExecutor(max_workers=opt.num_workers) as executor:
        res = opt.minimize(objFun, executor=executor, batch_mode=True)
```
The argument `batch_mode=True` means that `num_workers` evaluations are launched, and only when all are finished does another batch get launched. This is currently recommended in the docs.

</details>

<details>
<summary> Multiple objective functions </summary>
Sometimes you may want to optimize two different metrics at the same time. Of course, it is often the case that both metrics are NOT fully optimal for the same parameter values. In the language of multi-objective optimization problems (MOOs), the goal is the indentification of the Pareto front:
  
![Pareto](/Assets/pareto.jpg)

The solutions along the Pareto front *cannot be improved without DECREASING the optimality of one of the objective functions*. Nevegrad will return the solution closest to the origin (in some n-dimensional loss space).

ONLY THE DE OPTIMIZER FAMILY IS ACTUALLY DOING THIS CALCULATION. The other optimizers are not yet fully implementing MOOs.

We define out MOO function with two return values in a list:
```
def multiobjective(a: int, b: int, c: int ) -> float:
    return [abs(a/b - np.pi), abs(c/b - np.e)]
```
In this example we're looking for 3 integers, such that (a/b) is close to pi, and (c/b) is close to e. 

We define the parameters as in the multi-parameter template.

```
a = ng.p.Scalar(25, lower=1, upper=50).set_integer_casting()
b = ng.p.Scalar(25, lower=1, upper=50).set_integer_casting()
c = ng.p.Scalar(25, lower=1, upper=50).set_integer_casting()
instrum = ng.p.Instrumentation(a, b, c)
```
The optimizer is constructed normally afterward.
```
opt = ng.optimizers.DE(parametrization=instrum, budget=400)
opt.minimize(multiobjective)
```
The last bit of code in the template displays the Pareto front

```
vals = [str(pfm.value[0]) for pfm in opt.pareto_front()]
losses = np.vstack([pfm.losses for pfm in opt.pareto_front()])
# Plot the Pareto front    
plt.scatter(losses[:,0], losses[:,1]) 
for i, val in enumerate(vals):
    plt.text(losses[i,0], losses[i,1], val, fontsize=6)
```
![front](/Assets/front_example.png)

</details>

## Further reading 

Decent article on when you'd want to use gradient-free methods [here](https://openmdao.github.io/PracticalMDO/Notebooks/Optimization/when_to_use_gradient_free_methods.html).

See also Sam's excellent example script [here](https://github.com/mullerlab/NETSIM-dev/blob/master/analysis/python/test_nevergrad.py).
