## One objective function, one parameter, no parallelization 
# Import block
import numpy as np                            # Usually necessary 
import matplotlib.pyplot as plt               # For visualization 
import nevergrad as ng                        # Canonical import name

# Objective function definition
# This example attempts to arrive at target vector (0.707, 0.707)
target = np.array([0.707, 0.707])             
def objFun(x: list[float]) -> float:          # sum of squared errors
    return np.sum((x - target) ** 2)
  
## Nevergrad code
# Parameterization = dimension of your single objective function input
# Budget = number of iterations to perform 
opt = ng.optimizers.DE(parametrization=2, budget=500)
logger = ng.callbacks.ParametersLogger("./log", append=False) # log loss vals
opt.register_callback("tell",  logger)
res = opt.minimize(objFun)

print("Target: ", target)
print("Result: ", res.value)

# Display loss as a function of budget
log = logger.load()
losses = [dict['#loss'] for dict in log]
plt.figure()
plt.plot(range(500), losses)            
plt.xlabel("Budget")
plt.ylabel("Loss")
