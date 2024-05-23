## One objective function with multiple parameters of different types
# Import block
import numpy as np                            # Usually necessary 
import matplotlib.pyplot as plt               # For visualization 
import nevergrad as ng                        # Canonical import name

# Objective function definition
# This example attempts to approximate pi using 2 integers. The keyword argument
# is just a placeholder here, but it could be used in your code in some way
def objFun(a: int, b: int, c: str) -> float:          # difference from pi
    return abs(a//b - np.pi)

# We first define all the attributes of each parameter, then combine them into
# one instrumentation package. We could have also combined them with p.tuple or
# p.dict
a = ng.p.Scalar(50, lower=10, upper=100).set_integer_casting()
b = ng.p.Scalar(5, lower=1, upper=10).set_integer_casting()
instrum = ng.p.Instrumentation(a, b, c="not_used")
  
## Nevergrad code
opt = ng.optimizers.DE(parametrization=instrum, budget=500)
logger = ng.callbacks.ParametersLogger("./log", append=False)
opt.register_callback("tell",  logger)
res = opt.minimize(objFun)

print("Target: ", np.pi)
print("Result: ",res.value[0][0],"/",res.value[0][1],"=",
      res.value[0][0]/res.value[0][1])

# Display loss as a function of budget
log = logger.load()
losses = [dict['#loss'] for dict in log]
plt.figure()
plt.plot(range(500), losses)
plt.xlabel("Budget")
plt.ylabel("Loss")
