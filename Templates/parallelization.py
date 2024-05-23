## One objective function, Parallelization 
# Import block
import numpy as np                           # Usually necessary 
import matplotlib.pyplot as plt              # For visualization 
import nevergrad as ng                       # Canonical import name
from concurrent import futures               # You can also use multiprocessing

# Objective function definition
target = np.array([0.707, 0.707])             # target vector for example
def objFun(x: list[float]) -> float:          # sum of squared errors
    return np.sum((x - target) ** 2)

def main():
    ## Nevergrad code
    # Notice num_workers as a new argument
    # You can find your cpu core count using print(os.cpu_count())
    opt = ng.optimizers.DE(parametrization=2, budget=500, num_workers=20)
    logger = ng.callbacks.ParametersLogger("./log", append=False)
    opt.register_callback("tell",  logger)
    # This line tells minimize to use the cpu-bound process executor
    # Consider using ThreadPoolExecutor if your algorithm is IO bound
    with futures.ProcessPoolExecutor(max_workers=opt.num_workers) as executor:
        res = opt.minimize(objFun, executor=executor, batch_mode=True)
    
    print("Target: ", target)
    print("Result: ",res.value)
    
    # Display loss as a function of budget
    log = logger.load()
    losses = [dict['#loss'] for dict in log]
    plt.figure()
    plt.plot(range(500), losses)
    plt.xlabel("Budget")
    plt.ylabel("Loss")

if __name__ == '__main__':
    main()
