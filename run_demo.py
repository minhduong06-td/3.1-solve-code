import numpy as np
from abc import ArtificialBeeColony 


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return A * x.size + np.sum(x**2 - A * np.cos(2.0 * np.pi * x))


if __name__ == "__main__":
    D = 10
    lower = -5.12 * np.ones(D)
    upper =  5.12 * np.ones(D)

    sn = 25        
    limit = 100      
    max_cycles = 200 
    seed = 42

    abc = ArtificialBeeColony(
        obj_func=rastrigin,
        bounds=(lower, upper),
        sn=sn,
        limit=limit,
        max_cycles=max_cycles,
        seed=seed,
    )
    result = abc.run()


    print("=== ABC Demo on Rastrigin ===")
    print(f"Best f: {result.f_best:.6f}")
    print(f"Best x (first 5 dims): {result.x_best[:5]}")
    print(f"History length: {len(result.history['f_best_per_cycle'])} cycles")
