import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyddm
import pyddm.plot

model = pyddm.gddm(drift=0.5, noise=1.0, bound=0.6, starting_position=0.3, nondecision=0.2)

model.show()

# create solution object --> represents probability distribution function over time for choices associated with upper and lower bound crossings
sol = model.solve()

# if we want to fit some of the parameters, give ranges instead of concrete values
model_to_fit = pyddm.gddm(drift="d", noise=1.0, bound="B", nondecision=0.2, starting_position="x0",
                          parameters={"d": (-2,2), "B": (0.3, 2), "x0": (-.8, .8)})
model_to_fit.show()

# Before fitting the model, visualize it using model GUI to make sure it behaves as expected
pyddm.plot.model_gui(model_to_fit)

# --- PARAM RECOV ---
# use solution object to simulate data
samp_simulated = sol.sample(10000)

# fit model to simulated data
model_to_fit.fit(samp_simulated, lossfunction=pyddm.LossBIC, verbose=False)
model_to_fit.show()
