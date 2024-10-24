import pandas as pd
import numpy as np


def vote_soft(models, X_test):
    predictions = np.zeros(X_test.shape[0])
    for model in models:
        predictions += model.predict(X_test) / len(models)

    return predictions
