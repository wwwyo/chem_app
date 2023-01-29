import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = "DejaVu Serif"

class Graph():
    def __init__(self, obs: pd.Series, pred: pd.Series, title: str):
        self.obs = obs
        self.pred = pred
        self.title = title

    def yyplot(self):
        yvalues = np.concatenate([self.obs, self.pred])
        ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(self.obs, self.pred)
        plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
        plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
        plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
        plt.xlabel('Real Y', fontsize=24)
        plt.ylabel('Predicted Y', fontsize=24)
        plt.title(self.title, fontsize=24)
        plt.tick_params(labelsize=16)
        return fig
