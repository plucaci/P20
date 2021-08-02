import matplotlib.pyplot as plt
import pandas as pd

def training_plots(metrics_filename, title):
    ax = plt.gca()
    df = pd.DataFrame.from_dict(pd.read_pickle(metrics_filename), orient='index')
    df.plot('episode', 'highest_score', ax=ax)
    df.plot('episode', 'rolling_reward', ax=ax)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")
    ax.legend(["Highest Average Score in 100 Episodes", "Average Score in 100 Episodes"])
    plt.title(title)
    plt.show()
    return df