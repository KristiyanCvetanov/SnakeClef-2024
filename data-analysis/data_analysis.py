import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def get_data_distribution():
    data_dir = '../data/SnakeCLEF2023-small_size'
    years = os.listdir(data_dir)

    df = []
    for year in tqdm(years):
        species = os.listdir(data_dir + '/' + year)
        for s in species:
            current_s = os.listdir(data_dir + '/' + year + '/' + s)
            number = len(current_s)
            df.append({'year': year, 'species': s, 'num': number})

    data = pd.DataFrame(df)
    return data.groupby('species', as_index=False).num.sum().sort_values(by='num')
