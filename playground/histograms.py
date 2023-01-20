import pathlib

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel(pathlib.Path('playground', 'test_dataset_resolutions.xlsx'))
df.plot.bar()

plt.show()
plt.clf()