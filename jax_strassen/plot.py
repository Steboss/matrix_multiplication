import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sbn 
import pandas as pd 

sbn.set_style("whitegrid")
palette = sbn.color_palette() 
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

df = pd.read_csv("jax_results.csv")
x_axis = df['SIZE']
base_y = df[' BASE_RES_MEAN']
base_err = df[' BASE_RES_STD']

opt_y = df[' OPT_RES_MEAN']
opt_err = df[' OPT_RES_STD']

fig,ax = plt.subplots(figsize=[15,10])
ax.set_xlabel("Matrix Size", fontsize=20)
ax.set_ylabel("Time / s", fontsize=20)
ax.errorbar(x_axis, base_y, yerr=base_err, c=palette[0], label="MatMul")
ax.errorbar(x_axis, opt_y, yerr=opt_err, c=palette[1], label="Strassen")
ax.legend(loc="best", fontsize=20)
fig.savefig("strassen.png", dpi=300)
