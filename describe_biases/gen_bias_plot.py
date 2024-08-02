import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats
from matplotlib.ticker import AutoMinorLocator

x = np.random.normal(1,0.5,1000)
x_bias = np.random.normal(1.5, 0.5, 1000)

x = np.arange(0,2,0.0001)
x_bias = np.arange(1,3,0.0001)
mu_true = 1
mu_unbiased = 1.2
mu_bias = 2
sigma = 0.25

y_no_bias = np.exp(-0.5 * (x - mu_true)**2 / sigma**2)
y_unbiased = np.exp(-0.5 * (x - mu_unbiased)**2 / sigma**2)
y_bias = np.exp(-0.5 * (x_bias - mu_bias)**2 / sigma**2)

fig,ax = plt.subplots(1,1, figsize = (16,8))

ax.plot(x,y_no_bias, label = r'Truth: $\mathcal{R} = 0$', c = 'blue')
ax.axvline(x = mu_true, label = r'True Parameter: $\theta_{tr}$', c = 'blue', linestyle = 'dashed',linewidth=4)
ax.plot(x_bias,y_bias, label = r'Biased Approximate: $\mathcal{R} > 1$', c = 'purple')
ax.axvline(x = mu_bias, label = r'strong bias: $\theta_{bf}$', c = 'purple', linestyle = 'dashed', linewidth =4)
ax.plot(x,y_unbiased, label = r'Unbiased Approximate: $\mathcal{R} < 1$', c = 'red')
ax.axvline(x = mu_unbiased, label = r'weak bias: $\theta_{bf}$', c = 'red', linestyle = 'dashed', linewidth =4)

px=np.arange(0.68,1.32,0.0001)
ax.fill_between(px,np.exp(-0.5 * (px - mu_true)**2 / sigma**2),color='yellow', alpha = 0.3)

ax.legend(fontsize = 17,loc = "upper right",ncols = 3, bbox_to_anchor=(0.9, -0.22), shadow = True)
ax.set_ylabel(r'')
ax.set_xlabel(r'Parameter: $\theta$', fontsize = 30)
ax.set_title(r'Instructive example: Unbiased vs Biased',fontsize = 40)
ax.set_xlim([0,3])
ax.set_ylim([0,1.5])
ax.set_facecolor('#EBEBEB')
# [ax.spines[side].set_visible(False) for side in ax.spines]
# # Style the grid.
# ax.grid(which='major', color='white', linewidth=1.2)
# ax.grid(which='minor', color='white', linewidth=0.6)
# # Show the minor ticks and grid.
# ax.minorticks_on()
# # Now hide the minor ticks (but leave the gridlines).
# ax.tick_params(which='minor', bottom=False, left=False)

# Only show minor gridlines once in between major gridlines.
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax.set_xticks(fontsize = 16)
ax.set_yticks([])
ax.set_ylabel(r'Posterior: $p(\theta|d)$', fontsize = 30)
ax.xaxis.set_tick_params(labelsize=24)
plt.tight_layout()
plt.savefig("Biased_example.pdf",bbox_inches = 'tight')
plt.show()
plt.clf()