import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R

# reactor values
T_ref = 303.0                   # [=] K
T_rxn = 353.0                   # [=] K
X = 0.8
CAo, CHo, CIo = 1.0, 0.001, 1.0 # [=] M
Q = 10.0                        # [=] L/min

# parameters and their stdev (95% CI / 1.96)
params = {
    'Ea':    (111200, 5600  / 1.96),  # [=] J/mol
    'ln_A':  (36.0,   2.2   / 1.96), 
    'alpha': (1.026,  0.063 / 1.96),
    'beta':  (0.954,  0.068 / 1.96),
    'gamma': (-0.134, 0.067 / 1.96)
}

# func to calculate volume
def calc_V(Ea, ln_A, alpha, beta, gamma):
  k_ref = np.exp((-Ea / (R * T_ref)) + ln_A)
  k_rxn = k_ref * np.exp((-Ea / R) * (1/T_rxn - 1/T_ref))
  
  CA = CAo * (1 - X)
  CH = CHo + CAo * X
  CI = CIo - CAo * X
  
  r = k_rxn * (CA**alpha) * (CH**beta) * (CI**gamma)
  return CAo * Q * X / r

# volume for given parameters 
avg = {k: v[0] for k,v in params.items()}
vol = calc_V(**avg)
print (f"CSTR Volume: {vol:.1f} L")


# lets do a montecarlo simulator for the 90% CI
np.random.seed(42)
N = 100000

samples = {k: np.random.normal(mu, sig, N) for k, (mu, sig) in params.items()}
V_mc = calc_V(**samples)
lo, hi = np.percentile(V_mc, [5, 95])
print(f"\n90% CI: [{lo:.4f}, {hi:.4f}] L")
print(f"Mean:   {V_mc.mean():.4f} L  |  Median: {np.median(V_mc):.4f} L")

# print the sensitivities
print("\nSensitivity (effect on V):")
sensitivities = {}
for name, (mu, sig) in params.items():
    kwargs_lo = {**avg, name: mu - sig}
    kwargs_hi = {**avg, name: mu + sig}
    delta = (calc_V(**kwargs_hi) - calc_V(**kwargs_lo)) / 2
    sensitivities[name] = abs(delta)
    print(f"  {name:6s}: {abs(delta):.4f} L  (relative: {abs(delta)/vol*100:.1f}%)")

# plotting
pct_change = np.linspace(0, 100, 300)

# names and colors for each param
params_plot = {
    "Ea":    ("Activation Energy",   "#D81B60"),
    "alpha": ("Acetone Rate Order ()",  "#2196F3"),
    "beta":  ("HCl Rate Order ()",      "#FFC107"),
    "gamma": ("Iodine Rate Order ()",   "#004D40"),
    "ln_A":  ("ln(A)",               "#7F77DD"),
}

fig, ax = plt.subplots(figsize=(8, 5))

for name, (label, color) in params_plot.items():
    pct_V = []
    for p in pct_change:
        new_params = {**avg, name: avg[name] * (1 + p/100)}
        V_new = calc_V(**new_params)
        pct_V.append((V_new - vol) / vol * 100)
    ax.plot(pct_change, pct_V, label=label, color=color, linewidth=2)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Percent Change in Parameter (%)", fontsize=12)
ax.set_ylabel("Percent Change in Required CSTR Volume (%)", fontsize=12)
ax.set_ylim(-100, 100)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spider_sensitivity.png", dpi=300)
