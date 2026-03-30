# ---- kinetic parameters ----
alpha = 1.026
beta = 0.954
gamma = 0.0

Ea, ln_A = 111.2e3, 36. # holders
T = 303.0 # K
k = np.exp(ln_A)*np.exp((-Ea / (R*T)))

# ---- initial concentrations ----
C_I0 = 10.0e-3
C_Ac0 = 2.0
C_H0 = 1e-3

def concentrations_from_ice(x_val):
    '''use the results from the ICE table to eval concentrations'''
    C_I2 = C_I0 * (1 - x_val)
    C_Ac = C_Ac0 - x_val * C_I0
    C_H3O = C_H0 + x_val * C_I0
    return C_Ac, C_H3O, C_I2
def dxdt_equation(_t, y):
    '''rate equation for numerical simulations'''
    x_val = np.clip(y[0], 0.0, 1.0)
    C_Ac, C_H3O, C_I2 = concentrations_from_ice(x_val)
    dxdt = (
        k
        * max(C_Ac, 0.0) ** alpha
        * max(C_H3O, 0.0) ** beta
        * max(C_I2, 0.0) ** gamma
        / C_I0 )
    return [dxdt]
sol = solve_ivp( # this is imported from scipy !!
    dxdt_equation,
    (0, 5000),
    [0],
    max_step=1.0
)
x = np.clip(sol.y[0], 0.0, 1.0) # no negative concentrations, bruh
C_Ac_prof, C_H3O_prof, C_I2_prof = concentrations_from_ice(x)

# from there these may be plotted accordingly.
