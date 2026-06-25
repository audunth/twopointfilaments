me = 9.109e-31 # Electron mass in kg
Da = 1.661e-27 # Unified atomic mass unit in kg
mD = 2.014*Da # Deuterium mass in kg

# Mass of species in kg
mass = {'protium': 1.008*Da,
        'deuterium': 2.014*Da,
        }

def xi_factor(tau,ion_species):
    return 7*(1+tau)**1.5*(me/mass[ion_species])**0.5/(3.2*2)
