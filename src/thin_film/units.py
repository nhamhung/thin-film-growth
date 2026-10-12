"""Convert real-world deposition conditions into the simulation's dimensionless knobs.

The simulations only understand two numbers:
  epsilon : bond strength divided by thermal energy, E_bond / kT. Large epsilon
            (cold) means a bonded atom almost never breaks free; small epsilon
            (hot) means bonds are easily shaken loose.
  hops    : how many hop attempts happen per deposited atom. This is the ratio
            of "how fast atoms diffuse" to "how fast new atoms arrive".

Physics behind the conversion (Arrhenius law): a free surface atom vibrates
about nu0 ~ 1e13 times per second and each vibration has a chance
exp(-E_diff / kT) of carrying it over the energy barrier to the next site, so
its hop rate is D = nu0 * exp(-E_diff / kT). New atoms arrive at a flux F
(monolayers per second). Between two arrivals at a given site there are about
D / F hop attempts. Hot substrates or slow deposition give many hops, so atoms
find good sites and the film is smooth. Cold substrates or fast deposition
give few hops, so the film is rough.
"""

import numpy as np

K_B = 8.617333262e-5  # Boltzmann constant, eV per kelvin


def thermal_parameters(temperature: float, e_bond: float, e_diff: float, flux: float,
                       nu0: float = 1e13, max_hops: float = 1e4) -> tuple[float, float]:
    """Return (epsilon, hops) for a substrate temperature (K) and flux (ML/s).

    e_bond, e_diff : bond energy per neighbour and free-atom diffusion barrier, in eV
    max_hops       : computational cap. Above it the surface is effectively in
                     its fast-diffusion limit and extra hops change little but
                     cost a lot of run time.
    """
    kt = K_B * temperature
    epsilon = e_bond / kt
    hops = nu0 * np.exp(-e_diff / kt) / flux
    return float(epsilon), float(min(hops, max_hops))
