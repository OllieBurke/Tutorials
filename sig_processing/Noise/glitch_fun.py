
# Credit: Martina Muratori + LDC

import numpy as np

co = 299758492
# add function
def HeavisideTheta(x, xp=None):

    if xp is None:
        xp = np
    
    squeeze = True if x.ndim == 1 else False

    x = xp.atleast_1d(x)

    out = 1.0 * (x >= 0.0)

    if squeeze:
        out = out.squeeze()

    return out

def DiracDelta(x, xp=None):

    if xp is None:
        xp = np
    squeeze = True if x.ndim == 1 else False

    x = xp.atleast_1d(x)

    # 1e-10 to guard against numerical error
    out = 1.0 * (xp.abs(x) < 1e-10)

    if squeeze:
        out = out.squeeze()

    return out

def tdi_glitch_XYZ1(t_in, T=8.3, tau_1=480.0, tau_2=1.9394221536001746, Deltav=2.22616837*10**(-11), t0=600.0, mtm=1.982, xp=None):
    """
    t_in: Values of time that the glitch is evaluated on 
    T: Time it elapses for
    tau_1: Decay time
    tau_2: Decay time
    deltav: Amplitude
    t0 = Starting time of glitch
    mtm:
    """

    if xp is None:
        xp = np

    out = xp.zeros((3, len(t_in)))

    run = ~(xp.isinf(xp.exp((-t_in + t0))) | xp.isnan(xp.exp((-t_in + t0))))

    t = t_in[run]
    tdiX1link12 = (
        ((mtm*Deltav*(tau_1*(-t + t0 + tau_1 - xp.exp((-t + t0)/tau_1)*tau_1) + tau_2*(t - t0 + (-1 + xp.exp((-t + t0)/tau_2))*tau_2))*DiracDelta(t - t0))/(-tau_1 + tau_2) + 
          (mtm*Deltav*((-1 + xp.exp((-t + t0)/tau_1))*tau_1 + tau_2 - xp.exp((-t + t0)/tau_2)*tau_2)*HeavisideTheta(t - t0))/(-tau_1 + tau_2))/co - 
       ((mtm*Deltav*(tau_1*(-t - 4*T + t0 + tau_1 - xp.exp((-t - 4*T + t0)/tau_1)*tau_1) + tau_2*(t + 4*T - t0 + (-1 + xp.exp((-t - 4*T + t0)/tau_2))*tau_2))*DiracDelta(t + 4*T - t0))/(-tau_1 + tau_2) + 
          (mtm*Deltav*((-1 + xp.exp((-t - 4*T + t0)/tau_1))*tau_1 + tau_2 - xp.exp((-t - 4*T + t0)/tau_2)*tau_2)*HeavisideTheta(t + 4*T - t0))/(-tau_1 + tau_2))/co
    )

    
    tdiY1link12 = (
        (-2*((mtm*Deltav*(-(tau_1*(t + T - t0 + (-1 + xp.exp(-(t + T - t0)/tau_1))*tau_1)) + tau_2*(t + T - t0 + (-1 + xp.exp(-(t + T - t0)/tau_2))*tau_2))*DiracDelta(t + T - t0))/(-tau_1 + tau_2) + 
        (mtm*Deltav*((-1 + xp.exp(-(t + T - t0)/tau_1))*tau_1 + tau_2 - tau_2/xp.exp((t + T - t0)/tau_2))*HeavisideTheta(t + T - t0))/(-tau_1 + tau_2)))/co + (2*((mtm*Deltav*(tau_1*(-t - 3*T + t0 + tau_1 - xp.exp((-t - 3*T + t0)/tau_1)*tau_1) + tau_2*(t + 3*T - t0 + (-1 + xp.exp((-t - 3*T + t0)/tau_2))*tau_2))*DiracDelta(t + 3*T - t0))/        (-tau_1 + tau_2) + (mtm*Deltav*((-1 + xp.exp((-t - 3*T + t0)/tau_1))*tau_1 + tau_2 - xp.exp((-t - 3*T + t0)/tau_2)*tau_2)*HeavisideTheta(t + 3*T - t0))/(-tau_1 + tau_2)))/co
    )

    tdiZ1link12 = xp.zeros_like(tdiX1link12)

    out[0, run] = tdiX1link12
    out[1, run] = tdiY1link12
    out[2, run] = tdiZ1link12

    return out



