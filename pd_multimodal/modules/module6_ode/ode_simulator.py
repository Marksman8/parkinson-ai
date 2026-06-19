"""
MODULE 6 — ODE Neuron Simulation Engine
Hodgkin-Huxley neuron model + Dopamine kinetics.
Runs entirely on CPU using scipy — no GPU needed.
"""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    t:          np.ndarray
    y:          np.ndarray
    model_name: str
    params:     Dict
    figure_path: str = ""

    def to_dict(self) -> Dict:
        return {
            "model_name":  self.model_name,
            "params":      self.params,
            "figure_path": self.figure_path,
            "t_range":     [float(self.t[0]), float(self.t[-1])],
            "n_points":    len(self.t),
        }


# ─────────────────────────────────────────────────────────────────────
# Model 1 — Hodgkin-Huxley
# ─────────────────────────────────────────────────────────────────────
class HodgkinHuxleyModel:
    """
    Classic Hodgkin-Huxley neuron model.
    Simulates action potential dynamics in dopaminergic neurons.
    Perturb I_ext or conductances to model PD neuron dysfunction.
    """
    def __init__(self, params: Optional[Dict] = None):
        p = params or {}
        self.C_m   = p.get("C_m",   1.0)    # membrane capacitance (uF/cm²)
        self.g_Na  = p.get("g_Na",  120.0)  # max Na conductance (mS/cm²)
        self.g_K   = p.get("g_K",   36.0)   # max K conductance
        self.g_L   = p.get("g_L",   0.3)    # leak conductance
        self.E_Na  = p.get("E_Na",  50.0)   # Na reversal potential (mV)
        self.E_K   = p.get("E_K",  -77.0)   # K reversal potential
        self.E_L   = p.get("E_L",  -54.4)   # leak reversal potential
        self.I_ext = p.get("I_ext", 10.0)   # external current (uA/cm²)

    def _alpha_m(self, V): return 0.1*(V+40)/(1-np.exp(-(V+40)/10)) if abs(V+40)>1e-7 else 1.0
    def _beta_m(self, V):  return 4.0*np.exp(-(V+65)/18)
    def _alpha_h(self, V): return 0.07*np.exp(-(V+65)/20)
    def _beta_h(self, V):  return 1.0/(1+np.exp(-(V+35)/10))
    def _alpha_n(self, V): return 0.01*(V+55)/(1-np.exp(-(V+55)/10)) if abs(V+55)>1e-7 else 0.1
    def _beta_n(self, V):  return 0.125*np.exp(-(V+65)/80)

    def ode_system(self, t, y):
        V, m, h, n = y
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K  = self.g_K  * n**4     * (V - self.E_K)
        I_L  = self.g_L              * (V - self.E_L)
        dVdt = (self.I_ext - I_Na - I_K - I_L) / self.C_m
        dmdt = self._alpha_m(V)*(1-m) - self._beta_m(V)*m
        dhdt = self._alpha_h(V)*(1-h) - self._beta_h(V)*h
        dndt = self._alpha_n(V)*(1-n) - self._beta_n(V)*n
        return [dVdt, dmdt, dhdt, dndt]

    def simulate(
        self,
        t_span: Tuple = (0, 50),
        y0: Optional[list] = None,
        save_plot: bool = True,
        save_path: str = "data/outputs/hh_simulation.png",
    ) -> SimulationResult:
        y0  = y0 or [-65.0, 0.05, 0.6, 0.32]
        sol = solve_ivp(
            self.ode_system, t_span, y0,
            method="RK45", dense_output=True, max_step=0.01
        )
        t_eval = np.linspace(t_span[0], t_span[1], 5000)
        y_eval = sol.sol(t_eval)
        result = SimulationResult(
            t=t_eval, y=y_eval,
            model_name="Hodgkin-Huxley",
            params={k: v for k, v in self.__dict__.items()}
        )
        if save_plot:
            from .visualizer import plot_hodgkin_huxley
            result.figure_path = plot_hodgkin_huxley(result, save_path)
        logger.info(f"HH simulation complete. Plot: {result.figure_path}")
        return result


# ─────────────────────────────────────────────────────────────────────
# Model 2 — Dopamine Kinetics
# ─────────────────────────────────────────────────────────────────────
class DopamineKineticsModel:
    """
    3-compartment dopamine dynamics model.
    pd_loss parameter (0.0–1.0) simulates progressive
    nigrostriatal degeneration as seen in Parkinson's disease.
    """
    def __init__(self, params: Optional[Dict] = None):
        p = params or {}
        self.k_syn   = p.get("k_syn",   1.0)   # synthesis rate
        self.k_rel   = p.get("k_rel",   0.8)   # vesicular release rate
        self.k_reup  = p.get("k_reup",  0.5)   # DAT reuptake rate
        self.k_deg   = p.get("k_deg",   0.3)   # MAO-B degradation rate
        self.V_max   = p.get("V_max",   10.0)  # max synthesis capacity
        self.pd_loss = p.get("pd_loss", 0.0)   # 0=healthy, 1.0=complete loss

    def ode_system(self, t, y):
        D_store, D_syn, D_extra = y
        syn_rate = self.k_syn * self.V_max * (1.0 - self.pd_loss)
        dDstore  = syn_rate - self.k_rel * D_store
        dDsyn    = self.k_rel * D_store - (self.k_reup + self.k_deg) * D_syn
        dDextra  = self.k_reup * D_syn  - 0.1 * D_extra
        return [dDstore, dDsyn, dDextra]

    def simulate(
        self,
        t_span: Tuple = (0, 30),
        y0: Optional[list] = None,
        save_plot: bool = True,
        save_path: str = "data/outputs/dopamine_kinetics.png",
    ) -> SimulationResult:
        y0  = y0 or [5.0, 0.5, 0.1]
        sol = solve_ivp(
            self.ode_system, t_span, y0,
            method="RK45", max_step=0.05
        )
        result = SimulationResult(
            t=sol.t, y=sol.y,
            model_name="Dopamine Kinetics",
            params={k: v for k, v in self.__dict__.items()}
        )
        if save_plot:
            from .visualizer import plot_dopamine_kinetics
            result.figure_path = plot_dopamine_kinetics(result, save_path)
        logger.info(f"Dopamine simulation complete. Plot: {result.figure_path}")
        return result
