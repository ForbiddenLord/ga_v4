Problem Statement: the paper derivation assumes two-time-scale separation, so the stage-R assumption that inner loop is fast enough that Ω_R = Ω_C probably breaks with motor dynamics, thus affecting \dot{V_R} via time scale separation, and in turn violating the stage-1 thrust direction mismatch. 

probably slowing the whole cascade down and enforcing slower convergence than motor dynamics works?
but does the paper provide convergence rate or if the rate can be inferred, we can enforce time scale separation by making stage-R and stage-2 slower than motor dynamics.

Solution:
## Yes, the rates are inferable from the Lyapunov identities.

Here is exactly what each stage provides:

---

### Stage Ω (Inner Torque Loop): Clean Exponential Rate

From the exact identity in Section V-D:

$$
\frac{d}{dt}\left[\frac{1}{2}z_\Omega^\top J z_\Omega\right] = -\kappa_\Omega \|z_\Omega\|^2
$$

Using the eigenvalue bounds of $J$:

$$
\frac{1}{2}\lambda_{\min}(J)\|z_\Omega\|^2 \leq V_\Omega \leq \frac{1}{2}\lambda_{\max}(J)\|z_\Omega\|^2
$$

This gives:

$$
\dot{V}_\Omega \leq -\frac{2\kappa_\Omega}{\lambda_{\max}(J)} V_\Omega
$$

So the **exponential convergence rate** is:

$$
\boxed{
\lambda_\Omega = \frac{2\kappa_\Omega}{\lambda_{\max}(J)}
}
$$

This is a clean, constant rate. The time constant is $\tau_\Omega = 1/\lambda_\Omega$.

---

### Stage R (Attitude Funnel): Time-Varying Rate

From the exact identity in Section V-C:

$$
\dot{V}_R = -\frac{\kappa_R}{\rho_R(1-e_R)}\varepsilon_R^2
$$

Since $V_R = \frac{1}{2}\varepsilon_R^2$, this gives:

$$
\dot{V}_R = -\frac{2\kappa_R}{\rho_R(1-e_R)} V_R
$$

The rate is **time-varying** because $\rho_R(t)$ and $e_R(t)$ change over time. However, it is bounded below:

$$
\frac{2\kappa_R}{\rho_R(1-e_R)} \geq \frac{2\kappa_R}{\rho_{R,\max}(1-\bar{e}_R)}
$$

where $\bar{e}_R < 1$ is the maximum allowed normalized error (can be fixed at the time of tuning). So the **worst-case (slowest) convergence rate** is:

$$
\boxed{
\lambda_R^{\min} = \frac{2\kappa_R}{\rho_{R,\max}(1-\bar{e}_R)}
}
$$

---

### Stage 1 (Translational Tube): No Clean Exponential Rate

From Proposition 1 in Section V-A:

$$
\dot{V}_1 = -\kappa_v \|z_v\|^2 - \kappa_1 \varepsilon_p \xi_p e_1 \frac{\dot{\rho}_p}{\rho_p}
$$

The first term is dissipative, but the second term can be **positive** when $\dot{\rho}_p < 0$ (tube shrinking near obstacles). This means:

$$
\dot{V}_1 \leq -\kappa_v \|z_v\|^2 + \text{bounded positive term}
$$

There is **no clean exponential rate** for Stage 1. So Stage 1 does not have a simple time constant. Its convergence depends on the tube geometry and obstacle proximity.

---

### Can You Enforce Time-Scale Separation?

**Yes, in principle**, by choosing gains such that:

$$
\tau_{\text{motor}} \ll \tau_\Omega \ll \tau_R \ll \tau_{\text{Stage 1}}
$$

where:
- $\tau_{\text{motor}} \approx 20\text{--}50\text{ ms}$ (typical motor/ESC time constant)
- $\tau_\Omega = \frac{\lambda_{\max}(J)}{2\kappa_\Omega}$
- $\tau_R \approx \frac{\rho_{R,\max}(1-\bar{e}_R)}{2\kappa_R}$
- $\tau_{\text{Stage 1}}$ is not well-defined, but you can make the translational gains $\kappa_v, \kappa_1$ small enough that the translational loop is slow

---

### Summary

| Stage | Convergence Rate | Time Constant | Clean Exponential? |
|-------|------------------|---------------|-------------------|
| Stage Ω | $\lambda_\Omega = \frac{2\kappa_\Omega}{\lambda_{\max}(J)}$ | $\tau_\Omega = \frac{\lambda_{\max}(J)}{2\kappa_\Omega}$ | Yes |
| Stage R | $\lambda_R(t) = \frac{2\kappa_R}{\rho_R(t)(1-e_R(t))}$ | $\tau_R(t) = \frac{\rho_R(t)(1-e_R(t))}{2\kappa_R}$ | No (time-varying) |
| Stage 1 | No clean rate | Not well-defined | No (due to $\dot{\rho}_p$ term) |

---

Follow up problem: forget time constant for stage-1,
if you can find gains κ_Ω, κ_R by fixing time constants for stage-2, stage-R, motor time constant, $λ_{max}(J)$, $ρ_{R,max}(1-\bar{e_R})$
the only thing to tune remains κ_1, because κ_v is heuristically set using $\sqrt{\frac{κ_1}{\rho_p}}$.

Supplementary Question: It is currently working well without motor dynamics, and on introducing motor dynamics (first order lag), and communication delay, will it diverge or enter limit cycle?

or does it fail due to the assumption that stage_R and stage-2 time scale separation?

in the convergence rates, aren't we concerned about minimum time constant for stage-R? as it may violate time-scale-separation with stage-2, thus don't we need max convergence rate (min time constant) of the outer loop to guarantee the condition $\tau_{\text{motor}} \ll \tau_\Omega \ll \tau_R \ll \tau_{\text{Stage 1}}$ ?
or
1. $3\tau_{\Omega,max} \leq \tau_{R,min} ⇒ 3\frac{\lambda_{\max}(J)}{2\kappa_\Omega} \leq \frac{\rho_{R,\min}(1-\bar{e}_R)}{2\kappa_R}$ 

2. $3\tau_{motor} \leq \tau_{\Omega,min} ⇒ 3\tau_{motor} \leq \frac{\lambda_{\min}(J)}{2\kappa_\Omega}$

and in general for most category of quadrotors, Jxx (or Jyy) : Jzz = 1:1.5 or 1:2 so the inequality holds.
I have chosen 3x separation as a common cascaded system heuristic, it's a thing to be verified.