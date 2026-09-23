## Tuning Stage 1: Translational Loop ($\kappa_1, \kappa_v$)

Unlike Stage $\Omega$ and Stage R, Stage 1 does not have a clean, constant exponential convergence rate. Its Lyapunov derivative is:
$$
\dot{V}_1 = -\kappa_v \|z_v\|^2 - \kappa_1 \varepsilon_p \xi_p e_1 \frac{\dot{\rho}_p}{\rho_p}
$$
The first term is strictly dissipative (damping). The second term can be **positive** when the tube is shrinking near obstacles ($\dot{\rho}_p < 0$). Therefore, we cannot simply assign a time constant to Stage 1. Instead, we tune $\kappa_1$ and $\kappa_v$ by satisfying three physical and stability constraints.

### Constraint 1: Stability During Tube Shrinkage
To ensure that the damping term dominates the energy injected by the shrinking tube, we require $\dot{V}_1 < 0$ whenever the velocity error $\|z_v\|$ is above some minimum expected threshold $\|z_v\|_{\min}$. 

Let $\gamma_{\max} := \max_t \left|\frac{\dot{\rho}_p(t)}{\rho_p(t)}\right|$ be the maximum relative tube shrinkage rate (provided by the STT generator). 
Let $\bar{e}_1 < 1$ be the maximum allowed normalized position error (e.g., $\bar{e}_1 = 0.8$). In this safe operating region, the barrier terms are bounded by:
$$
M(\bar{e}_1) := \varepsilon_p(\bar{e}_1) \xi_p(\bar{e}_1) \bar{e}_1 = \ln\left(\frac{1+\bar{e}_1}{1-\bar{e}_1}\right) \frac{2\bar{e}_1}{1-\bar{e}_1^2}
$$
For $\dot{V}_1 < 0$, we need:
$$
\kappa_v \|z_v\|_{\min}^2 > \kappa_1 M(\bar{e}_1) \gamma_{\max}
$$
This gives our first tuning inequality (a lower bound on the ratio of the gains):
$$
\boxed{
\frac{\kappa_v}{\kappa_1} > \frac{M(\bar{e}_1) \gamma_{\max}}{\|z_v\|_{\min}^2}
}
$$

### Constraint 2: Transient Response (Critical Damping)
Away from obstacles ($\dot{\rho}_p = 0$), the linearized Stage-1 dynamics for small errors ($e_1 \ll 1$) behave as a damped harmonic oscillator:
$$
\ddot{\tilde{p}} + \kappa_v \dot{\tilde{p}} + \frac{4\kappa_1}{\rho_p^2}\tilde{p} = 0
$$
The natural frequency is $\omega_n = \frac{2\sqrt{\kappa_1}}{\rho_p}$ and the damping ratio is $\zeta = \frac{\kappa_v \rho_p}{4\sqrt{\kappa_1}}$. 
To avoid oscillatory behavior and ensure optimal transient tracking of the tube center, we choose critical damping ($\zeta = 1$). This provides our second relationship:
$$
\boxed{
\kappa_v = \frac{4\sqrt{\kappa_1}}{\rho_p}
}
$$

### Solving for $\kappa_1$ and $\kappa_v$
Substitute the critical damping relation (Constraint 2) into the stability ratio (Constraint 1):
$$
\frac{4\sqrt{\kappa_1}}{\kappa_1 \rho_p} > \frac{M(\bar{e}_1) \gamma_{\max}}{\|z_v\|_{\min}^2} \implies \frac{4}{\sqrt{\kappa_1} \rho_p} > \frac{M(\bar{e}_1) \gamma_{\max}}{\|z_v\|_{\min}^2}
$$
Solving for $\kappa_1$ yields the **maximum allowable stiffness** to maintain stability during aggressive obstacle avoidance:
$$
\boxed{
\kappa_1 \le \frac{16 \|z_v\|_{\min}^4}{M(\bar{e}_1)^2 \gamma_{\max}^2 \rho_p^2}
}
$$
Once $\kappa_1$ is chosen (typically at or slightly below this upper bound), $\kappa_v$ is directly computed via Constraint 2.

---

## Constraint 3: Physical Feasibility (Thrust Limit)
The previous constraints ensure mathematical stability, but we must guarantee that the virtual acceleration $u_1$ does not exceed the physical thrust capabilities of the quadrotor. 

The physical feasibility condition for the thrust vector $F_d = m(u_1 + g e_3)$ is $\|F_d\| \le f_{\max}$, which implies:
$$
\|u_1\| \le \frac{f_{\max}}{m} + g
$$
From the Stage-1 control law:
$$
u_1 = \ddot{\sigma} - \kappa_v z_v - \frac{\kappa_1}{\rho_p} \varepsilon_p \xi_p \hat{\tilde{p}}
$$
Bounding the norm of each term using worst-case expected operating conditions:
1. Feedforward: $\|\ddot{\sigma}\| \le \sigma_{\ddot{\max}}$
2. Damping: $\|z_v\| \le z_{v,\max}$
3. Barrier: $\frac{1}{\rho_p} \varepsilon_p \xi_p \le K_{\text{bar}}(\bar{e}_1) := \frac{1}{\rho_p} \ln\left(\frac{1+\bar{e}_1}{1-\bar{e}_1}\right) \frac{2}{1-\bar{e}_1^2}$

Applying the triangle inequality, the maximum commanded acceleration is:
$$
\|u_1\|_{\max} \le \sigma_{\ddot{\max}} + \kappa_v z_{v,\max} + \kappa_1 K_{\text{bar}}(\bar{e}_1)
$$
To guarantee physical feasibility, we require $\|u_1\|_{\max} \le \frac{f_{\max}}{m} + g$. Defining the **Available Acceleration Budget** as $a_{\text{budget}} := \frac{f_{\max}}{m} + g - \sigma_{\ddot{\max}}$, we get our third inequality:
$$
\boxed{
\kappa_v z_{v,\max} + \kappa_1 K_{\text{bar}}(\bar{e}_1) \le a_{\text{budget}}
}
$$
*Note: If the gains derived from Constraints 1 & 2 violate this physical feasibility constraint, the drone lacks sufficient thrust to track the tube under the chosen $\bar{e}_1$. In this case, one must relax the tube boundary (increase $\bar{e}_1$) or reduce the expected maximum velocity error $z_{v,\max}$.*

---

## Complete Tuning Algorithm Summary

**Step 1: Inner Loop (Stage $\Omega$)**
**Step 2: Middle Loop (Stage R)**
**Step 3: Outer Loop (Stage 1)**
1. Define operational limits: $\bar{e}_1$ (max position error), $\gamma_{\max}$ (max tube shrinkage rate), $z_{v,\min}$ (min velocity error for stability), $z_{v,\max}$ (max expected velocity error), and $\sigma_{\ddot{\max}}$ (max tube center acceleration).
2. Compute the barrier bounds $M(\bar{e}_1)$ and $K_{\text{bar}}(\bar{e}_1)$.
3. Compute the maximum allowable stiffness from the tube-shrinkage stability constraint:
   $$
   \kappa_{1,\max} = \frac{16 \|z_v\|_{\min}^4}{M(\bar{e}_1)^2 \gamma_{\max}^2 \rho_p^2}
   $$
4. Choose $\kappa_1 \le \kappa_{1,\max}$.
5. Compute $\kappa_v = \frac{4\sqrt{\kappa_1}}{\rho_p}$.
6. **Verify Physical Feasibility:** Check if $\kappa_v z_{v,\max} + \kappa_1 K_{\text{bar}}(\bar{e}_1) \le \frac{f_{\max}}{m} + g - \sigma_{\ddot{\max}}$. If it fails, reduce $\kappa_1$ or increase $\bar{e}_1$.