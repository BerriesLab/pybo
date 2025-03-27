<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# SPARC: Simulation Platform for Advanced Rough-cut Control in Wire EDM

### Design

The Simulation Platform for Advanced Rough-cut Control in Wire EDM is based on five modules, each focused on modeling
and simulating a single physics of the multiphyiscs problem.

+ **Module 1: Ignition**. Models and simulates the *stochastic* behavior of the ignition sparks, determining spark
  occurrence and
  characteristics. The probability of breakdown occurrence is captured by the hazard
  function $$h(t) = \frac{f(t)}{1-F(t)} $$ where $F(t)$ is the Cumulative Distribution Function (CDF) and $f(t)$ the
  density probability function $f(t)$ of a breakdown event. Approximations to the hazard function are possible
  depending on $f(t)$.
  In the simulation, at each timestep $\Delta t$ (typically $1 \, \mathrm{\mu s}$) the following actions are performed:
    + If $d=0$ initiate a short circuit evnet.
    + If $d>0$ and the dielectric channel is already ionized, initiate an arc event.
    + If $d>0$ and the dielectric channel is not already ionized, determine if the breakdown occurs: calculate the
      gap-dependent breakdown rate $\lambda(d)$, sample a random number $n \in [0, 1]$, and if $n<\lambda(d)$ initiate a
      spark discharge at a *random* location along the wire.


+ **Module 2: Material Removal**. Models and simulates the erosion of material from the workpiece using
  empirically-derived probability distributions. Specifically, experimental studies have demonstrated that the crater
  volume $\mathcal{V}$ follows a Gaussian distribution with mean $\mu_\mathcal{V}(\mathbf{p})$ and standard
  deviation $\sigma_\mathcal{V}(\mathbf{p})$, where $\mathbf{p}$ is a vector defining the pulse profile. For each spark
  event, the workpiece geometry is updated based on the sampled crater volume. Under 1D approximation, and assuming a
  constant kerf width (wire diameter plus twice the gap) for all spark energies, the volume is distributed uniformly
  across the workpiece height. The increment in the workpiece position $\Delta x_w$ is calculated
  as $\Delta x_w = \mathcal{V} / \left(k \cdot h_w\right)$, where $k$ is the kerf width. The workpiece position is then
  updated by adding this increment.


+ **Module 3: Dielectric State**. Models the dielectric fluid state, including debris concentration, flow and ionization
  status.
    + The normalized debris concentration $c \in [0, 1]$ in the gap
      is $$ \frac{dc}{dt}=\beta \mathcal{V} \left( 1-c \right) - \gamma fc$$ where $\mathcal{V}$ is the crater volume
      from each discharge, $\beta$ is the debris generation coefficient, and $f$f is the liquid flow rate. Note that
      when $c=1$ the gap is saturated with debris, leading to a short circuit.
    +


+ **Module 4: Wire Condition**. Models the wire condition as a function of temperature, stress and material properties,
  in order to predict the
  probability of wire breakage.


+ **Module 5: Machine Mechanics**. Model the EDM machine mechanics, including servo control and wire vibrations, to
  model the relative motion of the wire with respect to the workpiece.

### State Vector

SPARC models EDM processes as a discrete-time Markovian systems, with state vector $s_t \in \mathit{S}$,
where $\mathit{S}$ is the state space. The state vector encapsulates physical descriptors (voltage $V$, current $I$,
wire temperature field $T_\mathrm{wire}$, dielectric temperature field $T_\mathrm{diel}$, ...), actuator states (servo,
generator, wire tension, ...), and a set of temporal counters to track time-dependent phenomena, such as time since
voltage application to determine the probability of electrical discharge. The system state is the concatenation of all
modules states
$$
s_t = \left[ s_{1,t}, s_{2,t}, s_{3,t}, s_{4,t}, s_{5,t} \right]
$$
where $s_{i,t}$ is the state at the time $t$ of the *i*-th module. The simulation framework distinguishes between state
variables $s_t$ and simulation parameters $\theta$. While the former represent dynamic quantities that evolve during the
simulation, the latter are fixed quantities that remain fixed throughout a simulation run.
The system's evolution is governed by probabilistic state transitions defined by teh function
$$
P\left(\mathbf{s}_{t+1}\vert \mathbf{s}_t, \mathbf{\theta}\right) :\mathit{S}\times\mathit{S}\times\Theta \rightarrow\left[0, 1\right]
$$

### State update

+ $P\left(\mathbf{s}_{1,\, t+1} \, \vert \, \mathbf{s}_t, \, \mathbf{\theta}\right)$
+ $P\left(\mathbf{s}_{2,\, t+1} \, \vert \, \mathbf{s}_t, \, \mathbf{s}_{1,\, t+1}, \, \mathbf{\theta}\right)$
+ $P\left(\mathbf{s}_{3,\, t+1} \, \vert \, \mathbf{s}_t, \, \mathbf{s}_{1,\, t+1}, \, \mathbf{s}_{2,\, t+1}, \, \mathbf{\theta}\right)$
+ $P\left(\mathbf{s}_{4,\, t+1} \, \vert \, \mathbf{s}_t, \, \mathbf{s}_{1,\, t+1}, \, \mathbf{s}_{2,\, t+1}, \, \mathbf{s}_{3,\, t+1}, \, \mathbf{\theta}\right)$
+ $P\left(\mathbf{s}_{5,\, t+1} \, \vert \, \mathbf{s}_t, \, \mathbf{s}_{1,\, t+1}, \, \mathbf{s}_{2,\, t+1}, \, \mathbf{s}_{3,\, t+1}, \, \mathbf{s}_{4,\, t+1}, \, \mathbf{\theta}\right)$

The complete state transition probability is thus given by the following product of module probabilities:

$P\left(\mathbf{s}_{t+1} \, \vert \mathbf{s}_t, \, \mathbf{\theta} \right)=
P\left(\mathbf{s}_{1,\, t+1} \, \vert \cdot \right)
P\left(\mathbf{s}_{2,\, t+1} \, \vert \cdot \right)
P\left(\mathbf{s}_{3,\, t+1} \, \vert \cdot \right)
P\left(\mathbf{s}_{4,\, t+1} \, \vert \cdot \right)
P\left(\mathbf{s}_{5,\, t+1} \, \vert \cdot \right)$

### Next steps: Testing

What do we need for testing????

+ Improve model
    + Module 1: Include spark location distribution.
    + Module 4: Include thermal stress.
    + Module 2: 3D geometrical removal
    + Module 3: Enhance debris slow model.
    + Module 5: Include vibrations to improve actual gap width.


+ Test model
    + Collect data from machine, including voltage $V(t)$, current $I(t)$, and wire position $r(t)$. Which data can we
      extract from the machine?
    + 


