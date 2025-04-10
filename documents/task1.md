# Task1 - *x* vs *I*

### Current generator

+ $I_0$: the pedestal (or initial) current (in A)
+ $I_{\mathrm{max}}$: the maximum current (in A)
+ $t_{\mathrm{ramp}}$: the ramping time (in s)
+ $t_{\mathrm{ON}}$: the discharge time (in s)
+ $t_{\mathrm{pre}}$: Time Ignition delay, or pre-breakdown time (in s)
+ $\delta I$: current resolution = $0.1 \, \mathrm{A}$

The slope $\theta$ of the current pulse shape is
$$\theta = \frac{I_{\mathrm{max}} - I_{0}}{t_{\mathrm{ramp}}}$$

The closer the electrode is to the workpiece, the earlier the electrical discharge occurs, i.e. the
shorter $t_{\mathrm{pre}}$ is. In formula: $\downarrow d \, \Rightarrow \, \downarrow t_{\mathrm{pre}}$

Depending on $t_{\mathrm{pre}}$, the shape of the current pulse is more or less rectangular, i.e. the slope is

### Wear loss function

The process optimization includes finding the sweet spot between minimum electrode wearing and maximum process speed, or
minimum proces duration. Therefore, a possible loss function for the process optimization is
$$
\mathscr{L}
$$
