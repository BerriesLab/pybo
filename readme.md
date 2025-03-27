# WP2. FormACO Feasibility

It has been observed that the shape of the spark current pulse can affect the material removal
rate (MRR) and the amount of wear on the electrode. By restricting the pulse shape to a
trapezoidal shape and adjusting the initial current level (pedestal current) and the slope of the
ramp, it is possible to influence the amount of wear in real time.
To address this issue, the first challenge is to develop an artificial intelligence (AI) or machine
learning (ML) system that can change the slope and pedestal current of the spark current pulse
in a way that minimizes wear. This AI/ML system should be able to operate in real time,
adjusting the pulse shape as needed to prevent wear on the electrode. The ultimate goal is to
implement this AI/ML algorithm on a field-programmable gate array (FPGA) in order to shape
the spark current pulse on a spark-by-spark basis, maximizing the efficiency and effectiveness
of the DS-EDM process while minimizing wear on the tool electrode The developments of this
work package will be performed using the FORM S 450 prototype and using standard copper
and steel electrode/workpiece pairs for the machining experiments

## Algorithm Development

### Input

+ $V(t)$: Discharge Voltage (in V), xxx Hz, number of samples averaged per acquisition...
+ $I(t)$: Discharge current (in A), xxx Hz, number of samples averaged per acquisition...
+ $d(t)$: Relative position of the electrode (in m), xxx Hz, number of samples averaged per acquisition...

Estimated number of samples $N\sim 10^5$...??

### Constraints

+ $V_{\mathrm{min}}\, \mathrm{V} < V(t) < V_{\mathrm{max}}\, \mathrm{V}$
+ $I_{\mathrm{min}}\, \mathrm{A} < I(t) < I_{\mathrm{max}}\, \mathrm{A}$
+ $d_{\mathrm{min}}\, \mathrm{m} < d(t) < d_{\mathrm{max}}\, \mathrm{m}$

### Relations

+ The gap size $L$ can be estimated from $L(x, y, z)=$...?
+ The electrode wear is related to the initial current level via $f(I_0) =$...
+ The electrode wear is related to the current slope via $f(\frac{dI}{dt})=$...

### Data Acquisition

On a ... EDM machine. The Voltage and the current are measured via an oscilloscope (model xxx) and Python API are used
to extract the relative position of the electrode.

### Output

The model should set the trapezoidal shape of the current pulse, specifically:

+ The initial current level.
+ The slope of the ramp.

In response to the current status of the process.

