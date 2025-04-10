## List of topic to review

+ Fuzzy logic and fuzzy control systems
+ Reinforcement learning
+ Markov event

## Introduction

## Optimization

There are two main methods to optimize the efficiency and accuracy of DS-EDM:

+ Optimize the electrode design.
+ Optimize the control process, including the spark gap, voltage, and current, the relative movement between the
  electrode and the workpiece, and the flushing of the dielectric to remove the debris from the gap.

The complexity of the physical processes occurring during the erosion process in a typical DS-EDM process makes the
development of a mathematical model a very hard task to accomplish.
Historically, optimization has followed routes that included *fuzzy control strategy* of the servo control .

## Objectives}

### Improve electrode wearing

**Description**: To enhance the optimal Pareto frontier, i.e. the ratio between electrode wear and MRR. The state
of the art for minimal electrode wear consists in applying a fixed trapezoidal current shape. While this minimizes the
electrode wear, it also limits the MRR.

**Solution**: There is some level of understanding about how the electric current patterns, resulting from the voltage
pulse, affects the MRR and the electrode wear.\cite{} Typically, a trapezoidal current-shape is used. The resulting
current profile shows an initial current value, called pedestal current. The slope of the current ramp allows to control
the electrode wear. Within this project, a fast, high-voltage shape generator to control the voltage pulse shape can be
used as part of the DS-EDM machine. The proposed solution consists in developing a ML/AI system capable of adjusting the
voltage pulse shape to adjust the current profile, and then minimize the electrode wear. The ultimate goal is to
implement an algorithm on a FPGA.

### Improve side gap control

**Description**: While a side gap is necessary to remove the machined material particles, the gap should be as
small as possible to (i) ensure high accuracy of the final cavity shape, (ii) ensure effective pumping out of the
machined particles, (iii) avoid high-energy large lateral discharge, and (iv) avoid mechanical vibrations that can
affect accuracy. Notably, the voltage shapes of lateral and front discharges are different, and therefore they can be
used to discriminate among the two.\newline  
**Solution**: Implement a 3-step voltage pulse to simulate less than optimal machining conditions. Using Bayesian
optimization, identify a set of optimal parameters. The final solution would be to adapt the ignition voltage in real
time, instead of using static parameters.

### Improve the mechanical response

**Description**: the electrical and mechanical dynamics in an DS-EDM process occur on two different time scales,
the former being in the tens of Hertz range, the latter being in the kilo Hertz range. Since the position of the
electrode is adjusted in response to the measured voltage of the electrical discharge, the adjustment must be taken in
the fastest time possible to prevent the electrical discharge to occur in the wrong place.\newline
**Solution**: train the "Hodge-Podge" servo control mode, which offers the greatest degree of freedom, with AI/ML
techniques to dynamically adapt the mode parameters in real-time during machine operation.

## Scientific Goals

+ Quantify the effect of the current pulse shape, and thus of the voltage pulse shape, on the size of craters in the
  electrode and in the workpiece.
+ Quantify how the open voltage and gap distance affects the ignition delay time, for use in control
  algorithms.

## Technological Goals

\begin{itemize}
\item Develop and evaluate, by integration into existing DS-EDM solutions, new process control strategies based on AI/ML
methods.
\item \textcolor{red}{Develop a strategy to infer electrode wear and MRR from pulse shape in order to provide feedback
for the learning algorithms.}
\end{itemize}

\newpage\section

\section{Process}
\begin{figure}
\centering
\includegraphics[width=0.9\linewidth]{images/process_isopulse.png}
\caption{Caption}
\label{fig:enter-label}
\end{figure}

\end{document}
