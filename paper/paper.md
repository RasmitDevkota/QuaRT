---
title: "QuaRT: a toolkit for the exploration of quantum methods for radiation transport"
tags:
  - Python
  - astronomy
  - astrophysics
  - cosmology
  - radiative transfer
  - radiation transport
  - lattice boltzmann methods
  - computational physics
  - computational astrophysics
  - computational cosmology
  - quantum
  - quantum information
  - quantum computing
  - quantum computation
  - quantum computer
  - quantum algorithms
  - quantum simulation
  - qiskit
  - physics
authors:
  - name: Rasmit Devkota
    orcid: 0009-0009-3294-638X
    affiliation: "1,2"
  - name: John Wise
    orcid: 0000-0003-1173-8847
    affiliation: "1,3"
affiliations:
 - name: School of Physics, College of Sciences, Georgia Institute of Technology, Atlanta, GA
   index: 1
 - name: School of Mathematics, College of Science, Georgia Institute of Technology, Atlanta, GA
   index: 2
 - name: Center for Relativistic Astrophysics, School of Physics, College of Sciences, Georgia Institute of Technology, Atlanta, GA
   index: 3
date: XX November 2025
bibliography: paper.bib
---

# Summary

`QuaRT` is a Python library for quantum simulation of radiative transfer in astrophysical and cosmological problems.

The source code for `QuaRT` is available on [GitHub](https://github.com/RasmitDevkota/QuaRT/). It can be installed via `pip` from the [`pypi` index](https://pypi.org/project/TODO/). Its [documentation](https://quart-lbm.readthedocs.io/) is hosted publicly.

# Statement of need

<!-- I accidentally wrote this part in almost the exact same way as the main paper, is it too similar? -->
<!-- Not sure which parts need more/less explanation -->
Computational cosmology, the use of simulations to study the evolution of the universe, is a rapidly-growing field of research, driven largely by the exponential increase in computing power following Moore's law. Numerous codes [@Iliev2006,Iliev2009,Bryan2014,BrummelSmith2019,OShea2015,Kannan2019,Kannan2021,Davis2012,Jiang2014,Hayes2003] have been written to study questions about the early universe and to obtain a better understanding of the plethora of observational results which have come with new telescopes such as the James Webb Space Telescope [@Adams2024]. However, classical high-performance computing hardware is slowly approaching the fundamental quantum limit where electronics cannot be scaled down any further [@Powell2008]. Quantum computers presents a potential path for further scaling of physical simulations by taking advantage of quantum phenomena such as superposition and entanglement which enable new models of computation. Many quantum algorithms have already been developed for the simulation of cosmological problems [@Mocz2021,Yamazaki2025,Kaufman2019,Joseph2021,Joseph2022,Wang2024,Liu2021]. Such simulations must model physical processes such as radiation transport from stars, magnetohydrodynamics of matter, gravitation between massive particles, gas chemistry, and the formation of structures such as stars, black holes, halos, and galaxies [@BrummelSmith2019]. Of these, radiation transport tends to be one of the most expensive steps due to the high dimensionality of the problem, but it also the most difficult to develop because of the lack of problems with analytical solutions [@Iliev2006,Iliev2009]. Quantum algorithms have been formulated for radiation transport, such as those based on ray tracing [@Lu20221,Lu20222,Lu2023,Mosier2023,Santos2025], random walks [@Lee2025], and other novel differential equations solvers [@Gaitan2024]. Classical lattice Boltzmann methods (LBMs), which track the distribution of a quantity on a grid with discretized propagation directions [@McNamaraZanetti1988], have already been applied extensively to study radiation transport [@McCulloch2016,BindraPatil2012,Mink2020,Olsen2025,Weih2020] and radiation hydrodynamics [@Asahina2020]. Quantum LBMs have also been constructed to study hydrodynamics [@Budinski2021,Budinski2022,Wawrzyniak20251,Wawrzyniak20252] and radiation transport [@Igarashi2024]. These quantum LBMs reduce the memory constraints of classical simulations by storing information in quantum state amplitudes, the number of which grows exponentially with the number of qubits, enabling the storage of data with only logarithmic scaling with problem size. Individual simulation steps can thus be made very high resolution and only the necessary amount of data needs to be stored classically. However, existing quantum LBMs are not suited for cosmological problems because such simulations are typically non-scattering, but isotropic sources under stars are not accurately resolved angularly by LBMs due to their discretized angular structure. `QuaRT` features the first known implementation of a quantum LBM which accurately resolves isotropic sources in non-scattering media; it does so via a novel methodology which we refer to as "angular redistribution", where radiation is redistributed between angular directions based on the expected angular distribution. This can even be done globally for an entire simulation domain with no increase in computational complexity, enabling larger and more accurate simulations of the evolution of the universe than currently possible.

# Functionality

The `qlbm_rt` module features the `simulate` method which is called to perform simulations with the lattice Boltzmann method. This method constructs the full quantum circuit for each timestep of the simulation and returns the lattice data.

The `qlbm_circuits` module features constructors for the necessary circuits for radiative transfer simulation in 1D, 2D, and 3D, including a constructor for the novel angular redstribution step. These constructors are called by the `simulate` method which composes them to construct the full quantum circuit.

`QuaRT` features a variety of utility methods for both general and quantum lattice Boltzmann methods in `lbm_utils` and `qlbm_utils`, respectively. It also features analysis utilities in the `analysis` module. These utilities are used by the `simulate` method for problem setup and analysis.

The `test` module features a variety of common test cases used for radiative transfer codes, including the isotropic source, opaque cloud shadow, and crossing radiation beams tests. These tests demonstrate the general correctness of the codebase, with a particular emphasis on the performance of the angular redistribution methodology.

# Scholarly Work

`QuaRT` is currently being used to study lattice Boltzmann methods for radiative transfer (see upcoming [@Devkota2025]).

# Acknowledgements

This research was supported in part through research cyberinfrastructure resources and services provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology, Atlanta, Georgia, USA [@PACE].

# References



