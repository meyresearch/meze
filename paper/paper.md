---
title: "meze: A Metalloenzyme parameterisation program"
tags:
  - Python
  - molecular dynamics
  - metal modelling
  - computational chemistry
authors:
  - name: J. Jasmin Güven
    orcid: 0000-0003-1555-0075
    equal-contrib: false
    affiliation: 1 
  - name: Antonia S.J.S. Mey
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 2
  - name: Adrian J. Mulholland
    corresponding: true
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: Centre for Computational Chemistry, School of Chemistry, University of Bristol, United Kingdom
   index: 1
 - name: EaStCHEM School of Chemistry, The University of Edinburgh, United Kingdom
   index: 2
date: 9 April 2026
bibliography: paper.bib
---

# Summary

Metalloenzymes -- enzymes containing transition metals -- are abundant in nature and important in pharmaceutical research. Despite their importance as drug discovery targets, computational methods for preparing accurate models for enzymes containing transition metals are scarce. Computational methods such as quantum mechanics molecular mechanics (QM/MM), molecular dynamics (MD) simulations and relative binding free energy (RBFE) calculations are widely used in drug discovery applications both in academia and industry. However, applying these methods to metalloenzymes is often a convoluted and time-consuming process. 


# Statement of need

Parameterising metalloenzymes for MD simulations and RBFE calculations is a convoluted and time-consuming process, often requiring manual input and computationally expensive QM calculations. `Meze`[^1] is a metalloenzyme-parameterisation program, which can be used to prepare and run QM/MM, MD or RBFE workflows for a system of small molecules binding to a zinc-metalloenzyme, using three different metal modelling options. The goal of `meze` is to simplify the setup of these models, and enable them to be easily applied to MD and RBFE workflows at scale. `Meze` was built to bring together different metal model options into one place for ease of use.

[^1]: Meze is a Turkish side dish or appetizer.

# State of the field

*A description of how this software compares to other commonly-used packages in the research area. If related tools exist, provide a clear “build vs. contribute” justification explaining your unique scholarly contribution and why existing alternatives are insufficient.*

 Metal models in MD are generally grouped into *bonded*, *nonbonded* and *cationic dummy atom* models. In bonded models, covalent bond and angle terms are added to the force field to describe the coordination bonds between a metal and its ligands. An example of a bonded zinc force field is the zinc Amber force field (ZAFF) [@peters_structural_2010]. ZAFF contains parameters for 12 tetrahedrally coordinated zinc metalloenzymes, and it was built with one of the more widely used open-source metalloenzymes parameterisation tools, called the metal center parameter builder (MCPB.py) [@li_mcpbpy_2016], part of AmberTools~24 [@case_ambertools_2023]. MCPB.py is a Python-based tool for developing and generating force field parameters for running molecular dynamics (MD) simulations with metalloenzymes. With MCPB.py, bespoke bonded force field terms may be generated for many transition metals. Alternatively, the user may choose the extended zinc Amber force field (eZAFF) [@yu_extended_2018] bonded model, which contains empirically determined bonded force field terms for zinc-coordinating residues. The partial charges of atoms in the metal-containing active site have to be determined from quantum mechanical (QM) density functional theory (DFT) calculations. 

Currently, there is no direct Python interface for including MCPB.py-developed force fields, or other metal modelling options, in open-source RBFE tools, such as BioSimSpace [@hedges_biosimspace_2019]. Furthermore, these bonded models are not directly applicable to RBFE workflows focused on small molecule transformations. Nonbonded metal models are more readily applied to these pipelines, because they do not contain bonded parameters between the metal and its coordinating residues, and often replace them with e.g. harmonic distance restraints or redefined electrostatics and van der Waals terms [@peters_structural_2010]. For example, Li and Merz Jr. developed a restrained nonbonded model to complement MCPB.py-developed force fields [@li_building_2015], where distance restraints are applied in place of zinc-ligand coordination bonds, while the partial charges obtained from DFT calculations are retained. In [@guven_protocols_2024], we applied this model to RBFE calculations with a metallo-$\beta$-lactamase (MBL), by applying flat-bottomed distance restraints to zinc coordination, while keeping the partial charges of the underlying force field unchanged. In [@guven_protocols_2024], we also used another nonbonded zinc force field developed by Macchiagodena et al. [@macchiagodena_upgrading_2019; macchiagodena_upgraded_2020] in RBFE calculations. In this *upgraded Amber force field* (UAFF), the partial charges and van der Waals terms of zinc-coordinating amino acids are redefined. The zinc ion is given a charge of $+2$e, while its van der Waals radius is adjusted. 

# Software design

*An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description.*

Currently, `meze` includes workflows for preparing and carrying out both MD and RBFE workflows with zinc-containing metalloenzymes. Figure \autoref{fig:md-and-meze-workflow} a) shows an overview of the general steps of an MD or RBFE workflow, and b) shows where `meze` fits in these workflows. Both MD and RBFE workflows start with selecting either an experimental X-ray or NMR structure, or a machine learning (ML) predicted structure of the enzyme. For MD simulations with small molecules bound, and for RBFE calculations, a high-quality experimental crystal structure with the small molecule bound to the active site of the enzyme is desirable. The user has to then decide on e.g. which protonation states to use for amino acid side chains and the small molecule, or whether to retain crystallised waters or co-factors such as transition metals or other molecules. These steps are also pre-requisites for using `meze`. Depending on the system, preprocessing may also include fixing gaps in the amino acid sequence and adding terminal caps. A guide to preprocessing enzyme files can be found in [@degiacomi_course_2025]. 

![a) The general steps for a molecular dynamics (MD) or relative binding free energy (RBFE) workflow. b) The metalloenzyme parameterisation program `meze` covers a part of the system preparation stages and has scripts to carry out system equilibration, production runs and basic analysis of MD simulations or RBFE calculations for systems containing small molecules bound to a metalloenzyme.\label{fig:md-and-meze-workflow}](figures/md_and_meze_workflow.png){ width=20% }

The molecules in the system, i.e. enzyme, water, ions, small molecule, etc., have to then be parameterised with a force field. Currently, the default enzyme force field included in `meze` is ff14SB [@maier_ff14sb_2015]. Since the code has been built on top of BioSimSpace functions and includes wrapper functions for running AmberTools-based parameterisations, changing the underlying enzyme force field is relatively easy. As discussed above, `meze` has two nonbonded metal modelling options included: 1) the restraint-based approach [@guven_protocols_2024] and the 2) upgraded Amber force field (UAFF) by Macchiagodena et. al. [@macchiagodena_upgraded_2020; macchiagodena_upgrading_2019]. These parameters can be defined before the RBFE preparation step (described below). Furthermore, the usage of UAFF requires some manual input from the user for the renaming of zinc-bound residues. We have corrected some missing atom types in the force field parameter files, as described in [@guven_protocols_2024]. These parameter files are shared in this [GitHub repository](https://github.com/meyresearch/alchemistry_with_betalactamases/tree/main/input_data/vim2/uaff/enzyme/parameters).

Following the system preparation, both MD and RBFE workflows include a system minimisation and equilibration stage. This step ensures that the system is at a local energy minimum, and at the correct temperature and pressure. Then, if this is an RBFE workflow, the alchemical transformation network, merged small molecule topologies as well as the $\lambda$ window simulation directories have to be prepared. Then, either the MD or individual $\lambda$ production simulations are carried out, often including repeat runs. These three steps, equilibration, RBFE preparation and production simulations can all be carried out within `meze`. The user can also input options for each of the steps, such as the length of equilibration simulations, number of $\lambda$ windows and number of repeat simulations. Depending on computational resources, the whole pipeline can be submitted entirely using one command. Finally, at the end of the workflow, the simulations can be analysed. For RBFE calculations, `meze` contains scripts and functions for obtaining $\Delta\Delta$G estimates from MBAR within BioSimSpace. When the $\lambda$ simulations are analysed, `meze` will also produce analysis figures for the phase space overlap and forward-backward time convergence from the alchemlyb [@wu_alchemlyb_2024] software package. 



# Research impact statement

*Evidence of realized impact (publications, external use, integrations) or credible near-term significance (benchmarks, reproducible materials, community-readiness signals). The evidence should be compelling and specific, not aspirational.*

We have used `meze` to prepare the restraint approach and the UAFF model for the MBL VIM-2 to carry out MD and RBFE workflows [@{guven_protocols_2024]. We also used `meze` to prepare the serine-$\beta$-lactamase (SBL) KPC-2 for both MD simulations and RBFE calculations. While these results show that the code works for its intended purpose, i.e. preparing metalloenzymes for RBFE calculations, some improvements in usability and code design are needed. For example, for greater accessibility, the code should be made available for installation via pip or conda. This would also help to ensure the installation of correct dependencies. Furthermore, due to the object-oriented structure of the code, it would be helpful to make use of Python's data classes and either replace or supplement the command line arguments with JSON-file inputs and outputs. Such structures have been used for example by the Open Force Field initiative's BespokeFit [@{horton_open_2022] to improve reproducibility of workflows across users. Additionally, future versions of `meze` will further automate the existing non-bonded options and include new metal modelling options, such as including angle restraints as well as distance restraints, as well as preparing a hybrid nonbonded model for MD and RBFE calculations. Finally, the code could also include other transition metals in a semi-automated way for both molecular dynamics and RBFE calculations. While it has only been tested with an SBL and a zinc MBL, expanding the code's scope to include other transition metals and metal modelling options is simple due to the modular, and object oriented structure of the code.

# AI usage disclosure

*Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified.*

# References
