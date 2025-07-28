Data Repository
===============

Source code and data files for the manuscript. 

How to cite
-----------
If this work is used, please cite "Hamiltonian parameter inference from RIXS spectra with active learning", Marton K. Lajer, Xin Dai, Kipton Barros, Matthew R. Carbone, S. Johnston, and M. P. M. Dean

This work is based on data from the papers

- NiPS3: W. He, Y. Shen, K. Wohlfeld, J. Sears, J. Li, J. Pelliciari, M. Walicki, S. Johnston, E. Baldini, V. Bisogni, M. Mitrano, and M. P. M. Dean, Magnetically propagating Hund’s exciton in van der Waals antiferromagnet, NiPS3, Nature Communications 15 (2024).

- NiCl2: C. A. Occhialini, Y. Tseng, H. Elnaggar, Q. Song, M. Blei, S. A. Tongay, V. Bisogni, F. M. F. de Groot, J. Pelliciari, and R. Comin, Nature of excitons and their ligand-mediated delocalization in nickel dihalide charge-transfer insulators, Phys. Rev. X 14, 031007 (2024)

- Fe2O3: J. Li, Y. Gu, Y. Takahashi, K. Higashi, T. Kim, Y. Cheng, F. Yang, J. Kuneˇs, J. Pelliciari, A. Hariki, and
V. Bisogni, Single- and Multimagnon Dynamics in Anti-ferromagnetic α -Fe2O3 Thin Films, Physical Review X
13, 011012 (2023).

- Ca3LiOsO6: A. E. Taylor, S. Calder, R. Morrow, H. L. Feng, M. Upton, M. D. Lumsden, K. Yamaura, P. M. Woodward, A. D. Christianson, and A. D. Christianson, Spin-orbit coupling controlled J = 3/2 electronic ground state in 5d3 oxides., Physical review letters 118 20, 207202 (2016).

Please cite these as appropriate.


Run locally
-----------

Work with this by installing `docker <https://www.docker.com/>`_ and pip and then running

.. code-block:: bash

       pip install jupyter-repo2docker
       jupyter-repo2docker  --editable --Repo2Docker.platform=linux/amd64 .

Change `tree` to `lab` in the URL for JupyterLab.

Run remotely
------------

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/mpmdean/He2024dispersive/HEAD?filepath=plot.ipynb


