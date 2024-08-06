# UQ_tutorial_code
This repository contains MATLAB and Python code for the paper "Taming Uncertainty in a Complex World: The Rise of Uncertainty Quantification — A Tutorial for Beginners", by [Nan Chen](https://people.math.wisc.edu/~nchen29/), [Stephen Wiggins](https://scholar.google.com/citations?user=FmdPqIUAAAAJ&hl=en), and [Marios Andreou](https://sites.google.com/wisc.edu/mariosandreou/homepage).

The MATLAB codebase was tested on version R2020b, while the Python codebase was tested on version 3.10.6.

# Script Files
The following MATLAB m-file (script) files can be found in the respective subdirectories (the numbered sections correspond to the ones found in the Supplementary Document of the paper, which is available in the [arXiv](https://arxiv.org/abs/2408.01823#) version of this work):
* [2.1 - Shannon's entropy](https://github.com/marandmath/UQ_tutorial_code/tree/main/2.1%20-%20Shannon's%20entropy)
  + Computing_Entropy.m
* [2.2 - Relative entropy](https://github.com/marandmath/UQ_tutorial_code/tree/main/2.2%20-%20Relative%20entropy)
  + Computing_Relative_Entropy.m
* [3.1 - Uncertainty propagation in the linear damped system](https://github.com/marandmath/UQ_tutorial_code/tree/main/3.1%20-%20Uncertainty%20propagation%20in%20the%20linear%20damped%20system)
  + Linear_System_UQ.m
* [3.2 - Uncertainty propagation in the chaotic Lorenz 63 model](https://github.com/marandmath/UQ_tutorial_code/tree/main/3.2%20-%20Uncertainty%20propagation%20in%20the%20chaotic%20Lorenz%2063%20model)
  + L63_UQ.m
* [4.1 - Uncertainties in posterior distributions](https://github.com/marandmath/UQ_tutorial_code/tree/main/4.1%20-%20Uncertainties%20in%20posterior%20distributions)
  + Bayes_Formula.m
* [4.2 - Lagrangian DA](https://github.com/marandmath/UQ_tutorial_code/tree/main/4.2%20-%20Lagrangian%20DA)
  + Flow_Model.m
  + LDA_Function_of_L.m
  + LDA_Main_Filter.m
* [5.1 - Parameter estimation with uncertainties in data](https://github.com/marandmath/UQ_tutorial_code/tree/main/5.1%20-%20Parameter%20Estimation%20with%20Uncertainties%20in%20Data)
  + Parameter_Estimation.m
* [5.2 - Eddy identification](https://github.com/marandmath/UQ_tutorial_code/tree/main/5.2%20-%20Eddy%20identification)
  + Eddy_Identification.m
  + Flow_Model.m
  + LDA_Main_Smoother.m
* [6 - Calibrating Stochastic Models Based on UQ](https://github.com/marandmath/UQ_tutorial_code/tree/main/6%20-%20Calibrating%20Stochastic%20Models%20Based%20on%20UQ)
  + Calibrating_Stochastic_Model_with_UQ.m

The following Python script files can be found in the respective subdirectories (the numbered sections correspond to the ones found in the Supplementary Document of the paper, which is available in the [arXiv](https://arxiv.org/abs/2408.01823#) version of this work):
* [2.1 - Shannon's entropy](https://github.com/marandmath/UQ_tutorial_code/tree/main/2.1%20-%20Shannon's%20entropy)
  + Computing_Entropy.py
* [2.2 - Relative entropy](https://github.com/marandmath/UQ_tutorial_code/tree/main/2.2%20-%20Relative%20entropy)
  + Computing_Relative_Entropy.py
* [3.1 - Uncertainty propagation in the linear damped system](https://github.com/marandmath/UQ_tutorial_code/tree/main/3.1%20-%20Uncertainty%20propagation%20in%20the%20linear%20damped%20system)
  + Linear_System_UQ.py
* [3.2 - Uncertainty propagation in the chaotic Lorenz 63 model](https://github.com/marandmath/UQ_tutorial_code/tree/main/3.2%20-%20Uncertainty%20propagation%20in%20the%20chaotic%20Lorenz%2063%20model)
  + L63_UQ.py
* [4.1 - Uncertainties in posterior distributions](https://github.com/marandmath/UQ_tutorial_code/tree/main/4.1%20-%20Uncertainties%20in%20posterior%20distributions)
  + Bayes_Formula.py
* [4.2 - Lagrangian DA](https://github.com/marandmath/UQ_tutorial_code/tree/main/4.2%20-%20Lagrangian%20DA)
  + Flow_Model.py
  + LDA_Function_of_L.py
  + LDA_Main_Filter.py
* [5.1 - Parameter estimation with uncertainties in data](https://github.com/marandmath/UQ_tutorial_code/tree/main/5.1%20-%20Parameter%20Estimation%20with%20Uncertainties%20in%20Data)
  + Parameter_Estimation.py
* [5.2 - Eddy identification](https://github.com/marandmath/UQ_tutorial_code/tree/main/5.2%20-%20Eddy%20identification)
  + Eddy_Identification.py
  + Flow_Model.py
  + LDA_Main_Smoother.py
* [6 - Calibrating Stochastic Models Based on UQ](https://github.com/marandmath/UQ_tutorial_code/tree/main/6%20-%20Calibrating%20Stochastic%20Models%20Based%20on%20UQ)
  + Calibrating_Stochastic_Model_with_UQ.py
 
The figures that are produced by each script can also be found in the corresponding subdirectory both in .png and MATLAB's .fig figure source format for quick reference. Output files (whenever applicable), for both MATLAB and Python, are also available in .txt format. Finally, the Python module/package requirements for the script file of each section (of the paper's Supporting Document) are also located in each subdirectory with the name "Python_Requirements.txt", and are in a format that is understood by package managers (e.g. pip or conda).

# Citing This Work

[[arXiv](https://arxiv.org/abs/2408.01823#)]

BibTeX Entry:

# License
This code is released under the MIT License. See the file [LICENSE](https://github.com/marandmath/UQ_tutorial_code/blob/main/LICENSE) for copying permission.
