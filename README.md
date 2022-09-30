# DCNN for Machine RUL Prediction using Time-series Data

In the session I will talk on RUL (Remaining Useful Life) estimation of a machine using sensor data. I will use a realistic multivariate time-series data for leveraging the power of deep neural networks in the hands-on. RUL is the remaining time or cycles that the machine is likely to operate without any failure. By estimating RUL the operator can decide the frequency of scheduled maintenance and avoid unplanned downtime. We will be focusing on building a DCNN (Deep Convolutional Neural Networks) model for the prediction.

[Download Data](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)

### Turbofan Jet Engine
<img src='pics/engine_schematic.jpg' width=500>

### Degradation Data for Prognostic Algorithm Development
<img src='pics/data_challenges.jpg' width=500>

### Damage Propagation Modeling
<img src='pics/operative_margins.jpg' width=500>

### Run to failure data
In figure, the degradation profiles of historical run-to-failure data sets from an engine are shown in blue and the current data from the engine is shown in red. Based on the profile the engine most closely matches, the RUL is estimated to be around 65 cycles.

<img src='pics/run_to_failure_plot.jpg' width=500>

### References:
[1] NASA Datasets: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

[2] https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

[3] Data Set Citation: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

[4] https://www.mathworks.com/company/newsletters/articles/three-ways-to-estimate-remaining-useful-life-for-predictive-maintenance.html 

[5] https://www.mathworks.com/help/predmaint/ug/remaining-useful-life-estimation-using-convolutional-neural-network.html

[6] https://github.com/datrikintelligence/stacked-dcnn-rul-phm21