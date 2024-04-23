# Time Series Missing Imputation with Multivariate Radial Basis Function Neural Network	Time Series Missing Imputation with Multivariate Radial Basis Function Neural Network
- we propose a time series imputation model based on RBFNN.
-  Our imputation model learns local information from timestamps to create a continuous function. Additionally, we incorporate time gaps to facilitate learning information considering the missing terms of missing values. We name this model the Missing Imputation Multivariate RBFNN (MIM-RBFNN).
- However, MIM-RBFNN relies on a local information-based learning approach, which presents difficulties in utilizing temporal information.
- Therefore, we propose an extension called the Missing Value Imputation Recurrent Neural Network with Continuous Function (MIRNN-CF) using the continuous function generated by MIM-RBFNN.
### MIM-RBFNN
![MIM-RBFNN](https://github.com/cooling-0/MissingRBF/blob/main/MIM-RBFNN.jpg)
### MIRNN-CF
![MIRNN-CF](https://github.com/cooling-0/MissingRBF/blob/main/MIRNN-CF2.jpg)
### Solving long term missing with MIRNN-CF

