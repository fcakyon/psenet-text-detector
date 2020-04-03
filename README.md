## PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network
Packaged Version of the Pytorch implementation of PSENet text detector

### Overview
PSENet is designed as a segmentation-based detector with multiple predictions for each text instance. These predictions correspond to different `kernels' produced by shrinking the original text instance into various scales. Consequently, the final detection can be conducted through our progressive scale expansion algorithm which gradually expands the kernels with minimal scales to the text instances with maximal and complete shapes.

THIS REPO IS A WORK ON PROGRESS
