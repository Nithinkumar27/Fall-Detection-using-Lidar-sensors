# pointnet_code
## Results:
A detailed report on the project is given in this link:https://docs.google.com/document/d/1fdTEQS1sVJ6R9LsCr-Z3WNP0fxfVCj49Jo0h6hCfe7E/edit?usp=sharing

### 70-30 split:
#### Training Graph:
![image](https://github.com/mcwfall/pointnet_code/assets/83899953/5234ff6a-d090-447d-80c3-54bd43e3c709)

#### Metrics:
Metric Name|Value
---|---
Accuracy|0.857
---|---
Precision|0.854
---|---
Recall|0.857
---|---
F1 Score|0.854
---|---
Specificity|0.9

### 80-20 split:
#### Training Graph:
![image](https://github.com/mcwfall/pointnet_code/assets/83899953/ec3c65d1-59ae-4587-85c3-c7fec1386fec)

#### Metrics:
Metric Name|Value
---|---
Accuracy|0.897
---|---
Precision|0.902
---|---
Recall|0.897
---|---
F1 Score|0.895
---|---
Specificity|0.930
## Setup:
1. the code is present in `pointnet_github/pointnet2/pointnet2_pytorch (3).ipynb`
2. the data and the weights are present in the `data/70-30`  and `data/80-20` for the 70-30 and the 80-20 split respectively 
3. load `train.npy`,`train_labels.npy`,`test.npy` and `test_labels.npy` into  `train_points`,`train_labels`,`test_points` and `test_labels` respectively
4. the model weights can also be loaded with the file present in the path mentioned in point 2
