# MLops_G2JN: Automatic Pipeline for fixing uncertainties
## Introduction
We present in this repo our pipeline spotting uncertaincies in a given dataset, and then applying automatic solution on inspected epistemic and aleatoric uncertainties.<br>
- For inspecting uncertainties we use [MACEst](https://github.com/oracle/macest).
- For data generation we use [SDV](https://github.com/sdv-dev/SDV).
- For outliers removal we use [PyOD](https://github.com/yzhao062/pyod). 

![Our proposed pipeline](https://github.com/gilzeevi25/MLops_G2JN/blob/master/utils/pipeline.PNG)
## Running the pipeline
In order to activate our pipeline you can do the follwoing steps:<br>
1. Press [Start_G2JN.ipynb](https://github.com/gilzeevi25/MLops_G2JN/blob/master/Start_G2JN.ipynb)
2. Run the first cell:
```
%%capture
!pip install macest
!pip install sdv
!pip install pyod
import os
os.kill(os.getpid(), 9)
```
It will crash after finishing the installments - thats ok!<br>
3. Run the following cell and watch output:
```
!git clone https://github.com/gilzeevi25/MLops_G2JN 
%cd MLops_G2JN
!python main.py 
```

## Thanks for visiting
 ![visitors](https://visitor-badge.glitch.me/badge?page_id=gilzeevi25.MLops_G2JN.issue.1) <br/>



