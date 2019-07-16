Vector quantization and classification

Goal : Classify variable length time-series data into 14 activities

Data : https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer


Tasks coded:
 1) Segment the vector 
 2) k-means clustering of segments : Hierarchical k-means was performed
 3) Generating histogram as signature of a vector 
 4) Classification using random forest 


Top-level is Part2Top.py

The second half of the report in report.pdf has additional details 

Experiments were done by varying the segment size and cluster size. 
