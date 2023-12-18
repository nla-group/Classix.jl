# Classix.jl
Fast and explainable clustering in Julia.

Basic usage:
```
using Classix

data = [8.0391 11.3790 9.7221;
9.8023    8.9418   10.7015;
-9.7180  -10.2991  -10.8314;
-9.9665   -9.9771  -10.9792;
8.7922    9.5314    7.9482;
12.9080    9.7275    9.6462;
10.8252   11.0984    9.1764;
-11.5771   -8.8725  -11.7502;
-9.4920   -9.6498  -10.2857;
-11.3337  -10.2620  -11.1564]

labels, explain, out = classix(data, radius=0.2, minPts=1, merge_tiny_groups=true)

```
