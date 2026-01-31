# MultiTaskMPN
zhixin is working on this branch.

The model is a MPN with directed acyclic graph, which is more general than multi-layer MPN.
The current code (300 nodes) works in a A100 GPU for seq-MNIST (batch size=640), with performace being >79% after 20 epoches
Currently, Zhixin is still debugging the code to find out the memory leak through the batches. 

