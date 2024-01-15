# Compressed Model Inference
This repository contains small experiments on how different optimization and compression techniques affect a model's latency. 

## Included Experiments: 
1. Compilation using XLA 
2. Depthwise Separable Convolutions
3. Pruning
4. Quantization

## Experimental Setup
### Hardware
- Apple M3 Max with 64 GB RAM
- NVIDIA A100-SXM-80GB GPU with an AMD EPYC 7763 64-Core 1.8GHz CPU

### Software
#### Mac
- TensorFlow 2.12.0
- Python 3.8.15
#### A100
- NVIDIA’s TensorFlow NGC Container (tag: 23.12-tf2-py3)

## Results
### BERT and DistilBERT Latency when Compiled with XLA 
<img src="/results/bert/bert-latency.png" alt="BERT Latency" width="650"/>

### GPT2 and DistilGPT2 Latency when Compiled with XLA 
<img src="/results/gpt2/gpt2-latency.png" alt="GPT2 Latency" width="650"/>

### Depthwise Separable Convolutions
<img src="/results/depthwise/convolutions-perf-plot.png" alt="Separable Convolutions Latency" width="650"/>
