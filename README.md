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
### Compilation & Distillation
#### BERT and DistilBERT Latency when Compiled with XLA 
<img src="/results/bert/bert-latency.png" alt="BERT Latency" width="650"/>

#### GPT2 and DistilGPT2 Latency when Compiled with XLA 
<img src="/results/gpt2/gpt2-latency.png" alt="GPT2 Latency" width="650"/>

### Depthwise Separable Convolutions
<img src="/results/depthwise/convolutions-perf-plot.png" alt="Separable Convolutions Latency" width="650"/>

### Pruning & Quantization

#### LeNet-5
|          | Model Size [MiB] | Accuracy | Compression Factor | Accuracy Change |
|----------|--------:|---------:|---------:|---------:|
| Baseline | 0.16   |    98.78%  | 1.0x | ± 0.00% |
| Pruned | 0.05  |    98.89%  | 3.2x | + 0.11% |
| Quantized | 0.04   |    98.86%  | 4.0x | + 0.08% |
| Quantized & Pruned | 0.02   |    98.78%  | 8.0x | ± 0.00% |

#### ResNet-50
|          | Throughput  [images/second] | Latency [ms] | dtype | Model Size [MiB] | Speedup |
|----------|--------:|---------:|---------:|---------:|---------:|
| Baseline           | 1117.16 | 0.930 | float32 | 98 | 1.00x |
| Sparse             | 1185.75 | 0.875 | float32 | 98 | 1.06x |
| Quantized          | 2053.65 | 0.522 | int8    | 25 | 1.78x |
| Quantized & Sparse | 2123.08 | 0.507 | int8    | 27 | 1.83x |

