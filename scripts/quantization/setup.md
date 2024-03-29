# Setup
### Pull Image
`singularity pull docker://nvcr.io/nvidia/tensorrt:23.12-py3`

### Start Interactive GPU Session
`sintr -t 1:0:0 --exclusive -A COMPUTERLAB-SL3-GPU -p ampere`

### Start container with Nvidia support
`singularity shell --nv /home/fjr38/rds/hpc-work/tensorflow_23.12-tf2-py3.sif`

### Run INT8 Quantization Test
`trtexec --onnx=scripts/quantization/model_qat.onnx --int8 --saveEngine=scripts/quantization/model_qat.engine --verbose &> results/quantization/A100/quantized_output.log`

### Run Baseline Test
`trtexec --onnx=scripts/quantization/model_baseline.onnx --saveEngine=scripts/quantization/model_baseline.engine --verbose &> results/quantization/A100/baseline_output.log`

### Run Baseline Sparsity Test
`trtexec --onnx=scripts/quantization/model_baseline.onnx --sparsity=force --saveEngine=scripts/quantization/model_baseline.engine --verbose &> results/quantization/A100/baseline_sparsity_output.log`

### Run INT8 Quantization & Sparsity Test
`trtexec --onnx=scripts/quantization/model_qat.onnx --int8 --sparsity=force --saveEngine=scripts/quantization/model_qat.engine --verbose &> results/quantization/A100/quantized_sparsity_output.log`