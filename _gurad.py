import onnxruntime as ort

# Retrieve the list of available execution providers
providers = ort.get_available_providers()

# Check if GPU (CUDA) is available
if "CUDAExecutionProvider" in providers:
    print("GPU is available.")
else:
    print("GPU is not available. Using CPU.")
