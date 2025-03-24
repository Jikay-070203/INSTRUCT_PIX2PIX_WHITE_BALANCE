import onnx
import onnxruntime as ort
import os       

def check_onnx_model(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    session = ort.InferenceSession(model_path)
    model_inputs = session.get_inputs()
    model_outputs = session.get_outputs()
    
    print("Model Inputs:")
    for inp in model_inputs:
        print(f"  Name: {inp.name}, Type: {inp.type}, Shape: {inp.shape}")
    
    print("\nModel Outputs:")
    for out in model_outputs:
        print(f"  Name: {out.name}, Type: {out.type}, Shape: {out.shape}")

# Path model ONNX
model_path = r"D:\SourceCode\ProjectOJT\complete\OJT_TASK3_LOCAL\Deploy\WB\pix_triton\model\vae_encoder\1\model.onnx"
check_onnx_model(model_path)



