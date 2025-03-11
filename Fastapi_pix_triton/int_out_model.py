# import onnx
# import os

# # Load mô hình ONNX
# model = onnx.load(r"D:\SourceCode\ProjectOJT\complete\OJT_TASK3_LOCAL\Deploy\pix_triton\model\text_encoder\1\model.onnx")

# #thông tin input/output
# def extract_io_info(graph):
#     def get_info(tensors):
#         info = []
#         for tensor in tensors:
#             name = tensimport onnx

# model_path = "path/to/your/text_encoder.onnx"
# model = onnx.load(model_path)

# for output in model.graph.output:
#     print(f"Name: {output.name}, Shape: {[dim.dim_value if dim.dim_value > 0 else -1 for dim in output.type.tensor_type.shape.dim]}")
# or.name
#             data_type = tensor.type.tensor_type.elem_type
#             dims = [d.dim_value if d.dim_value > 0 else -1 for d in tensor.type.tensor_type.shape.dim]
#             info.append({"name": name, "data_type": data_type, "dims": dims})
#         return info

#     return get_info(graph.input), get_info(graph.output)

# #thông tin
# input_info, output_info = extract_io_info(model.graph)

# print("Inputs:", input_info)
# print("Outputs:", output_info)
import onnx
import os

model_path = r"D:\SourceCode\ProjectOJT\complete\OJT_TASK3_LOCAL\Deploy\pix_triton\model\text_encoder\1\model.onnx"
model = onnx.load(model_path)

for output in model.graph.output:
    print(f"Name: {output.name}, Shape: {[dim.dim_value if dim.dim_value > 0 else -1 for dim in output.type.tensor_type.shape.dim]}")

