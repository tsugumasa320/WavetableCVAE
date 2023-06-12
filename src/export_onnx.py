import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)
data_dir = root / "data"

import torch.onnx
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from src.models.cvae import LitCVAE


# Function to Convert to ONNX
def Convert_ONNX(model):
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 1, 600, requires_grad=True), torch.randn(1, 3, requires_grad=True)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,  # model input (or a tuple for multiple inputs)
        "wavetable_cvae.onnx",  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
        dynamic_axes={"modelInput": {0: "batch_size"}, "modelOutput": {0: "batch_size"}},  # variable length axes
    )
    print(" ")
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    # ckpt_path = "/Users/tsugumasayutani/Documents/GitHub/My-reserch-project/ckpt/2023-04-23-18:18:10.452469-LitCVAE-30000.ckpt"
    ckpt_path = "/Users/tsugumasayutani/Documents/GitHub/My-reserch-project/torchscript/files.ckpt"

    # モデルの読み込み
    model = LitCVAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )
    Convert_ONNX(model)
