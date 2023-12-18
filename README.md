# Bittensor 3D Asset Extension

## 3D Bittensor is a 3D asset creation extension for the bittensor network.

### Install

Clone the repository:
```bash
git clone https://github.com/404-Repo/3DAssetSubnet.git
cd 3DAssetSubnet
```

Install Python Environment:
```bash
pip install -r requirements.txt
```

Install shap-e:
```bash
pip install git+https://github.com/openai/shap-e.git
```

Install submodules:
```bash
git submodule update --recursive --remote
```

Install Build Environment (MacOS):
```bash
xcode-select --install
brew install llvm
```

> Troubleshooting: If you get an error try reinstalling `xcode-select` and `CommandLineTools`: [See StackOverflow](https://stackoverflow.com/questions/58897928/macos-sdk-headers-for-macos-10-14-pkg-is-incompatible-with-this-version-of-maco). If re-install fails go directly to [Apple Developer](https://developer.apple.com/download/all/) page and download the latest `Command Line Tools for Xcode` and install manually.

See install instructions for other platforms in the respective submodules.

### Build LLaVA C++ Server:
```bash
cd llava-cpp-server
make
```

### Download Models:
Download one of `ggml-model-*.gguf` and `mmproj-model-f16.gguf` from [here](https://huggingface.co/mys/ggml_llava-v1.5-13b/tree/main).


### Run LLaVA C++ Server:
```bash
bin/llava-server -m ggml-model-q5_k.gguf --mmproj mmproj-model-f16.gguf
```

### Configure Neural-Captioning Engine
Run `symwzd` in your terminal to create a default configuration for `symbolicai`.
Set the `.symai/symai.config.json` `CAPTION_ENGINE_MODEL` property to `llavacpp`:
```json
{
  ...
  "CAPTION_ENGINE_MODEL": "llavacpp"
  ...
}
```
See further details [here](https://github.com/ExtensityAI/symbolicai).

### Install embeddings extensions:
```bash
sympkg i ExtensityAI/embeddings
```

### Read the Docs:
Open the [Docs.ipynb](Docs.ipynb) notebook to read a summary of the Bittensor repository.

### Try it out:
Open the Jupyter notebook [Experiment.ipynb](Experiment.ipynb) and run the cells to run initial experiments for the 3D asset extension.
