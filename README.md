# 3D Bittensor
## 3D Bittensor is a 3D asset creation extension for the bittensor network.


### Install

Install submodules:
```bash
git submodule update --recursive --remote
```

Install Build Environment:
```bash
xcode-select --install
brew install llvm
```

> Troubleshooting: If you get an error try reinstalling `xcode-select` and `CommandLineTools`: [See StackOverflow](https://stackoverflow.com/questions/58897928/macos-sdk-headers-for-macos-10-14-pkg-is-incompatible-with-this-version-of-maco). If re-install fails go directly to [Apple Developer](https://developer.apple.com/download/all/) page and download the latest `Command Line Tools for Xcode` and install manually.