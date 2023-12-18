# Evaluating 3D Assets with LLaVA

The future of asset evaluation in the realm of three-dimensional modeling and rendering is poised for transformation with the introduction of novel techniques that harness artificial intelligence. This blog post presents a comprehensive guide to using the Bittensor network's 3D asset extension, leveraging RewardModel and LLaVA (Lightweight Language and Vision Assistant) for automated evaluation of 3D assets. We delve into the intricacies of the system, offering insights into its setup, functionality, and potential applications.

## Introduction to Bittensor and its 3D Asset Extension

[Bittensor](https://github.com/opentensor/bittensor) is an innovative platform designed to decentralize machine learning, where nodes on the network provide services such as dataset storage, processing power, or algorithms. With the latest advancements, Bittensor has extended its capabilities into the world of three-dimensional assets. The realm of 3D asset creation and distribution is expanding rapidly, and the need for an efficient assessment system is more pronounced than ever. This is where the Bittensor 3D asset extension shines, delivering a streamlined environment for the creation, exchange, and validation of 3D assets using artificial intelligence.

## RewardModel: A Mechanism for Asset Similarity Evaluation

The RewardModel is a Python class, crafted to work seamlessly with symbolic AI methodology, to assess and compare 3D assets. Let's unravel how this sophisticated tool functions to revolutionize 3D asset evaluation.

### Implementation Details of RewardModel

The RewardModel, found in the `func.py` file, is structured as a subclass of the `Expression` class, an essential concept in symbolic AI. The RewardModel is initialized by defining several parameters such as `in_memory`, `metric`, and `aggregation`, which configure its caching behavior, similarity metric, and aggregation function, respectively.

Its primary functions are:

- **_dynamic_cache**: This method caches the embeddings of the reference items in memory for quick retrieval during subsequent similarity comparisons.

- **forward**: This is the core function that computes the similarity between images and a set of referenced embeddings. The images are initially described through an AI model (via the `model` interface), followed by embedding with the `embed` import. Following the embeddings, similarity scores are computed and aggregated to yield a final reward representation.

The RewardModel utilizes the `llava` interface to discern descriptions of the images, and it leverages embeddings generated through `ExtensityAI/embeddings` to quantify similarity. This comprehensive assessment encapsulates three major operations: embedding references, embedding target images, and comparing them based on the selected metric (e.g., Jaccard index).

### RewardModel's Role in Asset Evaluation

The RewardModel's unique selling propositions are its ability to measure similarity between assets and its flexible integration with various other components. When evaluating 3D assets, such as images of models or renders, the RewardModel can be trained on a set of reference assets to establish a baseline for quality or style. Subsequently, it can process new assets to estimate their similarity to the references, providing an automated approach to assessing consistency and relevance within a body of work.

### Aggregation Strategies

The RewardModel supports three aggregation strategies for the similarity scores – 'sum', 'mean', and 'median'. These options ensure that users can fine-tune the evaluation process based on the specific needs of the assessment task, whether it prioritizes collective similarity, average representation, or robustness against outliers in the reference set.

## LLaVA Server: The AI-Powered Descriptive Engine

The [LLaVA server](https://github.com/ggerganov/llama.cpp) functions as the backbone for processing natural language and visual data. Integrating the LLaVA server into the RewardModel workflow offers an AI-powered assistant that can describe images and produce embeddings that the RewardModel utilizes for similarity evaluation.

### Setting Up the LLaVA C++ Server

To integrate the LLaVA service into the Bittensor network, one must prepare the server environment:

- Install the necessary build tools and libraries, such as Xcode for MacOS.
- Clone and build the LLaVA server from its [repository](https://github.com/ggerganov/llama.cpp), making sure to initialize and update the required git submodules.
- Download appropriate models such as `ggml-model-*.gguf` and `mmproj-model-f16.gguf`, provided on [Hugging Face](https://huggingface.co/mys/ggml_llava-v1.5-13b/tree/main), which are necessary for the server's operation.

The LLaVA server is launched using a command-line interface, where users specify the model files and configure network parameters such as host and port. The server's API is neatly exposed over HTTP, accepting image data and prompts to return descriptions and conversational responses.

### Usage of LLaVA with RewardModel

Once the LLaVA server is operational, it collaborates with the RewardModel in the following manner:

1. An image representing a 3D asset is forwarded to the LLaVA server along with the user prompt (e.g., "Describe this 3D model").
2. LLaVA processes the image and returns a natural language description.
3. RewardModel then uses this description to create an embedding and compare it against cached reference embeddings.
4. The similarity scores are aggregated to form an evaluation metric for the 3D asset.

This synergy allows for the automated, intelligent analysis of 3D assets, consolidating the distinct capabilities of LLaVA's image understanding and RewardModel's symbolic AI processing.

## Applications and Implications

The combination of RewardModel and LLaVA paves the way for a plethora of potential applications. Notable among them is the capacity to sift through vast libraries of 3D assets, identifying those that match a particular style or quality criterion. It also opens doors for real-time feedback during the creation of 3D content, allowing designers to align their work more closely with established benchmarks.

In the context of decentralized networks like Bittensor, the RewardModel harmoniously fits into the ecosystem by providing a standard for valuing 3D assets, contributing to fair compensation for content creators based on the relevance and quality of their assets in comparison to the network's growing repository.

## Conclusion

Exploring the integration of the RewardModel with LLaVA in the Bittensor network heralds a new horizon in the evaluation and assessment of 3D assets. By embracing the power of symbolic AI and advanced machine learning, the RewardModel stands as a beacon of progress in the automated, intelligent analysis of 3D content. As we continue to refine these technologies, the potential to enhance asset creation, evaluation, and exchange is immense. The future is truly three-dimensional in the world of Bittensor, with every pixel and polycount now assessable through the discerning eyes of artificial intelligence.