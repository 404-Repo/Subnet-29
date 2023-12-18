# Evaluating 3D Assets with LLaVA

Asset evaluation in the realm of 3D modeling and rendering is a difficult undertaking, however, with the advances in contemporary machine learning models based on the Transformer architectures, we can build a pipeline to tackle it. This blog post presents a comprehensive introduction to 3D asset generation and evaluation. We leverage LLaVA (Large Language and Vision Assistant) and text embeddings for automated evaluation of 3D assets created by OpenAI's Shap-E. We delve into the intricacies of capturing multiple perspective views of 3D assets, and offer insights into functionality, and potential applications for the Bittensor project.

## Introduction to Bittensor and its 3D Asset Extensions

[Bittensor](https://github.com/opentensor/bittensor) is an platform designed to decentralize machine learning, where nodes on the network provide services such as dataset storage, processing power, or algorithms. With the latest advancements, Bittensor has extended its capabilities into the world of 3D assets. The realm of 3D asset creation and distribution is expanding rapidly, and the need for an efficient assessment system is more pronounced than ever. This is where an efficient Bittensor 3D asset extension shines most, covering a streamlined environment for the creation, exchange, and validation of 3D assets using modern ML-based architectures.

## 3D Asset Creation and Evaluation

3D assets are digital representations of real-world objects, created using specialized software. These assets are used in a variety of applications, such as video games, movies, and virtual reality. The process of creating 3D assets is a complex one, requiring a combination of artistic and technical skills. Such assets are created using specialized software, including Blender or Maya, which allow users to model, texture, and animate objects in a 3D environment. The assets are then exported into a format that can be used in other applications, such as game engines or video editing software.
Since 3D asset creation is a time-consuming process, it is important to generate assets that are of high quality. This is where Shap-E sets first accents to 3D asset creation. Shap-E is a neural network that generates 3D assets from a text or image descriptions.  The network is trained on a dataset of 3D assets and their corresponding text descriptions. The resulting models are trained to generate 3D assets that are similar to the ones in the dataset and allow conditioning to generate interpolations between representations. However, the quality of the generated assets is not always consistent, and it is difficult to evaluate the quality of the generated assets. This is where our work comes into play.

## A Mechanism for Asset Similarity Evaluation

We created this project to introduce a means to evaluate 3D assets for the Bittensor project. Our main component is the `RewardModel` class, which is crafted to work seamlessly with a compositional methodology, to assess and compare 3D assets. Let's unravel how this tool functions for the evaluation of 3D assets.
Conventionally, we first create 3D assets using Shap-E, and then we create images from different views of the 3D model from multiple angles and use the `RewardModel` module to evaluate the quality of the generated assets. The `RewardModel` uses LLaVA to generate a description of the 3D asset, and then it uses text embeddings to compare the generated description with a set of reference descriptions. In addition we verify the consistency of the generated asset with the normalized text-to-image alignment of CLIP embeddings. The `RewardModel` then aggregates the similarity scores to produce a final evaluation metric for the 3D asset.

### Implementation Details of RewardModel

The RewardModel, found in the `src/func.py` file, is structured as a subclass of the `Expression` class, an essential concept in the SymbolicAI framework. Holding a symbolic representation as opposed to purely embedding-based approaches, is helpful for explainability and human verification. The `RewardModel` is initialized by defining several parameters such as `in_memory`, `metric`, and `aggregation`, which configures its caching behavior, similarity metric, and aggregation function behavior, respectively.

Its primary functions are:
- **Interface**: The project interfaces to several ML modules, including a LLaVA server for captioning images, CLIP embeddings for text-to-image consitency verification, and Sentence Transformers embeddings for similarity comparisons, provisioned through the `ExtensityAI/embeddings` package.

- **_dynamic_cache**: This method caches the embeddings of the reference items in memory for quick retrieval during subsequent similarity comparisons.

- **forward**: This is the core function that computes the similarity between images and a set of referenced embeddings. The images are initially described through an AI model (via the `model` interface), followed by embedding with the `embed` import. Following the embeddings, similarity scores are computed and aggregated to yield a final reward representation.

**As an overview:** The LLaVA server describes the images from different views and we obtain a textural representation for the respective asset. Alternatively, we can also use a vision model embeddings, however, since subsequent operations require additional compute time, we limit the number of representations. Since we now obtained textual descriptions, we utilize CLIP to create embeddings that are then used to produce a concistency score between the text and image, based on it's alignment. Lastly, we encode the text using different state-of-the-art embedding model `all-mpnet-base-v2` for comparing against pre-computed assets. We use two different embedding models to avoid exploiting embedding biases of CLIP alone, and to provide a more robust evaluation. The similarity scores of the pre-computed assets are cached in memory for quick retrieval during subsequent similarity comparisons. The similarity scores are then aggregated to produce a final evaluation metric for the 3D asset.

The `RewardModel` offers a set of metrics (e.g., Jaccard distance, cosine similarity, etc.) and aggregation methods to compute the final scores them.

### Aggregation Strategies

The RewardModel supports three aggregation strategies for the similarity scores – 'sum', 'mean', and 'median'. These options ensure that users can fine-tune the evaluation process based on the specific needs of the assessment task, whether it prioritizes collective similarity, average representation, or robustness against outliers in the reference set.

## LLaVA Server: The AI-Powered Descriptive Engine

The [LLaVA server](https://github.com/ggerganov/llama.cpp) functions as the backbone for processing natural language and visual data. Integrating the LLaVA server into the RewardModel workflow offers an AI-powered assistant that can describe images that the RewardModel utilizes for similarity evaluation.

To integrate the LLaVA service into the Bittensor network, one must prepare the server environment:

- Install the necessary build tools and libraries, such as Xcode for MacOS.
- Clone and build the LLaVA server from its [repository](https://github.com/ggerganov/llama.cpp), making sure to initialize and update the required git submodules.
- Download appropriate models such as `ggml-model-*.gguf` and `mmproj-model-f16.gguf`, provided on [Hugging Face](https://huggingface.co/mys/ggml_llava-v1.5-13b/tree/main), which are necessary for the server's operation.

The LLaVA server is launched using a command-line interface, where users specify the model files and configure network parameters such as host and port. The server's API is neatly exposed over HTTP, accepting image data and prompts to return descriptions and conversational responses.
See the `README.md` file for more details on the LLaVA server.

### Current Progress on the LLaVA Integration and RewardModel

Once the LLaVA server is operational, the API is exposed as follows:

1. An image representing a 3D asset is forwarded to the LLaVA server along with the user prompt (e.g., "Describe this 3D model").
2. LLaVA processes the image and returns a natural language description.
3. RewardModel then uses this description to create an embedding and compare it against cached reference embeddings.
4. The similarity scores are aggregated to form an evaluation metric for the 3D asset.

This synergy allows for the automated, intelligent analysis of 3D assets, consolidating the distinct capabilities of LLaVA's image understanding and RewardModel's symbolic AI processing.

## Applications and Future Work

The combination of RewardModel and LLaVA paves the way for automated evaluation of 3D assets. Notable among them is the capacity to sift through vast libraries of 3D assets, identifying those that match a particular style or quality criterion. It also opens doors for real-time feedback during the creation of 3D content, allowing designers to align their work more closely with established benchmarks, and the integration with the Bittensor network allows for the creation of a decentralized marketplace for 3D assets.

## Conclusion

Exploring the integration of the LLaVA-based RewardModel in the Bittensor network creates an interesting avenue for evaluation and assessment of 3D assets. By also including a symbolic methodology we also get insights on an explainable view opposed to purely embedding-based representations and offer human readable insights as a side product. As we continue to refine and enhance asset evaluation, we can expect more refined and robust 3D asset creation models, and a more streamlined process for their distribution.
