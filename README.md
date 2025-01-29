# Introduction
This repository contains variation of language models based on my recent learning of large language model ([notes](https://swortal.blogspot.com/2024/07/ai-reading-notes-deep-learning-and.html)) plus my own ideas. 



It contains following models
-  a baseline gpt2 model for comparison. The baseline model is implemented based on this great tutorial: [build-nanogpt](https://github.com/karpathy/build-nanogpt).  
- A transformer based model with some MLP network being replaced by convolution network so that multiple tokens' embedding are merged into one embedding and attention can be more easily applied on a longer sequence. See [conv_transformer](#convolution-attention-model)
- A transformer based model with multiple layers sharing same weights at the beginning and diverging as training going on. See [progressive_diverging](#progressive-diverging-model)

# Disclaimer
My main goal here is to practice building machine learning model with my novel ideas in short term. It is NOT for serious research purpose. Nor are the models well trained since modern language model training requires large amount of time and computation resource which isn't affordable for a personal side project.  And given the models are too large, their hyper parameters won't be tuned and compared using validation dataset. The models are only trained on training dataset once and tested on test dataset.  

# Dataset 
Currently this project mainly focuses on ideas which could improve reasoning capability. Therefore, I use math and reasoning data for training and testing.
- https://huggingface.co/datasets/ajibawa-2023/Maths-College  

- https://huggingface.co/datasets/open-web-math/open-web-math


# Contribution
- 2 variations of transformer based language model with novel ideas
- Refactored [nanogpt](https://github.com/karpathy/build-nanogpt) code into following reusable components  
    - A [base trainer](base_trainer.py) container common model training and testing logic which can also be extended for custom training algorithm in the future
    -  
# Custom Models
## Convolution Attention Model
embedding of sentence should be based on embedding or each word.
so shouldn’t use adaptive granularity
instead, maybe like encoder and decoder and conv network with higher dimension at the beginning and lower dimension in the middle, do the same thing based on attention mechanism?
early layer: large block size, small embedding; inner layer: small block size, large embedding


### Related Works

**sentence embedding** 

[https://airbyte.com/data-engineering-resources/sentence-word-embeddings#:~:text=Sentence and word embeddings are,process and analyze text accurately](https://airbyte.com/data-engineering-resources/sentence-word-embeddings#:~:text=Sentence%20and%20word%20embeddings%20are,process%20and%20analyze%20text%20accurately). 

You can generate these embeddings using the Universal Sentence Encoder (USE), Smooth Inference Frequency (SIF), InferSent, and BERT.

**bert generates sentence embedding** 

https://datascience.stackexchange.com/questions/62658/how-to-get-sentence-embedding-using-bert

either pool all output embedding, or use [cls]’s embedding

**sentence transformer lib** 

https://www.sbert.net/docs/quickstart.html#sentence-transformer 

https://www.sbert.net/ 

**universal sentence encoder**

https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf 

**An Exploration of Hierarchical Attention Transformers
for Efficient Long Document Classification**

https://arxiv.org/pdf/2210.05529

## Progressive Diverging model
- progressively increase number of layers
    - goal: improve perf on longer reasoning chain while overcome gradient deteriorate problem
        1.  train with less layers like 3 layers
        2. for each layer, duplicate to 2 layers with same weights , now we have 6 layers
        3. train with LORA
        4. iteratively do this

### Related Works
**1. Progressive Neural Networks:**

Progressive Neural Networks introduce new layers or columns alongside existing ones to facilitate transfer learning across different tasks. These networks expand by adding new layers with lateral connections to previously learned features, enabling the integration of new information without catastrophic forgetting.

[arXiv](https://arxiv.org/abs/1606.04671?utm_source=chatgpt.com)

**2. AutoGrow: Automatic Layer Growing in Deep Convolutional Networks:**

AutoGrow is a method that automates the process of determining the optimal depth of deep neural networks. Starting from a shallow architecture, it progressively adds new layers if such growth leads to improved accuracy. The process continues until adding more layers no longer yields performance gains.

[arXiv](https://arxiv.org/abs/1906.02909?utm_source=chatgpt.com)


**3. Gradual DropIn of Layers to Train Very Deep Neural Networks:**

This technique involves starting with a shallow network and gradually adding new layers during training. The newly added layers are initially bypassed, and their influence is incrementally increased. This method helps in training very deep networks by mitigating issues related to gradient vanishing or explosion.

[arXiv](https://arxiv.org/abs/1511.06951?utm_source=chatgpt.com)


**4. When To Grow? A Fitting Risk-Aware Policy for Layer Growing in Deep Neural Networks:**

This study investigates the optimal timing for adding new layers to a neural network during training. It reveals that neural growth inherently exhibits a regularization effect, and the timing of growth can significantly influence model performance.

[arXiv](https://arxiv.org/abs/2401.03104?utm_source=chatgpt.com)


**MixtureGrowth: Growing Neural Networks by Recombining Learned Parameters:**

MixtureGrowth explores growing neural networks by recombining learned parameters. It analyzes the sensitivity to growth points and the characteristics of learned features, providing insights into effective strategies for network expansion.

[Open Access CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Pham_MixtureGrowth_Growing_Neural_Networks_by_Recombining_Learned_Parameters_WACV_2024_paper.pdf?utm_source=chatgpt.com)


# Results and Comparison
| Model | # heads | embedding size | micro batch size | context window | Training Batches | Training Time per batch | Final Training Loss | Testing Batches | Inference time | Average test loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| Baseline GPT2 | 6 | 192 | 8 | 256 | 80k | 487ms | 3.54 | 20k | 5.2ms | 3.4604 |
| Convolution Attention | 6 | 48 - 496 | 8 | 1024 | 80k | 250ms | 1.42 | 20k | 12.23ms | 1.3 |


# Reference
- https://github.com/karpathy/build-nanogpt 
- https://github.com/karpathy/nanoGPT 