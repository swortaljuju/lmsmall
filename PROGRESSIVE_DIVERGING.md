# Idea
- progressively increase number of layers
    - goal: improve perf on longer reasoning chain while overcome gradient deteriorate problem
        1.  train with less layers like 3 layers
        2. for each layer, duplicate to 2 layers with same weights , now we have 6 layers
        3. train with LORA
        4. iteratively do this

# Training Dataset
https://huggingface.co/datasets/ajibawa-2023/Maths-College
https://huggingface.co/datasets/open-web-math/open-web-math

# Related Works
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

