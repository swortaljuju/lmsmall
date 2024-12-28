# Introduction
This repository contains variation of language models based on my recent learning of large language model ([notes](https://swortal.blogspot.com/2024/07/ai-reading-notes-deep-learning-and.html)) plus my own ideas. 



It contains following models
-  a baseline gpt2 model for comparison. The baseline model is implemented based on this great tutorial: [build-nanogpt](https://github.com/karpathy/build-nanogpt).  
- A transformer based model with some MLP network being replaced by convolution network so that multiple tokens' embedding are merged into one embedding and attention can be more easily applied on a longer sequence. See [conv_transformer](./CONV_TRANSFORMER.md)
- A transformer based model with multiple layers sharing same weights at the beginning and diverging as training going on. See [progressive_diverging](PROGRESSIVE_DIVERGING.md)

# Disclaimer
My main goal here is to practice building machine learning model with my novel ideas in short term. It is NOT for serious research purpose. Nor are the models well trained since modern language model training requires large amount time and computation resource which isn't affordable for a personal side project.   


# Reference
- https://github.com/karpathy/build-nanogpt 
- https://github.com/karpathy/nanoGPT 