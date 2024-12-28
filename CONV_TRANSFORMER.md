# Idea
embedding of sentence should be based on embedding or each word.
so shouldn’t use adaptive granularity
instead, maybe like encoder and decoder and conv network with higher dimension at the beginning and lower dimension in the middle, do the same thing based on attention mechanism?
early layer: large block size, small embedding; inner layer: small block size, large embedding

# Related Works

**sentence embedding** 

[https://airbyte.com/data-engineering-resources/sentence-word-embeddings#:~:text=Sentence and word embeddings are,process and analyze text accurately](https://airbyte.com/data-engineering-resources/sentence-word-embeddings#:~:text=Sentence%20and%20word%20embeddings%20are,process%20and%20analyze%20text%20accurately). 

You can generate these embeddings using the Universal Sentence Encoder (USE), Smooth Inference Frequency (SIF), InferSent, and BERT.

bert generates sentence embedding 

https://datascience.stackexchange.com/questions/62658/how-to-get-sentence-embedding-using-bert

either pool all output embedding, or use [cls]’s embedding

sentence transformer lib 

https://www.sbert.net/docs/quickstart.html#sentence-transformer 

https://www.sbert.net/ 

universal sentence encoder

https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf 

An Exploration of Hierarchical Attention Transformers
for Efficient Long Document Classification

https://arxiv.org/pdf/2210.05529