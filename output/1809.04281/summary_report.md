# Music Transformer: Generating Music with Long-Term Structure

## Authors
Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, Douglas Eck — Google Brain

## Abstract
The Music Transformer employs self-attention to model music, aiming to generate piece with long-term musical structure. Traditional approaches using pairwise distance for time representation in Transformers are insufficient for long sequences, given their quadratic memory requirements. This paper introduces an efficient algorithm to reduce this memory need to a linear scale, enabling the generation of minute-long music pieces, continuation of motifs, and accompaniment generation.

## Introduction
Musical compositions have inherent self-referential structures that span various scales (motifs, phrases, sections). The paper argues that self-attention is ideal for modeling these due to its capacity to access all prior output points, unlike recurrent models which must actively store these in a fixed memory. Absolute positions have been typically used in Transformers; however, this research asserts the importance of representing relative timing particularly in music.

## Contributions

### Domain Contributions
- First to apply Transformers for generating music with coherent long-term structure surpassing previous efforts utilizing LSTMs.
- Demonstrated model’s capability to generate coherent musical pieces over 60 seconds (~2000 tokens).
- Introduced a mechanism for relative self-attention that allows models to generalize beyond training lengths, enhancing music continuation and accompaniment generation.

### Algorithmic Contributions
- Introduced an efficiency improvement in relative self-attention that reduces the space complexity from O(L²D) to O(LD), allowing feasible training on extensive musical compositions.

## Model Approach

### Data Representation
- Utilizes symbolic music sequences formatted to fit the model needs.
- Implementations for different datasets: JSB Chorales serialized by voice and time, and Piano-e-Competition for expressive dynamic timing.

### Music Transformer Architecture
- Baseline model: Transformer with absolute sinusoidal positions.
- Innovation: Relative positional self-attention which captures relational data (timing, pitch), reducing memory footprint while maintaining complexity.

### Efficient Relative Attention
- Developed an algorithmic 'skewing' procedure that aligns relative embeddings to absolute positions, retaining essential relational indices while conserving memory.

### Local Attention for Long Sequences
- For extremely lengthy sequences, employed chunk-based local attention to minimize computational demand without declining the model's generative performance.

## Experiments
### Datasets
- Utilized JSB Chorales for structured harmonic music evaluation and Piano-e-Competition for complex expressive performances.

### Evaluation
- Transformers with the new relative attention mechanism outperformed traditional models in perplexity and sample quality on both datasets.
- Demonstrated qualitative advantages in handling motif elaboration and phrase consistency over longer sequences.

### Human Evaluation
- Conducted listening tests to statistically validate improved sample quality of models with relative self-attention compared to baselines and LSTMs.

## Conclusion
Relative attention significantly enhances the generation of music with coherent structure, offering potential applications as a creative tool. The research underscores the advantage of learning relational rather than absolute information, bolstering model generalization to unforeseen sequence lengths.

## Links
- [Sample Music Transformer Outputs](https://storage.googleapis.com/music-transformer/index.html)

---

The insightful approach of this research facilitates the proliferation of AI applications in musical composition, emphasizing the role algorithms can play in artistic domains. The advancements in relative attention mechanisms present future pathways for exploring longer sequence dependencies in neural networks, beyond the musical domain.