# Natural Language Processing and Language Models Guide

This guide provides resources and approaches for working with Language Models (LLMs) and Natural Language Processing (NLP), from basic concepts to advanced fine-tuning techniques.

If you have any questions or need help with your project feel free to make a Git issue or contact me!

## 📚 Core Learning Resources

1. **Foundations of NLP**: [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slpdraft/) by Jurafsky & Martin
   - Recommended starting point for understanding core NLP concepts from basics to deep learning

2. **Fine-tuning Guide**: [Comprehensive Fine-tuning Documentation](https://arxiv.org/pdf/2408.13296v1)
   - Essential reading: Introduction (pages 6-13)
   - Covers historical context and current landscape

3. **Practical Learning**: [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)
   - Hands-on tutorials and practical implementations

## 🛠️ Implementation Approaches

### 1. Prompting

#### Zero-shot Prompting
- Uses natural language to directly query LLMs
- Simple but results can be unpredictable
- Best with structured generation frameworks

#### In-context Learning
- Provides examples within the prompt
- More context-aware than zero-shot
- Still relatively naive approach

You can use the `outlines` library to generate structured outputs from LLMs, which is a great way to get better results than just using the LLM to generate a string, but that does not require fine-tuning.

**Example using Structured Generation:**

```python
python
import outlines
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?
Review: This restaurant is just awesome!
"""
generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
```

### 2. Transfer Learning

- Utilize pre-trained embedding models
- Check [Hugging Face Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for top models
- Recommended: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for practical implementation

Using the `sentence-transformers` library, you can easily encode sentences and compare them to find similar sentences.

We give a detailed example for both supervised and unsupervised transfer learning in the `transfer_learning.ipynb`. 

**Example Usage:**

```python
python
sentences = [
"The weather is lovely today.",
"It's so sunny outside!",
"He drove to the stadium.",
]
embeddings = model.encode(sentences)
similarities = model.similarity(embeddings, embeddings)
```

### 3. Fine-tuning

There are two types of fine-tuning: supervised and unsupervised. Supervised fine-tuning requires labeled data, while unsupervised fine-tuning does not. We go through both in the `finetuning_supervised.ipynb` and `finetuning_unsupervised.ipynb` notebooks.

Following our above advice, we recommend using `sentence-transformers` for fine-tuning since it produces meaningful embeddings. We also recommend using the `SetFit` library for fine-tuning since it is fast and easy to use.

If you want to dig in to fine-tuning larger models, we recommend looking at the PEFT library.

Another kind of fine-tuning is RAGs (Retrieval Augmented Generation). We provide some resources on how to implement RAGs using LangChain and Hugging Face below.

#### Recommended Tools:
- [SetFit Library](https://github.com/huggingface/setfit) - Fast, laptop-friendly fine-tuning
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/training) - Comprehensive fine-tuning toolkit
- [PEFT](https://huggingface.co/docs/transformers/en/peft) - Parameter-efficient fine-tuning

#### For RAG Implementation:
- Combine LangChain and Hugging Face
- [Tutorial: RAG Implementation](https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7)

## ⚠️ Important Notes

- Commercial APIs (ChatGPT, Claude) may have reproducibility issues due to model updates
- Consider using open models like LLAMA3 for more stable results
- Many modern models are prompt-focused, which may affect fine-tuning approaches
- Compute requirements vary significantly between models
