# Awesome LLM Resources - A Curated List of Free Learning Materials for Large Language Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of free resources and a learning roadmap for developers, researchers, and enthusiasts interested in Large Language Models (LLMs). This list aims to provide a clear, structured guide to help you systematically master the core concepts, cutting-edge technologies, and practical tools of LLMs, starting from scratch.

---

## Table of Contents

- [🧠 Core Concepts](#-core-concepts)
- [🎓 Structured Courses](#-structured-courses)
- [💻 Tools & Frameworks](#-tools--frameworks)
- [🎥 Video Tutorials & Lectures](#-video-tutorials--lectures)
- [📖 Open Source Models & Datasets](#-open-source-models--datasets)
- [🚀 Keep Learning](#-keep-learning)

---

## 🧠 Core Concepts

Before diving into code, it's crucial to understand the core ideas that drive LLMs. These are the cornerstones of the field.

### Must-Read Papers

* [**Attention Is All You Need** (2017)](https://arxiv.org/abs/1706.03762) - **The absolute must-read paper**. This paper introduced the Transformer architecture, which is the foundation for all modern Large Language Models. Understanding this is understanding half of the LLM landscape.
* [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018)](https://arxiv.org/abs/1810.04805) - Introduced the BERT model, which revolutionized Natural Language Understanding (NLU) tasks through bidirectional pre-training. It's key to understanding the application of Transformer Encoders.
* [**Language Models are Few-Shot Learners (GPT-3)** (2020)](https://arxiv.org/abs/2005.14165) - Introduced GPT-3 and demonstrated the surprising "few-shot" and "zero-shot" learning capabilities of massive models, opening a new chapter for general-purpose AI.

### In-depth Guides & Blogs

* [**The Illustrated Transformer** by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) - **A highly recommended starting point**. The author uses exceptionally clear diagrams to break down the internal workings of the Transformer, step-by-step. It's the best companion for understanding the "Attention Is All You Need" paper.
* [**LLM Powered Autonomous Agents** by Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/) - A deep-dive blog post from the former Head of Applied AI at OpenAI. This article systematically explains how to build autonomous agents using LLMs, making it excellent material for understanding the core ideas behind LangChain and AutoGPT.
* [**Andrej Karpathy's Blog**](https://karpathy.github.io/) - The blog of AI luminary Andrej Karpathy. While not frequently updated, each post contains profound insights into neural networks and the field of AI.

## 🎓 Structured Courses

Build a solid knowledge base with systematic, university-level courses.

* [**Stanford CS224n: NLP with Deep Learning**](https://web.stanford.edu/class/cs224n/) - **Stanford University's flagship NLP course**. It comprehensively covers everything from traditional NLP to the latest Transformers and LLMs. All lecture videos, notes, and assignments are available online for free, making it one of the most classic NLP courses in academia.
* [**DeepLearning.AI - Generative AI for Everyone**](https://www.coursera.org/learn/generative-ai-for-everyone) - An introductory course on Generative AI by Andrew Ng. It explains the working principles and applications of GenAI in non-technical language, making it ideal for product managers, project managers, and anyone interested in the business value of GenAI.
* [**DeepLearning.AI - LangChain for LLM Application Development**](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) - A short course by Andrew Ng in collaboration with LangChain's creator, Harrison Chase. It teaches you how to quickly build LLM-powered applications (like Q&A over documents) through hands-on coding.

## 💻 Tools & Frameworks

The right tools are essential for effective development. These are the core of today's LLM ecosystem.

* [**Hugging Face Transformers**](https://huggingface.co/docs/transformers/index) - **The absolute core of the ecosystem**. It's not just a library but a complete platform offering tens of thousands of pre-trained models, datasets, and evaluation tools. It is the go-to tool for using and fine-tuning models.
* [**LangChain**](https://python.langchain.com/v0.2/getting_started/) - A powerful framework designed to simplify the development of LLM applications. It standardizes complex logic like model calls, data connections (to documents, APIs), memory modules, and agents, allowing developers to build complex apps like assembling LEGO blocks.
* [**LlamaIndex**](https://www.llamaindex.ai/) - A "data-centric" framework. If your application needs to interact with large amounts of external, private data (like PDFs, databases, Notion), LlamaIndex provides powerful data indexing, ingestion, and querying capabilities. It's the premier tool for building advanced RAG (Retrieval-Augmented Generation) applications.
* [**PyTorch**](https://pytorch.org/) / [**TensorFlow**](https://www.tensorflow.org/) - While high-level libraries like Transformers abstract away the details, understanding at least one major deep learning framework is still crucial for model fine-tuning and low-level research.

## 🎥 Video Tutorials & Lectures

Visual learning materials can significantly accelerate understanding.

* [**Andrej Karpathy - "Let's build GPT"**](https://www.youtube.com/watch?v=kCc8FmEb1nY) - **A must-watch video**. Andrej Karpathy builds a miniature GPT model from scratch, line-by-line, using only Python and PyTorch. This is the ultimate hands-on tutorial for deeply understanding the internal mechanics of a Transformer.
* [**3Blue1Brown - Attention in Transformers, visually explained**](https://www.youtube.com/watch?v=mMa20x3aA0g) - A visual explanation of the attention mechanism from the famous educational channel 3Blue1Brown. If you're confused about matrix operations and multi-head attention, this video will provide clarity.
* [**NeurIPS 2023 Tutorial: Application Development using LLMs**](https://neurips.cc/virtual/2023/tutorial/70068) - A tutorial from a top-tier AI conference, presented by Andrew Ng's team. It systematically introduces the full lifecycle of LLM application development, from prototyping to deployment and evaluation, covering key techniques like RAG and Fine-tuning.

## 📖 Open Source Models & Datasets

The open-source community is the core driving force behind the democratization of LLM technology.

### Models

* [**Hugging Face Open LLM Leaderboard**](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - A dynamically updated leaderboard for open-source LLMs. It's the best place to start when looking for and evaluating the most powerful open models available today.
* [**Meta Llama 3**](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) - The flagship open-source model family from Meta. It is powerful, has broad community support, and a rich ecosystem of fine-tuned versions, making it a cornerstone of the current open-source landscape.
* [**Mixtral of Experts**](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) - Developed by Mistral AI, this model uses an innovative "Mixture of Experts" (MoE) architecture. It achieves performance comparable to much larger models while maintaining lower inference costs, representing a major step in model efficiency.

### Datasets

* [**The Pile**](https://pile.eleuther.ai/) - A large-scale, diverse, open-source text dataset that has been used to train many powerful open LLMs. It is composed of 22 high-quality sub-datasets and is an excellent case study for understanding what models are "fed".
* [**RedPajama-Data-v2**](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-v2) - An open-source reproduction of the training data used for the Llama family of models. Studying this dataset provides deep insight into how training data for top-tier models is constructed and cleaned.
* [**Awesome Datasets for LLM**](https://github.com/gkamradt/awesome-datasets-for-llm) - A curated list of high-quality datasets specifically for Instruction Tuning and Preference Alignment (RLHF/DPO).

## 🚀 Keep Learning

The field of large models is evolving daily. It's vital to maintain a habit of continuous learning.

* **Twitter / X**: Follow top researchers in the field, such as `Yann LeCun`, `Andrej Karpathy`, `Jim Fan`, and `Lilian Weng`.
* **Papers with Code**: [https://paperswithcode.com/](https://paperswithcode.com/) - Track the latest papers and their open-source implementations.
* **AI Newsletters**: Subscribe to newsletters like `The Batch` (from DeepLearning.AI) and `Import AI` to get industry updates and curated paper selections.

---
> If you find any valuable resources or wish to contribute, please feel free to submit a Pull Request!
