# Awesome LLM Resources - A Curated Learning Path & Selection of Free Resources for Large Language Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated learning roadmap and list of selected free resources for developers, researchers, and enthusiasts of Large Language Models (LLMs). This list aims to provide a clear, structured guide to help you systematically master the core concepts, cutting-edge technologies, and practical tools of LLMs, starting from scratch.

---

## Table of Contents

- [🧠 Core Concepts](#-core-concepts)
- [🎓 Structured Courses](#-structured-courses)
- [🏢 Corporate Learning Hubs](#-corporate-learning-hubs)
- [🛠️ Emerging AI Tools & Platforms](#️-emerging-ai-tools--platforms)
- [💻 General Tools & Frameworks](#-general-tools--frameworks)
- [🎥 Video Tutorials & Lectures](#-video-tutorials--lectures)
- [📖 Open Source Models & Datasets](#-open-source-models--datasets)
- [🚀 Keep Learning](#-keep-learning)

---

## 🧠 Core Concepts

Before diving into code, it's crucial to understand the core ideas that drive LLMs. These are the cornerstones of the field.

### Must-Read Papers

* [**Attention Is All You Need** (2017)](https://arxiv.org/abs/1706.03762) - **The absolute must-read paper**. It introduced the Transformer architecture, which is the foundation for all modern Large Language Models.
* [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018)](https://arxiv.org/abs/1810.04805) - Introduced the BERT model and is key to understanding the application of the Transformer Encoder.
* [**Language Models are Few-Shot Learners (GPT-3)** (2020)](https://arxiv.org/abs/2005.14165) - Introduced GPT-3 and demonstrated the surprising "few-shot" learning capabilities of massive models, opening a new chapter for general-purpose AI.

### In-depth Guides & Blogs

* [**The Illustrated Transformer** by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) - **A highly recommended starting point**. It uses exceptionally clear diagrams to break down the internal workings of the Transformer, step-by-step.
* [**LLM Powered Autonomous Agents** by Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/) - A deep-dive blog post from the former Head of Applied AI at OpenAI that systematically explains how to build autonomous agents using LLMs.
* [**Building LLM Applications: Application Development and Architecture Design** by Phodal](https://aigc.phodal.com/en/prelude.html) - An excellent open-source book that systematically explains the architecture and development practices of LLM applications, suitable for developers with some experience.
* [**Andrej Karpathy's Blog**](https://karpathy.github.io/) - The blog of AI luminary Andrej Karpathy, containing profound insights into neural networks and the field of AI.

## 🎓 Structured Courses

Build a solid knowledge base with systematic, university-level courses.

* [**Stanford CS224n: NLP with Deep Learning**](https://web.stanford.edu/class/cs224n/) - **Stanford University's flagship NLP course**. It comprehensively covers everything from traditional NLP to the latest Transformers and LLMs.
* [**LLMs from Scratch by Sebastian Raschka**](https://github.com/rasbt/LLMs-from-scratch/) - A course/book on building a large language model from scratch. It offers in-depth theoretical explanations and detailed code implementations, making it an excellent resource for understanding the underlying principles.
* [**LLM Course by Maxime Labonne**](https://github.com/mlabonne/llm-course) - A GitHub-based, hands-on LLM course that provides a clear roadmap and numerous Colab notebooks.

## 🏢 Corporate Learning Hubs

Get authoritative learning resources and best practices directly from the companies building these technologies.

### DeepLearning.AI Short Courses

A collection of free, concise courses from Andrew Ng's team and industry experts (OpenAI, LangChain, etc.), focused on quickly mastering a specific skill.

* [**Generative AI for Everyone**](https://www.deeplearning.ai/courses/generative-ai-for-everyone/) - A non-technical introduction to Generative AI, its potential, and its value proposition.
* [**ChatGPT Prompt Engineering for Developers**](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) - Learn prompt engineering best practices for application development.
* [**LangChain for LLM Application Development**](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) - Learn the fundamentals of the LangChain framework to build powerful applications.
* [**LangChain: Chat with Your Data**](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) - Focused on building Retrieval-Augmented Generation (RAG) applications with LangChain.
* [**Building Systems with the ChatGPT API**](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) - Learn how to chain multiple prompts together to build complex systems.
* [**Finetuning Large Language Models**](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) - Learn when to apply finetuning and how to do it effectively.
* [**How Diffusion Models Work**](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) - A deep dive into the principles behind text-to-image models like Stable Diffusion.
* [**Vector Databases: from Embeddings to Applications**](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) - Learn the core concepts of vector databases and their application in semantic search and RAG.

### Microsoft AI & LLM Resources

Microsoft provides a wealth of structured courses, official documentation, and practical code samples through its Learn platform and GitHub organizations.

* **Official Learning Paths & Documentation**:
    * [**Microsoft Learn AI Hub**](https://learn.microsoft.com/en-us/ai/) - The central hub for all of Microsoft's AI learning resources, including documentation, tutorials, and certification paths for Azure AI and OpenAI services.
    * [**Azure AI Fundamentals**](https://learn.microsoft.com/en-us/training/paths/get-started-with-artificial-intelligence-on-azure/) - The official learning path for the AI-900 certification, covering core AI and machine learning concepts on Azure.

* **GitHub Repositories & Courses**:
    * [**generative-ai-for-beginners**](https://github.com/microsoft/generative-ai-for-beginners) - A comprehensive, 18-lesson course on Generative AI, created by Microsoft developers.
    * [**azure-search-openai-demo**](https://github.com/Azure-Samples/azure-search-openai-demo) - A premier reference implementation for the RAG pattern using Azure AI Search and Azure OpenAI.
    * [**Semantic Kernel**](https://github.com/microsoft/semantic-kernel) - Microsoft's open-source LLM orchestration SDK, an alternative to LangChain, for building agents and planners.
    * [**Microsoft/AI**](https://github.com/microsoft/AI) - A central index repository that categorizes and links to a large number of Microsoft's open-source AI samples and best practices.

### Other Corporate Hubs

* [**Google - Generative AI Learning Path**](https://www.cloudskillsboost.google/paths/118) - **Official Google Cloud learning path**. It offers a series of courses on Generative AI fundamentals, large language models, and the Google Cloud AI platform.
* [**AWS - Generative AI Learning Plan for Developers**](https://explore.skillbuilder.aws/learn/public/learning_plan/view/1743/generative-ai-learning-plan-for-developers) - **AWS developer learning plan**. It includes 10 courses from beginner to advanced, designed to help developers learn to build and deploy generative AI applications on AWS.
* [**OpenAI Cookbook**](https://github.com/openai/openai-cookbook) - **Official OpenAI practice guide**. It provides numerous runnable code examples that demonstrate best practices for completing common tasks with the OpenAI API.
* [**Anthropic Cookbook**](https://github.com/anthropics/anthropic-cookbook/) - **Official hands-on guide**. Contains extensive code examples, best practices, and tutorials for building with Claude and Anthropic APIs.
* [**Meta Llama Cookbook**](https://github.com/meta-llama/llama-cookbook) - **Official Meta Llama hands-on guide**. This repo contains various code examples for inference, fine-tuning, and building RAG applications with Llama models.

## 🛠️ Emerging AI Tools & Platforms

These companies and tools are defining the development paradigm for modern AI applications. Understanding them is key to building advanced solutions.

### Foundation Model Providers

* [**Anthropic (Claude)**](https://docs.anthropic.com/en/docs/get-started) - **Official Docs**. The authoritative starting point for learning the Claude model family, its API, safety features, and prompt engineering. Their [GitHub Cookbook](https://github.com/anthropics/anthropic-cookbook) provides extensive hands-on code.
* [**Mistral AI**](https://docs.mistral.ai/) - **Official Docs**. Known for its high-performance open-source and commercial models, excelling in efficiency and performance. Their [GitHub](https://github.com/mistralai) contains model implementations and usage examples.
* [**Cohere**](https://docs.cohere.com/) - **Official Docs**. An AI platform focused on enterprise applications. Their [Cohere University](https://cohere.com/university) and [GitHub](https://github.com/cohere-ai) offer rich tutorials, especially for RAG and semantic search.

### Application Development Frameworks

* [**LangChain**](https://python.langchain.com/) - **Official Docs**. The most popular framework for LLM application development, providing modular components and extensive integrations. Its [GitHub](https://github.com/langchain-ai/langchain) includes many cookbooks and templates.
* [**LlamaIndex**](https://docs.llamaindex.ai/en/stable/) - **Official Docs**. A data framework focused on RAG, offering powerful capabilities for data ingestion, indexing, and querying. Its [GitHub](https://github.com/run-llama/llama_index) contains a wealth of examples.

### Critical Infrastructure: Vector Databases

* [**Pinecone**](https://docs.pinecone.io/) - **Official Docs**. A leading managed vector database that provides core support for large-scale, low-latency semantic search and RAG applications. Their [GitHub](https://github.com/pinecone-io) provides clients and examples.
* [**Weaviate**](https://weaviate.io/developers/weaviate) - **Official Docs**. A powerful open-source vector database that supports hybrid search and has an active community. Its [GitHub](https://github.com/weaviate/weaviate) provides the core code and clients.

### Production & Evaluation Tools

* [**Weights & Biases**](https://docs.wandb.ai/guides/prompts) - **W&B Prompts Docs**. A tool for tracking, visualizing, and evaluating LLM applications (especially Prompt Chains), which is a key part of moving from development to production.

## 💻 General Tools & Frameworks

* [**Hugging Face Transformers**](https://huggingface.co/docs/transformers/index) - **The absolute core**. It is a complete ecosystem providing a massive number of pre-trained models, datasets, and tools.
* [**PyTorch**](https://pytorch.org/) / [**TensorFlow**](https://www.tensorflow.org/) - Mainstream deep learning frameworks, fundamental for understanding model fine-tuning and underlying research.

## 🎥 Video Tutorials & Lectures

Visual learning materials can significantly accelerate understanding.

### Coding & Implementation

* [**Andrej Karpathy - "Let's build GPT" Series**](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - **A must-watch series**.
    * [**Let's build GPT**](https://www.youtube.com/watch?v=mMa2PmYJlCo9) - The ultimate hands-on tutorial for understanding the internal mechanics of a Transformer by building a mini-GPT from scratch.
    * [**Let's build the GPT Tokenizer**](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ0) - The sister video to the series, building a BPE tokenizer from scratch and completing the most fundamental piece of the LLM puzzle.
* [**Sebastian Raschka - Build an LLM From Scratch**](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ1) - The official video tutorial from the author of "LLMs from Scratch," explaining the complete process of building a small LLM in a solid, systematic way.

### Concepts & Theory

* [**Andrej Karpathy - The State of GPT**](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ2) - Karpathy personally explains the current state of GPT development, training techniques, and future trends. An excellent lecture for a high-level view.
* [**Jay Alammar's YouTube Channel**](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ3) - The official channel of the author of "The Illustrated Transformer," turning his famous illustrated blog posts into videos.
* [**3Blue1Brown - Attention in Transformers, visually explained**](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ4) - A visual explanation of the attention mechanism. If you're confused by matrix operations and multi-head attention, this video will bring clarity.
* [**Yannic Kilcher's YouTube Channel**](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ5) - Known for his in-depth explanations of the latest and most important AI papers. He walks you through papers line-by-line to understand their motivation and innovation.

### Conferences & Lectures

* [**NeurIPS 2023 Tutorial: Application Development using LLMs**](https://neurips.cc/virtual/2023/tutorial/70068) - A tutorial from a top-tier AI conference by Andrew Ng's team, systematically introducing the full workflow of LLM application development.

## 📖 Open Source Models & Datasets

The open-source community is the core driving force behind the democratization of LLM technology.

### Models

* [**Hugging Face Open LLM Leaderboard**](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - A dynamically updated leaderboard for open-source LLMs. The best starting point for finding and evaluating the most powerful open models available.
* [**Meta Llama 3**](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) - The flagship open-source model family from Meta, a cornerstone of the current open-source ecosystem.
* [**Mixtral of Experts**](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) - Developed by Mistral AI, this model uses an innovative "Mixture of Experts" (MoE) architecture, representing a major step in model efficiency.

### Datasets

* [**The Pile**](https://pile.eleuther.ai/) - A large-scale, diverse, open-source text dataset that has been used to train many powerful open LLMs.
* [**RedPajama-Data-v2**](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-v2) - An open-source reproduction of the training data used for the Llama family of models.
* [**Awesome Datasets for LLM**](https://github.com/gkamradt/awesome-datasets-for-llm) - A curated list of high-quality datasets specifically for Instruction Tuning and Preference Alignment.

## 🚀 Keep Learning

The field of large models is evolving daily. It's vital to maintain a habit of continuous learning.

* **Twitter / X**: Follow top researchers in the field, such as `Yann LeCun`, `Andrej Karpathy`, `Jim Fan`, and `Lilian Weng`.
* **Papers with Code**: [https://paperswithcode.com/](https://paperswithcode.com/) - Track the latest papers and their open-source implementations.
* **AI Newsletters**: Subscribe to newsletters like `The Batch` (from DeepLearning.AI) and `Import AI` to get industry updates and curated paper selections.

---

> If you find any valuable resources or wish to contribute, please feel free to submit a Pull Request!
