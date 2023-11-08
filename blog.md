# Enabling Visual Search for Ecommerce with CLIP and Amazon SageMaker
## How to build visual search for your online store using multi-modal models and vector similarity search

### Introduction 
- Explain the value of visual search for ecommerce - allowing customers to find products using image queries rather than just text search.
- Include data points on influence of visual search
- Overview of the architecture using SageMaker, Faiss, API Gateway, etc.

### On Embeddings in Search with Vector Stores
- V

### Setting Up CLIP Inference on SageMaker
- Briefly introduce CLIP and how it can enable visual search by creating joint embeddings for images and text.
- Step-by-step walkthrough of deploying CLIP inference endpoint on SageMaker using the Inference Toolkit.
- Explain the purpose of the inference endpoint and how it will be used.

### Building Product Image and Text Vector Stores with Faiss
- Using in memory, can use opensearch, pgvector postgres any other
- Explain how Faiss library allows efficient similarity search on vectors.
- Walk through process of indexing product images and text in Faiss using CLIP endpoint.

### Developing the Search API Backend
- Architecture overview - API Gateway, Lambda functions, DynamoDB.
- Explain role of each component.
- Provide code snippets for key functions.

### Creating the Search Application Frontend
- Use Streamlit to build simple interface for visual search.
- Overview of how frontend connects to backend API.
- Include screenshots of the final application.

### Conclusion
- Recap of end-to-end architecture and how the different services work together.
- Discuss potential enhancements like adding more products or improving relevance.
- suitable for single items eg. furniture, appliances
- recognizing attributes, good for multiple items, fashion use cases with models and multiple items
- Invite readers to get started building their own visual search for ecommerce.