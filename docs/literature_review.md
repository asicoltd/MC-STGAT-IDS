# Benchmarking Encoders and Pretext Tasks for GNN-NIDS

## Introduction

Graph neural networks (GNN) have become really useful for detecting network intrusions nowadays. This makes sense because network traffic is like a graph with hosts as nodes and flows as edges. Three recent papers: PPT-GNN, Anomal-E, and GraphIDS, have taken this idea. Each paper tries to figure out how to learn from network traffic with a small amount of labeled data or without needing any labeled data. Here are the comparisons of two parts of these papers: the pretext tasks they use to teach themselves and the encoder architectures they use to understand the data.

## Pretext Tasks: Three Different Games

### 1. PPT-GNN: Link Prediction
PPT-GNN uses link prediction. The model looks at a graph. It tries to decide if each edge is real or fake. The fake edges are added in a way that makes the model learn about the relationships between nodes over time and space. Once the model is trained, it's fine-tuned on the task of detecting intrusions with a little bit of labeled data. This approach is very straightforward: teach the model what normal connections look like, and use it to spot attacks.

```mermaid
graph LR
    A[Raw Network Graph] --> B[Add Fake Edges<br>Link Prediction Pretext]
    B --> C[Pre-train GNN<br>Learn Normal Connections<br>over Time & Space]
    C --> D[Fine-tune<br>Small Labeled Attack Data]
    D --> E[Detect Intrusions<br>Classify Edges as<br>Attack/Benign]
```

### 2. Anomal-E: Contrastive Learning
Anomal-E does things differently. Its pretext task is contrastive learning, which is a way of teaching the model to distinguish between real and fake data using Deep Graph Infomax (DGI). The model looks at the graph. It tries to figure out if the edges are real or not by comparing them to a summary of the whole graph. This approach requires negative examples, which are created by shuffling the edge features.

```mermaid
graph LR
    A[Raw Graph] --> C[Encoder<br>E-GraphSAGE]
    B[Corrupted Graph<br>Shuffled Features] --> C
    C --> D[Contrastive Learning<br>Real vs. Fake]
    D --> E[Pre-trained Encoder<br>Frozen]
    E --> F[Anomaly Detector<br>IF/CBLOF/PCA/HBOS]
    F --> G[Intrusion Detection]
```

### 3. GraphIDS: Masked Reconstruction
GraphIDS takes a different approach. Its pretext task is masked reconstruction. Instead of trying to predict if an edge is real or fake, the model tries to rebuild the graph from a partial view using a Transformer-based Masked Autoencoder (MAE). Some of the connections between nodes are hidden. The model can still see all the node features. The model has to figure out what the original graph looked like. This is similar to how some language models work, but applied to graphs instead of text. One key difference between GraphIDS and the other two papers is that it needs zero labeled data.

```mermaid
graph LR
    A[Raw Graph<br>Benign Traffic] --> B[E-GraphSAGE<br>Encoder]
    B --> C[Transformer MAE<br>Masked Attention]
    C --> D[Reconstructed<br>Embeddings]
    D --> E[Reconstruction Error<br>MSE Loss]
    E --> F[High Error =<br>Intrusion Detected]
```

## Encoder Architectures

### 1. PPT-GNN: Custom Heterogeneous GNN
The PPT-GNN model employs a specialized graph network, which is structured in such a way as to be able to deal with two classes of nodes: flow nodes and IP nodes. While flow nodes carry properties describing the traffic, the IP nodes carry properties describing timing information, among others. All layers of the model analyze the graph in both spatial and temporal aspects.

```mermaid
graph LR
    A[Input Graph<br>Flow Nodes + IP Nodes] --> B[Custom Heterogeneous GNN<br>Multi-Layer<br>Spatial + Temporal Processing]
    B --> C[Flow Embeddings]
    C --> D[Classifier<br>Intrusion Detection]
```

### 2. Anomal-E: Edge-Focused E-GraphSAGE
Anomal-E uses a modified version of GraphSAGE that can handle edge features. The node features are all set to the value one, so the model only learns from the edges. This approach is simpler than PPT-GNN. It only uses one layer. The authors chose this approach because it works better with contrastive learning. The model is trained, then frozen and used as a feature extractor for anomaly detection algorithms.

```mermaid
graph LR
    A[Input Graph<br>Node Features = Ones<br>Edge Features = Flows] --> B[E-GraphSAGE<br>1-Layer<br>Edge-Focused Only]
    B --> C[Edge Embeddings]
    C --> D[Pre-trained & Frozen]
    D --> E[Feature Extractor]
    E --> F[Anomaly Detector<br>IF/CBLOF/PCA/HBOS]
```

### 3. GraphIDS: Regularized GNN with Transformer
GraphIDS also uses a modified version of GraphSAGE with some key differences. The model is trained with a Transformer in a single step. The gradient flows through both components, which makes the model really good at understanding the relationships between nodes. The graph neural network is also heavily regularized to prevent overfitting. This approach is different from the two papers because the graph neural network is used as a preprocessor for the Transformer rather than the main model.

```mermaid
graph LR
    A[Input Graph<br>Benign Traffic] --> B[E-GraphSAGE<br>1-Layer<br>Heavily Regularized]
    B --> C[Transformer MAE<br>Masked Autoencoder]
    B -.->|Gradients Flow Back| C
    C --> D[Reconstruction Error<br>MSE Loss]
    D --> E[High Error =<br>Intrusion Detected]
```

## Observations and Reflections

What's really interesting about these three papers is how they represent different approaches to the same problem. PPT-GNN tries to build a graph network that can handle time and space natively. Anomal-E takes a simplified approach and uses contrastive learning to get good results. GraphIDS goes further and uses a Transformer to handle the complex patterns in the data.

Each approach has its trade-offs. It's not clear that any one of them is universally better. We can see the field moving towards more integrated, less label-dependent approaches, which is probably a good direction for practical intrusion detection.
