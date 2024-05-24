## Introduction

A toy project of JerryYin777

Reproduction Code of Paper "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention (MIT CSAIL)", [https://arxiv.org/pdf/2405.12981](https://arxiv.org/pdf/2405.12981)

According to the paper, we got we can combine CLA with MHA, MQA, and GQA. The paper got the best performance in CLA + MQA. The advantage of this architecture is some layers compute their own fresh K and V activations, while other layers reuse the K and V activations of earlier layers, thus, we can reduce the computation.

![image](https://github.com/JerryYin777/Cross-Layer-Attention/assets/88324880/5cc08c72-98ce-44f0-8525-27ca6dca008e)

- [x] Cross-Layer Attention
- [x] Cross-Layer Attention + MQA
- [ ] Cross-Layer Attention + GQA
- [ ] Cross-Layer Attention + MLA by DeepSeek(?)

The goal is to validate if I can implement CLA + MLA and achieve a SOTA reduction of KV Cache in a 1B to 3B model.
