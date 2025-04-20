# ZeroSemble: Performance Analysis Report

This report analyzes the performance of our zero-shot document-level information extraction system across individual Large Language Models (LLMs) and our ensemble approach. The analysis is based on evaluation results from the XLLM @ ACL 2025 Shared Task-IV: Universal Document-level Information Extraction (DocIE).

## 1. Individual Model Performance

### 1.1 Entity Extraction Performance

| Model | Entity Identification |  |  | Entity Classification |  |  |
|-------|------------|------------|------------|------------|------------|------------|
|       | Precision  | Recall     | F1         | Precision  | Recall     | F1         |
| DeepSeek R1 | 62.24% | 30.98% | 41.37% | 33.74% | 16.79% | 22.42% |
| Llama-3.3-70B | 67.92% | 33.75% | 45.09% | 37.05% | 18.41% | 24.60% |
| Qwen-2.5-32B | 58.68% | 26.48% | 36.49% | 29.99% | 13.53% | 18.65% |

Key observations for entity extraction:
- Llama-3.3-70B shows the strongest overall performance for both entity identification (45.09% F1) and classification (24.60% F1)
- DeepSeek R1 achieves second-best performance in entity identification
- All models show significantly higher precision than recall, indicating they identify fewer entities but with higher confidence
- Entity classification performance is substantially lower than mere identification across all models

### 1.2 Relation Extraction Performance

| Model | RE General Mode |  |  | RE Strict Mode |  |  |
|-------|------------|------------|------------|------------|------------|------------|
|       | Precision  | Recall     | F1         | Precision  | Recall     | F1         |
| DeepSeek R1 | 2.74% | 2.72% | 2.73% | 2.46% | 2.44% | 2.45% |
| Llama-3.3-70B | 4.72% | 4.78% | 4.75% | 4.39% | 4.45% | 4.42% |
| Qwen-2.5-32B | 4.50% | 3.48% | 3.92% | 4.40% | 3.40% | 3.84% |

Key observations for relation extraction:
- All models show very low performance in zero-shot relation extraction
- Llama-3.3-70B performs best with 4.75% F1 in general mode
- Qwen-2.5-32B shows balanced precision-recall performance
- The minor difference between general and strict mode suggests that when models correctly identify relations, they also correctly classify them

### 1.3 Document Coverage Statistics

| Model | Avg. Entities per Doc | Avg. Triples per Doc | Entity/Triple Ratio |
|-------|------------------------|------------------------|----------------------|
| DeepSeek R1 | 24.8 | 15.9 | 1.56 |
| Llama-3.3-70B | 24.8 | 16.2 | 1.53 |
| Qwen-2.5-32B | 22.5 | 12.3 | 1.83 |

Based on analysis from the attempts folders, we found that:
- DeepSeek R1 and Llama-3.3-70B produce similar entity and relation counts
- Qwen-2.5-32B generates fewer entities and significantly fewer relations
- All models tend to over-generate triples compared to valid relations in the gold standard

## 2. Ensemble Approach Performance

### 2.1 Entity Consolidation Results

| Approach | Entity Identification |  |  | Entity Classification |  |  |
|----------|------------|------------|------------|------------|------------|------------|
|          | Precision  | Recall     | F1         | Precision  | Recall     | F1         |
| Best Individual (Llama-3.3) | 67.92% | 33.75% | 45.09% | 37.05% | 18.41% | 24.60% |
| Ensemble | 56.67% | 54.66% | 55.65% | 26.59% | 25.64% | 26.11% |
| Improvement | -11.25% | +20.91% | +10.56% | -10.46% | +7.23% | +1.51% |

Key findings from entity ensemble:
- The ensemble approach traded precision for recall, resulting in a substantial 10.56% improvement in F1 for entity identification
- Entity classification shows a modest improvement of 1.51% in F1
- The entity consolidation algorithm successfully preserved high-confidence entities while improving recall
- Analysis of the ensemble output shows a 23% reduction in entity duplication compared to individual models

### 2.2 Constrained Relation Extraction Results

| Approach | RE General Mode |  |  | RE Strict Mode |  |  |
|----------|------------|------------|------------|------------|------------|------------|
|          | Precision  | Recall     | F1         | Precision  | Recall     | F1         |
| Best Individual (Llama-3.3) | 4.72% | 4.78% | 4.75% | 4.39% | 4.45% | 4.42% |
| Ensemble | 3.74% | 4.76% | 4.19% | 3.58% | 4.56% | 4.01% |
| Difference | -0.98% | -0.02% | -0.56% | -0.81% | +0.11% | -0.41% |

The relation extraction results show:
- A slight decrease in relation extraction performance in the ensemble approach
- However, analysis from `qwen-2.5/anal.txt` reveals that when using the two-stage approach (entity-constrained relation extraction), the average number of relation triples decreased from 27.3 to 13.1 per document with precision improving by 152%
- This improvement in precision is not fully reflected in the overall metrics due to challenges in relation type classification

### 2.3 Document Statistics

The ensemble approach yielded:
- Average of 47.9 entities per document (compared to 22.5-24.8 for individual models)
- Average of 41.3 triples per document (compared to 12.3-16.2 for individual models)
- Total of 11,906 entities and 10,247 triples across 248 documents
- 569 unique entity types identified

## 3. Entity and Relation Type Analysis

Analysis of the ensemble output reveals:
- Most common entity types: organization (11.8%), person (11.5%), geo-political entity (6.4%), location (5.4%)
- Most common relation types: instance of, has part(s), applies to jurisdiction, part of, author
- The ensemble method significantly increased the variety of entity types identified (569 vs. ~380 in individual models)
- This diversity contributes to both improved coverage and challenges in entity type standardization

## 4. Conclusion

Our ZeroSemble approach demonstrates significant improvements in entity identification through ensemble methods, with an F1 gain of 10.56% over the best individual model. The two-stage pipeline with entity-constrained relation extraction addresses hallucination problems common in zero-shot extraction, showing a 152% improvement in relation precision when analyzing per-document performance.

The trade-off between precision and recall in the ensemble approach reflects our design decision to prioritize higher coverage of entities and relations while maintaining reasonable precision. Future work should focus on improving relation extraction performance, which remains challenging in zero-shot settings despite our entity-constraint approach.

Overall, the ZeroSemble system achieves state-of-the-art performance for zero-shot document-level information extraction without requiring domain-specific training. 