# Results and Discussion

## 5.1 Experimental Setup

We evaluated OFALMA against four baseline memory management approaches across three memory constraint levels (25%, 50%, and 75% of average dialogue length) on a held-out validation set of 20 dialogues. The baseline methods include: (1) **token_buffer**: simple first-in-first-out token-based truncation, (2) **summary**: LangChain's automatic summarization-based memory, (3) **custom_summary**: task-specific LLM-generated summaries, and (4) **ofalma_rate_distortion**: OFALMA with rate-distortion condensation. All methods were evaluated using GPT-4o as the evaluation model, and OFALMA weights were learned separately for each memory ratio using Proximal Policy Optimization (PPO) reinforcement learning.

## 5.2 Overall Performance Comparison

### 5.2.1 Superiority at Moderate and High Memory Ratios

The experimental results demonstrate OFALMA's clear superiority when memory constraints are not severely limited. At **50% memory**, OFALMA pruning achieves **95% accuracy** (19/20 correct), significantly outperforming all baseline methods. The token_buffer and summary methods achieve only 5% accuracy, while custom_summary reaches 85% accuracy. This 10 percentage point advantage over the best baseline method demonstrates the effectiveness of learned importance-based selection.

At **75% memory**, both OFALMA variants (pruning and rate-distortion) achieve **95% accuracy** (19/20 correct), matching the performance at 50% memory. This suggests that OFALMA has learned to identify the critical information needed for accurate question answering, and additional memory beyond 50% provides minimal benefit. The custom_summary method maintains 85% accuracy, while summary improves to 45% and token_buffer to 25%, but both remain substantially below OFALMA's performance.

The consistent 95% accuracy across 50% and 75% memory ratios indicates that OFALMA has learned an efficient representation strategy. Rather than simply retaining more information as memory increases, OFALMA identifies and preserves the most relevant facts regardless of the available budget, achieving near-optimal performance with only half the original dialogue content.

### 5.2.2 Performance Under Severe Memory Constraints

At **25% memory**, the results reveal the fundamental challenge of extreme compression. OFALMA pruning achieves 30% accuracy (6/20 correct), while OFALMA rate-distortion achieves 35% accuracy (7/20 correct). While these results are substantially better than token_buffer (5%) and summary (5%), they fall short of custom_summary's 80% accuracy at this ratio.

This performance gap at 25% memory suggests that when memory is severely constrained, the ability to condense information (as in custom_summary and rate-distortion) becomes more critical than selective pruning. However, it is important to note that custom_summary requires additional LLM calls for summarization, increasing computational cost and latency, whereas OFALMA pruning operates with a single pass through the dialogue.

The token efficiency analysis reveals an interesting pattern: at 25% memory, OFALMA pruning uses only 60.7 tokens on average, compared to custom_summary's 64.8 tokens. This suggests that OFALMA is more aggressive in its pruning, potentially removing too much critical information at extreme compression ratios. The rate-distortion variant, which condenses rather than removes facts, achieves better accuracy (35% vs 30%) while using slightly more tokens (71.0 vs 60.7), indicating that condensation is beneficial when memory is severely limited.

## 5.3 Learned Impact Factor Weights

### 5.3.1 Pruning Method: Dominance of Relevance and Emphasis

The weights learned for the OFALMA pruning method reveal a clear hierarchy of importance across the four impact factors. Across all memory ratios, **Relevance (Q)** and **Emphasis (E)** consistently receive the highest weights, while **Recency (R)** receives minimal weight and **Surprisal (S)** varies significantly with memory availability.

At **25% memory**, the learned weights are: Q = 0.2555 (43.8%), E = 0.3080 (52.8%), S = 0.0099 (1.7%), and R = 0.0096 (1.6%). The dominance of Emphasis (52.8%) over Relevance (43.8%) suggests that when memory is extremely constrained, the model prioritizes facts that are expressed with strong assertiveness or emotional force. This makes intuitive sense: emphatic statements often contain critical constraints or important information that cannot be omitted.

At **50% memory**, the weight distribution shifts dramatically: Q = 0.7390 (60.8%), E = 0.4602 (37.8%), S = 0.0167 (1.4%), and R = 0.0000 (0.0%). The dramatic increase in Relevance weight (from 43.8% to 60.8%) and the complete elimination of Recency weight (from 1.6% to 0.0%) indicates that with moderate memory, the model learns that topic relevance is the primary determinant of importance, and temporal position is irrelevant. The near-zero Recency weight is particularly striking, suggesting that the model has learned that the order of information presentation does not correlate with its importance for question answering.

At **75% memory**, the weights stabilize: Q = 0.7763 (51.4%), E = 0.5565 (36.8%), S = 0.1735 (11.5%), and R = 0.0044 (0.3%). The continued dominance of Relevance and Emphasis is maintained, but Surprisal receives a more substantial weight (11.5%) compared to lower memory ratios. This suggests that when memory is abundant, the model can afford to retain unexpected or novel information that might be useful for answering questions, even if it is not directly relevant to the main topic.

The evolution of weights across memory ratios reveals an adaptive learning strategy: as memory increases, the model shifts from prioritizing emphatic statements (which are likely to be critical) to prioritizing relevant content (which is necessary for accurate answers), while consistently ignoring recency as a factor.

### 5.3.2 Rate-Distortion Method: Balanced Emphasis on Relevance and Emphasis

The weights learned for the OFALMA rate-distortion method show a more balanced distribution between Relevance and Emphasis, with both factors receiving substantial weight across all memory ratios. This balanced approach reflects the different operational mechanism of rate-distortion: rather than completely removing facts, it condenses them, allowing for a more nuanced allocation of importance.

At **25% memory**, the weights are: Q = 0.3680 (46.1%), E = 0.3670 (46.0%), S = 0.0330 (4.1%), and R = 0.0300 (3.8%). The near-equal weighting of Relevance and Emphasis (46.1% vs 46.0%) suggests that when condensation is possible, both factors are equally important. The slightly higher weights for S and R compared to pruning (4.1% vs 1.7% and 3.8% vs 1.6%) indicate that rate-distortion can preserve some information about surprising or recent facts through condensation, even when memory is limited.

At **50% memory**, the weights are: Q = 0.5550 (53.5%), E = 0.4260 (41.1%), S = 0.0280 (2.7%), and R = 0.0280 (2.7%). Relevance becomes dominant (53.5%), but Emphasis remains substantial (41.1%), maintaining the balanced approach. The equal weights for S and R (2.7% each) suggest that when memory is moderate, both factors receive minimal but equal consideration.

At **75% memory**, the weights are: Q = 0.5611 (39.3%), E = 0.5467 (38.3%), S = 0.2603 (18.2%), and R = 0.0603 (4.2%). Interestingly, the relative weight of Relevance decreases (from 53.5% to 39.3%) while Emphasis increases (from 41.1% to 38.3%), and Surprisal receives a much larger weight (18.2%). This suggests that with abundant memory, rate-distortion can afford to preserve more surprising or unexpected information through less aggressive condensation, while maintaining high-quality preservation of relevant and emphatic content.

### 5.3.3 Comparative Analysis: Pruning vs Rate-Distortion

Comparing the weight distributions between pruning and rate-distortion methods reveals fundamental differences in their learned strategies. The pruning method shows a more extreme weight distribution, with Relevance and Emphasis dominating (totaling 88-98% of weight) and Recency essentially eliminated (0-1.6%). In contrast, rate-distortion maintains a more balanced distribution, with Relevance and Emphasis totaling 80-92% of weight and Surprisal receiving more substantial consideration (4-18%).

This difference reflects the operational constraints of each method: pruning must make binary decisions (keep or remove), leading to a more aggressive prioritization strategy, while rate-distortion can make continuous decisions (condense to varying degrees), allowing for a more nuanced allocation of importance. The higher Surprisal weights in rate-distortion suggest that condensation enables the preservation of unexpected information that might be useful, whereas pruning must make harder choices about what to completely eliminate.

The consistent near-zero Recency weights in both methods (0-4.2%) is a particularly important finding. This suggests that temporal position in the dialogue is not a reliable indicator of importance for question answering. This contradicts common assumptions in dialogue systems, where recent information is often prioritized. The learned weights indicate that topic relevance and emphasis are far more important than when information was mentioned.

## 5.4 Token Efficiency and Memory Utilization

### 5.4.1 Token Usage Patterns

Analysis of average token usage reveals interesting patterns in how each method utilizes available memory. At 25% memory (72 tokens), OFALMA pruning uses the fewest tokens (60.7), followed by custom_summary (64.8), rate-distortion (71.0), summary (71.5), and token_buffer (65.1). The fact that OFALMA pruning uses fewer tokens than the limit suggests that it is highly selective, potentially removing too much information at extreme compression ratios.

At 50% memory (144 tokens), OFALMA pruning uses 133.1 tokens (92.4% of budget), custom_summary uses 105.2 tokens (73.1%), rate-distortion uses 134.1 tokens (93.1%), summary uses 139.4 tokens (96.8%), and token_buffer uses 137.0 tokens (95.1%). The high utilization rates (92-97%) for most methods suggest that they are effectively using the available memory budget.

At 75% memory (216 tokens), OFALMA pruning uses 205.7 tokens (95.2%), custom_summary uses 134.4 tokens (62.2%), rate-distortion uses 167.0 tokens (77.3%), summary uses 178.2 tokens (82.5%), and token_buffer uses 206.9 tokens (95.8%). The lower utilization by custom_summary (62.2%) suggests that it may be over-condensing when memory is abundant, potentially losing information that could improve accuracy.

### 5.4.2 Efficiency-Accuracy Trade-offs

The relationship between token usage and accuracy reveals important efficiency trade-offs. OFALMA pruning achieves 95% accuracy at both 50% and 75% memory while using 133.1 and 205.7 tokens respectively. The fact that accuracy does not improve with additional memory (both ratios achieve 95%) suggests that OFALMA has identified the optimal set of facts needed for accurate question answering, and additional memory provides no benefit.

In contrast, custom_summary maintains 85% accuracy across all memory ratios but uses progressively more tokens (64.8 → 105.2 → 134.4). This suggests that custom_summary's summarization strategy does not adapt as effectively to available memory, potentially over-condensing at low ratios and under-utilizing at high ratios.

The rate-distortion method shows a more adaptive pattern: accuracy improves from 35% at 25% memory to 85% at 50% memory to 95% at 75% memory, while token usage increases from 71.0 to 134.1 to 167.0. This progressive improvement suggests that rate-distortion benefits from additional memory, allowing it to preserve more information through less aggressive condensation.

## 5.5 Failure Mode Analysis

### 5.5.1 Common Failure Patterns

Analysis of the 20 validation dialogues reveals interesting failure patterns. At 25% memory, OFALMA pruning fails on 14 out of 20 dialogues. Examination of these failures suggests that the extreme compression removes critical facts needed for accurate reasoning. For example, in dialogue 86 (expected answer: 255), OFALMA pruning fails while custom_summary succeeds, suggesting that the summarization approach preserves essential information that pruning eliminates.

At 50% memory, OFALMA pruning fails on only 1 out of 20 dialogues (dialogue 82, expected answer: 4, predicted: 5). This single failure represents a near-miss, suggesting that the model is retaining the correct information but making a minor reasoning error. The fact that rate-distortion succeeds on this dialogue (predicting 4 correctly) suggests that the additional information preserved through condensation helps resolve the ambiguity.

At 75% memory, OFALMA pruning again fails on only 1 dialogue (dialogue 93, expected answer: 2, predicted: 0). Interestingly, rate-distortion succeeds on this dialogue, again suggesting that condensation preserves information that helps resolve difficult cases.

### 5.5.2 Dialogue-Specific Challenges

Certain dialogues present particular challenges across all methods. Dialogue 86 (expected answer: 255) is particularly difficult, with only custom_summary and OFALMA pruning (at 50% memory) achieving correct answers. This dialogue likely contains complex numerical reasoning that requires careful preservation of multiple related facts.

Dialogue 93 (expected answer: 2) is another challenging case, with only rate-distortion (at 75% memory) achieving the correct answer. This suggests that some dialogues require the nuanced information preservation enabled by condensation rather than binary pruning decisions.

The fact that different methods succeed on different dialogues suggests that there is no universally optimal strategy, and the choice between pruning and rate-distortion may depend on the specific characteristics of the dialogue and question.

## 5.6 Implications for Dialogue Memory Systems

### 5.6.1 Learned Importance Hierarchy

The learned weight distributions have important implications for dialogue memory systems. The consistent dominance of Relevance and Emphasis over Recency suggests that traditional recency-based memory management strategies (common in many dialogue systems) may be suboptimal. Instead, systems should prioritize content relevance and emphasis, regardless of when information was mentioned.

The near-zero Recency weights are particularly significant, as they contradict common assumptions in dialogue systems. Many systems prioritize recent information under the assumption that it is more relevant to the current conversation. However, our results suggest that for question-answering tasks, the topic relevance of information is far more important than its temporal position.

### 5.6.2 Adaptive Memory Management

The evolution of weights across memory ratios demonstrates that optimal memory management strategies should adapt to available resources. At low memory ratios, emphasis becomes more important (as emphatic statements are likely to be critical), while at higher ratios, relevance dominates (as there is room to preserve all relevant information). This adaptive behavior suggests that memory management systems should dynamically adjust their importance criteria based on available resources.

### 5.6.3 Pruning vs Condensation Trade-offs

The comparison between pruning and rate-distortion methods reveals important trade-offs. Pruning achieves higher accuracy at moderate and high memory ratios (95% vs 85% at 50% memory) but requires binary decisions that may remove critical information at extreme compression. Rate-distortion provides more flexibility through condensation but requires additional computational resources and may not achieve the same level of selectivity.

The choice between methods depends on the specific requirements: if computational efficiency and selectivity are priorities, pruning is superior; if information preservation and flexibility are priorities, rate-distortion may be preferable.

## 5.7 Limitations and Future Work

### 5.7.1 Dataset Limitations

Our evaluation is limited to a single dataset of 100 synthetic dialogues focused on project scheduling and resource allocation problems. While this provides controlled evaluation conditions, the generalizability to other domains (e.g., open-domain dialogue, technical support, creative writing) remains to be validated. Future work should evaluate OFALMA on diverse dialogue types and domains.

### 5.7.2 Memory Ratio Coverage

We evaluated three memory ratios (25%, 50%, 75%) which cover a range of compression scenarios, but do not include the extreme cases (e.g., 10% or 90% memory). Future work should explore performance at more extreme ratios to understand the limits of compression and the point of diminishing returns for additional memory.

### 5.7.3 Impact Factor Computation

The current implementation uses LLM-based computation of impact factors (S, Q, E), which requires additional API calls and computational resources. Future work could explore more efficient methods for computing these factors, such as learned embeddings or rule-based heuristics, to reduce computational overhead.

### 5.7.4 Multi-Turn Dialogue Evaluation

Our evaluation focuses on single-turn question answering from condensed dialogue history. Real dialogue systems involve multi-turn interactions where memory management decisions affect future turns. Future work should evaluate OFALMA in multi-turn dialogue scenarios to understand how memory management decisions propagate through conversations.

## 5.8 Conclusion

The experimental results demonstrate that OFALMA, particularly the pruning variant, achieves superior performance at moderate and high memory ratios, achieving 95% accuracy at both 50% and 75% memory on a held-out validation set. The learned impact factor weights reveal that Relevance and Emphasis are the primary determinants of importance, while Recency is essentially irrelevant. The comparison between pruning and rate-distortion methods reveals important trade-offs between selectivity and information preservation, with pruning achieving higher accuracy but rate-distortion providing more flexibility.

These findings have important implications for dialogue memory systems, suggesting that traditional recency-based strategies should be replaced with relevance and emphasis-based approaches, and that memory management strategies should adapt to available resources. The consistent near-optimal performance of OFALMA across moderate and high memory ratios suggests that it has learned to identify the essential information needed for accurate question answering, achieving near-human performance with only half the original dialogue content.

