# Comparative Analysis of Locally Fine-Tuned Open Models vs. Frontier Models for Automated Cybersecurity Operations

**Abstract**—The rapid evolution of cyber threats necessitates advanced automated analysis systems. While frontier models like Gemini 2.5 provide state-of-the-art reasoning capabilities, their deployment in sensitive security environments is often constrained by privacy, latency, and cost concerns. This study evaluates whether smaller, locally fine-tuned open-source models (Llama-3, Mistral, Qwen, DeepSeek) can approach frontier-level performance through advanced training strategies, specifically Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), and Proximal Policy Optimization (PPO). We introduce a "Silver-Standard Distillation" framework to transfer reasoning chains from frontier models to local experts and evaluate them using a multi-dimensional cybersecurity benchmark. Our results provide insights into the performance-efficiency tradeoffs of local security-specialized models.

## I. Introduction
The burden on Security Operations Center (SOC) analysts is increasing due to the volume of threat telemetry. LLMs show promise in automating report analysis. However, organizations are hesitant to send sensitive telemetry to public APIs. This research explores the viability of "distilled local experts" as an alternative.

## II. Related Work
Recent efforts in benchmarking LLMs for cybersecurity have shifted from general knowledge evaluation (e.g., CyberMetric [2024]) to specialized task-oriented frameworks. Key recent benchmarks include:
- **CTIBench (2024)**: Evaluates LLMs on MITRE ATT&CK mapping and vulnerability analysis.
- **CyberTeam (2025)**: A blue-team focused testbed for threat hunting workflows across 23 vulnerability databases.
- **CVE-Bench (2025)**: A sandbox-based evaluation for AI agents on real-world exploit generation.
Our work complements these by focusing specifically on the *fine-tuning efficiency* of smaller open-source models using reasoning distillation.

## III. Methodology

### A. Formalizing Cybersecurity Evaluation Metrics
To provide a rigorous assessment of local model performance, we define the following multi-dimensional metrics:

1.  **Accuracy ($A$):** The empirical accuracy over as set of $N$ threat reports is defined as:
    $$A = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(Y_{pred, i} = Y_{true, i})$$
    where $Y_{pred, i}$ is the set of extracted threat labels and $\mathbb{1}$ is the indicator function.

2.  **Technical Precision ($P$):** We define $P$ as the domain-specific technical density:
    $$P = \frac{|V_{cyber} \cap V_{model}|}{|V_{model}|}$$
    where $V_{cyber}$ represents a curated lexicon of 15 high-fidelity cybersecurity primitives (e.g., *persistence*, *registry manipulation*, *adversarial TTPs*) and $V_{model}$ is the set of unique tokens in the model's response.

3.  **Reasoning Density ($D$):** A measure of explicit latent reasoning:
    $$D = \frac{\tau_{thought}}{\tau_{total}}$$
    where $\tau_{thought}$ is the number of tokens contained within $\langle thought \rangle$ tags and $\tau_{total}$ is the total response token count.

### B. Group Relative Policy Optimization (GRPO) Modeling
Unlike traditional PPO, GRPO eliminates the value head by computing advantages relative to a group of $G$ outputs. The objective function $\mathcal{J}_{GRPO}(\theta)$ is defined as:
$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P, \{o_k\}_{k=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{k=1}^G \min \left( \frac{\pi_\theta(o_k|q)}{\pi_{\theta_{old}}(o_k|q)} A_k, \text{clip}\left(\frac{\pi_\theta(o_k|q)}{\pi_{\theta_{old}}(o_k|q)}, 1-\epsilon, 1+\epsilon\right) A_k \right) \right]$$
where the advantage $A_k$ is standardized within the group:
$$A_k = \frac{r_k - \text{mean}(r)}{\text{std}(r)}$$

### C. Multi-Objective Reward Architecture
The total reward $R_{total}$ for a model output is a weighted sum of format, correctness, and reasoning quality components:
$$R_{total} = w_f \cdot R_{format} + w_a \cdot R_{accuracy} + w_c \cdot R_{coherence}$$
where:
- $w_f=0.5$ (Format Adherence: Binary $\langle thought \rangle \dots \langle answer \rangle$ verification)
- $w_a=1.0$ (Task Correctness)
- $w_c=1.0$ (Reasoning Quality: Function of length and technical marker presence)

## V. Experiments
### A. Data Distillation
Extraction of $N=10,000$ reasoning chains from Gemini-2.5 based on raw sandbox reports (Hybrid-Analysis, VirusTotal).

### B. Ablation Studies
- Effect of reasoning length on downstream accuracy.
- Performance impact of SFT vs. GRPO in multi-step malware analysis.

## VI. Experimental Results (Baseline Evaluation)

In our initial baseline evaluation, we assessed three models: Baseline Gemini (Flash 2.5), Baseline Llama-3-8B, and Baseline Mistral-7B.

| Model             | Accuracy | Technical Precision | Reasoning Density | Words (Avg) |
|-------------------|----------|---------------------|-------------------|-------------|
| Baseline Gemini   | 0.0%     | 0.112               | 0.0               | 507.35      |
| Baseline Llama3   | 0.0%     | 0.091               | 0.0               | 208.45      |
| Baseline Mistral  | 0.0%     | 0.099               | 0.0               | 281.85      |

The 0% accuracy and reasoning density indicate a significant gap in specialized SOC analysis. This justifies the need for Supervised Fine-Tuning (SFT) and alignment.

## VII. Discussion
- Can 8B models replace 100B+ models for narrow SOC tasks?
- The "Security Privacy vs. Intelligence" tradeoff.

## VIII. Limitations and Future Work
- Generalization to unseen malware families.
- Real-time training on live SOC telemetry.

---
**Keywords**: Cybersecurity, LLM, Fine-Tuning, GRPO, SFT, PPO, Distillation.
