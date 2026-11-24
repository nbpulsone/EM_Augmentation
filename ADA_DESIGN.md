# Active Data Augmentation (ADA) for LEMONADE — Technical Design Spec

## 1. Research Context & Objective
We are refactoring the LEMONADE entity matching framework (`lemonade3576/lemonade_kbs25`) from static, offline LLM data augmentation (`LLM-DA`) into a **closed-loop Active Data Augmentation (ADA)** pipeline.

- **Base Architecture:** Bi-encoder / cross-encoder SLM (DeBERTa-v3-base or ModernBERT) paired with LEMONADE's Knowledge Transfer (`LLM-KT`) module (frozen LLM embedding encoder + differential autoencoder + consistency-aware alignment loss).
- **Core Contribution:** Replace upfront synthetic negative generation ($C=1$ for all positive anchors) with iterative, uncertainty-driven synthesis during training. This targets decision-boundary ambiguity while reducing LLM API/token spend to a fractional budget ($C_{\text{eff}} < 1.0$).

## 2. Closed-Loop Training Schedule
1. **Seed Pool ($\mathcal{D}_{\text{train}}^{(0)}$):** Initialize with only original labeled positive pairs ($\mathcal{P}, y=1$) and negative pairs ($\mathcal{N}, y=0$). Zero synthetic negatives at Epoch 0.
2. **Warm-Up Phase ($E_{\text{warm}}$):** Train purely on $\mathcal{D}_{\text{train}}^{(0)}$ for 3–5 epochs so SLM probabilities are calibrated before querying the LLM.
3. **Periodic Acquisition Rounds ($R$ rounds, every $E_{\text{step}} = 2\text{--}3$ epochs):**
   - Run a `torch.no_grad()` inference sweep over remaining **unaugmented** positive anchors $\mathcal{P}_{\text{cand}}$.
   - Score ambiguity and select the top-$K$ anchors **without replacement**.
   - Query the generative LLM (open-weight via vLLM/Ollama such as Qwen/Llama-3.1-8B, or Gemini API) to perturb one record in each selected anchor pair into a hard counterfactual negative ($y = 0$), validating that the output differs from the original record.
   - Compute frozen teacher embeddings $(e_{\text{syn}}, e_{\text{syn}}')$ for the newly synthesized pairs at runtime.
   - Append new pairs, labels ($y=0$), and embeddings to the active training buffer $\mathcal{D}_{\text{train}}^{(t+1)}$ and refresh the `DataLoader`.
4. **Consolidation / Cool-Down Phase:** After all $R$ rounds complete (or budget $B_{\max}$ is reached), train for 3–5 final epochs with no new synthesis so the SLM and autoencoder converge smoothly.

## 3. Acquisition Function & Budget Math
- **Target Fractional Budget ($C_{\text{eff}}$):** Defined as $C_{\text{eff}} = N_{\text{syn}} / N_a$ (default target: $C_{\text{eff}} = 0.40$ across $R = 4$ rounds, never exceeding $C_{\text{eff}} \le 1.0$ to prevent negative-class swamping).
- **Dynamic Query Size ($K$):** Computed per dataset from the number of positive anchors $N_a$:
  $$K = \max\left(1, \left\lfloor \frac{C_{\text{eff}} \times N_a}{R} \right\rfloor\right)$$
- **Uncertainty Scoring (Consistency-Gated Margin Sampling):**
  - Margin uncertainty: $u_i = 1 - 2|p_i - 0.5|$, where $p_i = P_\theta(y=1 \mid r_1, r_2)$.
  - If LEMONADE's teacher consistency weight $\omega_i^c$ is available, gate the score to avoid querying noisy/flipped labels:
    $$u_i^{\text{gated}} = \omega_i^c \cdot (1 - 2|p_i - 0.5|)$$
  - Mask out already-augmented anchors (`scores.masked_fill(augmented_mask, -1.0)`) and take `torch.topk(scores, k)`.
  - *(Optional diversity tie-breaker: cluster top-$2K$ embeddings via K-Means and pick highest score per cluster.)*

## 4. Key Engineering Constraints
- **Class Imbalance Guard:** Sample positive anchors strictly without replacement and cap total synthetic negatives at $C_{\text{eff}} \times N_a \le N_a$.
- **Loss Compatibility:** Newly generated pairs must expose both serialized text (for SLM matching loss $\mathcal{L}_m$) and frozen teacher embeddings (for autoencoder differential reconstruction $\mathcal{L}_r^d$ and consistency-weighted alignment $\alpha \omega^c \mathcal{L}_a$).
- **Deployment:** Only the fine-tuned SLM is used at inference time.
