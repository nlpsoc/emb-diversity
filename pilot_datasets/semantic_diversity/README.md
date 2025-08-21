# Pilot Datasets for Semantic Diversity

## Summary
This folder contains small, ready-to-use **pilot datasets** derived from **MRPC, QQP, STS-B, and Reuters-21578**.  
They are constructed to facilitate controlled comparisons of **semantic diversity** in NLP experiments.


## Sources
- **MRPC** — Microsoft Research Paraphrase Corpus (sentence pairs, paraphrase labels).
- **QQP** — Quora Question Pairs (sentence pairs, duplicate/non-duplicate labels).
- **STS-B** — Semantic Textual Similarity Benchmark (sentence pairs, real-valued similarity scores 0–5).
- **Reuters‑21578** — Newswire collection (1987, SGML).

> MRPC, QQP, and STS-B are *pairwise* datasets (`text_a`, `text_b`) with semantic equivalent labels; Reuters is single-document text with topic labels.


## Construction Details

### Semantic Textual Similarity Datasets: MRPC, QQP, STS‑B

#### Design principle
To enable fair comparison between **low** and **high** semantic diversity, the two sets **share 50% identical content**.  
The remaining 50% differs by condition:  
- In the **low_diversity** sets, the other half consists of sentences from semantically equivalent pairs.  
- In the **high_diversity** sets, the other half consists of completely unrelated sentences.  

This guarantees that, on top of the shared 50%, the high-diversity sets are strictly more semantically varied.

#### MRPC & QQP
- **Low-diversity** (`*_low_diversity.txt`, 400 lines):  
  - Select **200 positive pairs** *(label = 1, semantically equivalent)*.  
  - Write **both** `text_a` **and** `text_b` from these pairs → **400 sentences**.
- **High-diversity** (`*_high_diversity.txt`, 400 lines):  
  - Reuse the **same 200 `text_a`** from the low-diversity set (*shared base*).  
  - Add **200 unrelated sentences** taken from 200 as `text_a` of **negative pairs** *(label = 0)*.  
  - → **400 sentences total**, with **200 lines overlapping** exactly with the low set (the shared `text_a`).

> Note: Only `text_a` is reused across sets to make the overlap explicit and exact; `text_b` from the low set is **not** reused in the high set.

#### STS‑B
- **Scoring**: 0 = unrelated, 5 = semantically identical.
- **Low_diversity** (`stsb_low.txt`, 400 lines):  
  - Select **200 pairs** with **score = 5**.  
  - Write both `text_a` and `text_b` → **400 sentences**.
- **High_diversity** (`stsb_high.txt`, 400 lines):  
  - Reuse the **same 200 `text_a`** from the low_diversity set (*shared base*).  
  - Add **200 unrelated sentences** from the `text_a` of **score = 0** pairs.  
  - → **400 sentences total**, with **200 lines overlapping** with the low set.

### Topic Diversity Datasets: Reuters‑21578
- **Single-topic** documents selected from Reuters-21578 SGML.  
- **80 samples in total for each split**, differing by topic counts: 2, 4, 6, or 8.
- **Topics included**:  
  - **2-topic**: jobs, crude (40 samples for eahc topic)
  - **4-topic**: jobs, crude, money-fx, trade (20 samples for eahc topic)
  - **6-topic**: jobs, crude, money-fx, trade, sugar, ship (13/14 samples for eahc topic)
  - **8-topic**: jobs, crude, money-fx, trade, sugar, ship, coffee, cocoa (10 samples for eahc topic)
- **Assumption**: more topics ⇒ higher semantic diversity.  
- **Normalization**:
  - BODY collapsed to a single line (newlines → spaces; collapse consecutive whitespace; trim).
  - Remove common SGML artifacts and Reuters markers (e.g., “Reuter”, HTML entities).
  - Tabs → spaces.
