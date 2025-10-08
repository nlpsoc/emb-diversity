# Pilot Datasets for Semantic Diversity

## Summary
This folder contains small, ready-to-use **pilot datasets** for controlled comparisons of **semantic diversity** in NLP experiments. The datasets are organized into two main categories: **STS-based** (semantic textual similarity) and **topic-based** (topic diversity).

## Dataset Categories

### 1. STS-based Datasets (`STS_based/`)
Semantic textual similarity datasets derived from pairwise sentence comparison tasks.

#### MRPC (Microsoft Research Paraphrase Corpus)
- **Purpose**: Paraphrase detection (sentence pairs with semantic equivalence labels)
- **Files**: 
  - `mrpc_low_diversity.txt` (400 sentences)
  - `mrpc_high_diversity.txt` (400 sentences)
- **Construction**: 50% shared content + 50% semantically equivalent (low) vs. unrelated (high) sentences

#### QQP (Quora Question Pairs)
- **Purpose**: Duplicate question detection (question pairs with duplicate/non-duplicate labels)
- **Files**:
  - `qqp_low_diversity.txt` (400 sentences)
  - `qqp_high_diversity.txt` (400 sentences)
- **Construction**: 50% shared content + 50% semantically equivalent (low) vs. unrelated (high) sentences

#### STS-B (Semantic Textual Similarity Benchmark)
- **Purpose**: Semantic similarity scoring (sentence pairs with 0-5 similarity scores)
- **Files**:
  - `stsb_low_diversity.txt` (400 sentences)
  - `stsb_high_diversity.txt` (400 sentences)
- **Construction**: 50% shared content + 50% high-similarity (score=5, low) vs. unrelated (score=0, high) sentences

### 2. Topic-based Datasets (`topic_based/`)
Single-document datasets with topic labels for measuring topic diversity.

#### Reuters-21578
- **Purpose**: News classification (newswire articles with topic labels)
- **Files**: `reuters_2-topic.txt`, `reuters_4-topic.txt`, `reuters_6-topic.txt`, `reuters_8-topic.txt`
- **Samples**: 80 per file
- **Topics**: 
  - **2-topic**: jobs, crude (40 samples each)
  - **4-topic**: jobs, crude, money-fx, trade (20 samples each)
  - **6-topic**: jobs, crude, money-fx, trade, sugar, ship (13/14 samples each)
  - **8-topic**: jobs, crude, money-fx, trade, sugar, ship, coffee, cocoa (10 samples each)
- **Assumption**: More topics → higher semantic diversity

#### AgNews
- **Purpose**: News classification (news articles with 4 main categories)
- **Files**: `AgNews_1-topics.txt`, `AgNews_2-topics.txt`, `AgNews_3-topics.txt`, `AgNews_4-topics.txt`
- **Samples**: 1000 per file
- **Topics**: 
  - **1-topic**: Business only
  - **2-topics**: Sports, Business
  - **3-topics**: Sports, Business, Sci_Tech
  - **4-topics**: Business, Sci_Tech, World, Sports

#### 20NewsGroups
- **Purpose**: Newsgroup classification (newsgroup posts with hierarchical topic labels)
- **Files**: `20NewsGroups_2-topics.txt` through `20NewsGroups_20-topics.txt`
- **Samples**: 1000 per file
- **Topics**: 
  - **2-topics**: comp.sys.ibm.pc.hardware, alt.atheism
  - **4-topics**: sci.med, comp.sys.ibm.pc.hardware, rec.sport.hockey, talk.religion.misc
  - **6-topics**: rec.sport.hockey, rec.autos, sci.electronics, soc.religion.christian, comp.sys.mac.hardware, rec.sport.baseball
  - **8-topics**: talk.politics.guns, rec.motorcycles, sci.electronics, rec.autos, talk.politics.mideast, comp.windows.x, sci.med, sci.space
  - **10-topics**: comp.sys.mac.hardware, rec.motorcycles, talk.politics.misc, talk.politics.mideast, alt.atheism, sci.crypt, comp.graphics, rec.sport.baseball, comp.os.ms-windows.misc, sci.electronics
  - **12-topics**: alt.atheism, comp.windows.x, rec.sport.baseball, misc.forsale, soc.religion.christian, talk.politics.misc, rec.autos, comp.sys.ibm.pc.hardware, rec.motorcycles, sci.electronics, talk.politics.mideast, sci.space
  - **14-topics**: rec.sport.baseball, sci.med, rec.sport.hockey, comp.os.ms-windows.misc, talk.politics.guns, comp.graphics, sci.crypt, comp.sys.ibm.pc.hardware, sci.space, soc.religion.christian, talk.religion.misc, alt.atheism, talk.politics.misc, comp.windows.x
  - **16-topics**: talk.politics.guns, rec.autos, sci.crypt, sci.med, soc.religion.christian, comp.sys.mac.hardware, sci.electronics, comp.sys.ibm.pc.hardware, alt.atheism, talk.politics.mideast, rec.sport.hockey, talk.religion.misc, misc.forsale, comp.windows.x, talk.politics.misc, rec.sport.baseball
  - **18-topics**: talk.politics.mideast, soc.religion.christian, rec.sport.baseball, rec.sport.hockey, comp.os.ms-windows.misc, sci.crypt, comp.graphics, talk.religion.misc, sci.med, rec.motorcycles, sci.space, talk.politics.guns, talk.politics.misc, misc.forsale, comp.sys.mac.hardware, alt.atheism, comp.sys.ibm.pc.hardware, rec.autos
  - **20-topics**: talk.religion.misc, talk.politics.misc, comp.graphics, soc.religion.christian, talk.politics.mideast, rec.autos, comp.windows.x, sci.med, alt.atheism, comp.sys.mac.hardware, sci.space, misc.forsale, talk.politics.guns, comp.sys.ibm.pc.hardware, sci.electronics, rec.sport.hockey, comp.os.ms-windows.misc, rec.motorcycles, rec.sport.baseball, sci.crypt

## Usage Notes

### STS-based Datasets
- **Design Principle**: Each pair (low/high) shares exactly 50% identical content
- **Fair Comparison**: The remaining 50% differs by semantic similarity condition
- **File Format**: One sentence per line, plain text

### Topic-based Datasets
- **Design Principle**: Controlled topic diversity through systematic topic selection
- **Progressive Complexity**: Increasing number of topics per dataset
- **File Format**: One document per line, plain text
- **Normalization**: 
  - Newlines collapsed to spaces
  - Consecutive whitespace collapsed
  - Common artifacts removed (e.g., "Reuter" markers, HTML entities)
