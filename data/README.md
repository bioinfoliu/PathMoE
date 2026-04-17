# 📥 Raw Data Download Instructions

Due to GitHub's file size limits and data sharing best practices, the raw multi-omics datasets are not included in this repository. Please follow the instructions below to download the required data and place them in this directory.

## 1. MSigDB Hallmark Gene Sets
The MoPE-MOI architecture requires the hallmark pathway definitions as structural priors.
* **Download Link:** [GSEA MSigDB Downloads](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp)
* **Target File:** `h.all.v2023.1.Hs.symbols.gmt` (or the latest version)
* **Action:** Save this `.gmt` file directly into this `data/` folder.

## 2. TCGA Multi-omics Data
The pan-cancer subtyping and survival experiments utilize RNA-seq, Copy Number Variation (CNV), and DNA Methylation data from the TCGA cohorts. 
* **Recommended Source:** We recommend using the [UCSC Xena Browser](https://xenabrowser.net/datapages/) for pre-processed and normalized TCGA datasets.
* **Required Modalities:**
  * RNA-seq (HTSeq - TPM or FPKM)
  * Gene-level Copy Number (gistic2)
  * DNA Methylation (Illumina Human Methylation 450k)
  * Clinical data (for survival endpoints and subtype labels)


##  Expected Directory Structure
After downloading and running our preprocessing scripts, your `data/` folder should look like this:

```text
data/
├── raw_data/
│   ├── 0_download_and_process_omics.py
│   └── README.md
├── h.all.v2023.1.Hs.symbols.gmt
├── BRCA/
│   ├── filtered_subtype_data/
│   └── ...
└── LUAD/
    ├── filtered_subtype_data/
    └── ...


