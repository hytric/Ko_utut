# [Project] Korean Audio, Multilingual HuBERT Translation Training Guideline

![Slide10.jpg](/image/Slide10.jpg)

Recent advancements in multilingual Audio-to-Audio translation propose the use of HuBERT for direct translation.

- [Textless Unit-to-Unit training for Many-to-Many Multilingual Speech-to-Speech Translation](https://arxiv.org/abs/2308.01831)  
- [AV2AV: Direct Audio-Visual Speech to Audio-Visual Speech Translation with Unified Audio-Visual Speech Representation](https://arxiv.org/abs/2312.02512)

These studies support a variety of languages, but **Korean is not included**.

The goal of this project is to **train a model that supports Korean translation** alongside other languages.

Since training code is currently not provided, this guide offers a **detailed step-by-step guideline**.

### **Framework:**
- Code is implemented based on **Fairseq**.

---

## **Dataset**

### 1. [Multilingual Speech Translation Dataset](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71524)
- A collection of multilingual read speech datasets designed for translation tasks.

### 2. [Professional Korean-English Translation Dataset for International Conferences](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71693)
- Specialized datasets designed for professional Korean-English and English-Korean translation tasks.

These datasets serve as the foundation for training multilingual HuBERT models with Korean language support.

---

Stay tuned for the detailed training pipeline and setup instructions!
