# [프로젝트] Korean Audio, Multilingual Hubert translate Training Guideline

![Slide10.jpg](/image/Slide10.jpg)

최근 Audio2Audio multilingual translate에서 hubert를 활용한 direct translation이 제안되었다.

[Textless Unit-to-Unit training for Many-to-Many Multilingual Speech-to-Speech Translation](https://arxiv.org/abs/2308.01831)  
[AV2AV: Direct Audio-Visual Speech to Audio-Visual Speech Translation with Unified Audio-Visual Speech Representation](https://arxiv.org/abs/2312.02512)


논문에서는 다양한 언어를 지원하지만, 한국어를 지원하지는 않는다.

그래서 한글도 같이 지원하도록 모델을 학습하는 것이 목표이다.

현재 Training code가 제공되지 않기 때문에, 자세한 가이드라인도 같이 제공하고자 한다.


**fairseq를 기반으로 코드 작성**
 
<br>

## Dataset

[다국어 통번역 낭독체 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71524)

[국제 학술대회용 전문분야 한영/영한 통번역 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71693)
