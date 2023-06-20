
Code for paper  `Beyond Triplet: Leveraging the Most Data for Multimodal Machine Translation`



### 0. Requriment 

`pip install -r requirements.txt`
and open AI CLIP https://github.com/openai/CLIP



### 1. Download Data

get data from https://huggingface.co/datasets/Yaoming95/EMMT

### 2. process data

use `data_preprocessor.py` to preprocess data into fairseq/CLIP emb form.

### 3. model arch

see `fairseq` dir


### Citation
```
@inproceedings{ZhuSCHWW23,
  author       = {Yaoming Zhu and
                  Zewei Sun and
                  Shanbo Cheng and
                  Luyang Huang and
                  Liwei Wu and
                  Mingxuan Wang},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Beyond Triplet: Leveraging the Most Data for Multimodal Machine Translation},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {2679--2697},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.168},
  doi          = {10.18653/V1/2023.FINDINGS-ACL.168},
  timestamp    = {Thu, 10 Aug 2023 12:35:59 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/ZhuSCHWW23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```