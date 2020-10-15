# IndianNLP-Transliteration

The main goal of this project is to create open source input tools for under-represented languages in India. It started in collaboration with [**Story Weaver**](https://storyweaver.org.in/about) a non-profit working towards  foundational literary education for children, supported by
[**Google's AI for Social Good**](https://india.googleblog.com/2020/02/applying-ai-to-big-problemssix-research.html) initiative, inorder to create better tools for content creation.

Most languages in India do not have digital presence due to an underdeveloped ecosystem.  One of the major bottlenecks in content creation and language adoption, is difficulty to input text in several native Indian languages. Lack of stable input tools in underserved languages is one of the biggest roadblocks to creating digital content and datasets in these languages.


**Supported Languages**
* Hindi [hi]
* Konkani - Goan [gom]
* Maithili [mai]

---
## Repository Usage

### For Training

This repository is developed to facilate easier experimentation with different network architecture models, reformulated objectives with minimal effort and highly tinkerable, rather than a offshelf library. <br>

A Condensed standalone version of a simple model training, inferencing and accuracy computation is created as jupyter notebook.<br>
[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI4Bharat/IndianNLP-Transliteration/blob/jgeob-dev/NoteBooks/Xlit_TrainingSetup_condensed.ipynb)


### For Transliteration API

Transliteration models for languages is made available to be download and deploy.
Please download the latest release and follow procedures in [readme](https://github.com/AI4Bharat/IndianNLP-Transliteration/blob/jgeob-dev/apps/README.md).

---
## Data
Datasets created as part of the project for languages Maithili, Konkani, Hindi are made available as JSON files under [downloads](https://github.com/AI4Bharat/IndianNLP-Transliteration/releases/tag/DATA).

Xlit - Transliteration Dataset by [Story Weaver](https://storyweaver.org.in/) & [AI4Bharat](https://ai4bharat.org/) is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]


---
## Contact
email: opensource@ai4bharat.org

---



[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png

[cc-by-sa]: http://creativecommons.org/licenses/by/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png