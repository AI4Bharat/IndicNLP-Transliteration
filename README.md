# IndianNLP-Transliteration

The main goal of this project is to create open source input tools for content creation in under-represented languages in India. <br>
It started in collaboration with [**Story Weaver**](https://storyweaver.org.in/about) a non-profit working towards  foundational literary education for children, supported by [**Google's AI for Social Good**](https://india.googleblog.com/2020/02/applying-ai-to-big-problemssix-research.html) initiative.

Most languages in India do not have digital presence due to an underdeveloped ecosystem.  One of the major bottlenecks in content creation and language adoption, is difficulty to input text in several native Indian languages. Lack of stable input tools in underserved languages is huge barrier for creating digital content and NLP datasets in these languages.


**Supported Languages**
* Bengali - বাংলা
* Gujarati - ગુજરાતી
* Hindi - हिंदी
* Kannada - ಕನ್ನಡ
* Konkani Goan - कोंकणी
* Maithili - मैथिली
* Malayalam - മലയാളം
* Marathi - मराठी
* Panjabi Eastern - ਪੰਜਾਬੀ
* Sindhi - سنڌي‎
* Sinhala - සිංහල
* Telugu - తెలుగు
* Tamil - தமிழ்
* Urdu - اُردُو

---
## Repository Usage

For Attributions and Contributions lists, [check here](docs/attributions) 🖖

### Training Procedures

This repository is developed to facilate easier experimentation with different network architecture models, reformulated objectives with minimal effort and highly tinkerable, rather than a offshelf library. <br>

A Condensed standalone version of a simple model training, inferencing and accuracy computation is created as jupyter notebook.<br>
[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI4Bharat/IndianNLP-Transliteration/blob/jgeob-dev/NoteBooks/Xlit_TrainingSetup_condensed.ipynb)

### Pythonic Library

Pythonic transliteration library is available as [Python Package Index](https://pypi.org/project/ai4bharat-transliteration/) and also under github releases. <br>
Follow usages in [apps readme](apps/README.md).

---

## NeuralNet Models

Transliteration models for languages are made available as releases, in a easy deployable way.

All the NN models (along with metadata) of Xlit - Transliteration are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY SA 4.0][cc-by-sa-image]][cc-by-sa]

---
## Datasets

Datasets created as part of the project for languages Maithili, Konkani, Hindi are made available as JSON files under [downloads](https://github.com/AI4Bharat/IndianNLP-Transliteration/releases/tag/DATA).

Xlit - Transliteration Datasets by [Story Weaver](https://storyweaver.org.in/) & [AI4Bharat](https://ai4bharat.org/) are licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

Kindly attribute if you use the dataset for your research or products

---
## Contact
If you have benefited by our datasets/models/services or got motivated by our works, we would like to hear from you.

email: opensource@ai4bharat.org

---



[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png

[cc-by-sa]: http://creativecommons.org/licenses/by/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png