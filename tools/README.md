# Tools

For task independent Scripts and Programs like <br>
* Preprocessing, cleaning data <br>
* Analytics and Plots on data <br>
* Analytics on inference <br>


### Contents

1. Accuracy_reporter - Used for computing accuracy (NEWS metrics) from the predicted json. <br>
*Usage:* <br>
  a. Configure orchestrator.py with the exact truth files and topk and save directory for results
  b. Configure the tasks/infer_engine.py with appropriate model object (loaded from *_runner.py files of some other files) and correspondin weight files. <br>
  Note: The orchestrator invokes the inferencer function from infer_engine.py to run inference and internally runs the accuracy_news.py script

2. translit_engine.py - Used for running transliteration API from Google/Varnam/Quillpad and compile the results.

3. embeddings - scripts & Notebook for computing pretrained embeddings

4. visualization - scripts and Notebooks for Visualizing and analysis



