# Models

## Model Structure
Models are able to be saved and loaded from the ```models/``` directory.  At this stage, only one model is included in this set. 

## Dataset Acknowledgements
This project leverages the following datasets, and a great thank you is extended to those that share Cybersecurity data.

* [NetLab 360's DGA Feed](https://data.netlab.360.com/dga/)  
Qihoo 360 Technology Co. Ltd.'s Security Lab, they graciously provide an open source feed of domains created with malware strains.  This feed is created by sifting through Passive DNS and provides some indication about the origins of the original domain.  

* [Andreweva's dgaDomains](https://github.com/andrewaeva/DGA)  
Developed for research, Andrew's responsitory contains a collection of the dga Algorithms for research purposes.  

* [Alexa Top 1 Million](https://www.alexa.com/)  
Alexa provides market research, search engine optimisation and key word analysis.  They generously offer details about the top 1 million websites ranked publicly.  This dataset was leveraged for a large number of benign domains.

### Note:
Many of these datasets and feeds are updated periodically, it may be worth replacing these files within the ```datasets/``` folder.  At this stage there has been no effort to determine the impact of new strains versus those that have already been trained. 