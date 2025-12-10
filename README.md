# GUIDÆTA
## A Versatile Interactions Dataset with extensive Context Information and Metadata

![PyPI](https://img.shields.io/pypi/v/guidaeta?style=flat-square)
[![GitHub Repo](https://img.shields.io/badge/GitHub-lenxn%2Fapchis--guidaeta-blue?logo=github&style=flat-square)](https://github.com/lenxn/apchis-guidaeta)
![License](https://img.shields.io/github/license/lenxn/apchis-guidaeta?style=flat-square)
![Downloads](https://img.shields.io/pypi/dm/guidaeta?style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.1234%2Fabcd.56789-blue)](https://doi.org/10.2312/stag.202513359)



The GUIDÆTA interactions dataset was collected within an online user study in the [A+CHIS](https://apchis.cgv.tugraz.at/) project on Human-Centered Interactive Adaptive Visual Approaches in High-Quality Health Information, conducted between May 12 and June 23, 2025.

Extensive information on scope, data collection, metadata, preliminary analysis can be found in the respective [paper]() ("GUIDÆTA – A Versatile Interactions Dataset with extensive Context Information and Metadata"), published at Smart Tools and Applications in Graphics ([STAG](https://www.stag-conference.org/2025/)) 2025, an annual international conference organized by the Italian Chapter of the Eurographics association.

The raw dataset is publicly available on OSF: [https://osf.io/fhvbm/](https://osf.io/fhvbm/). 

### Install

	pip install guidaeta

### Usage

This repository contains everything necessary for handling the raw dataset. 
A few selected use cases are given below:

	import guidaeta


	# The path pointing to the root of the data
	DATA_ROOT = "<PATH_TO_YOUR_DATA_ROOT>"

	# load everything from files and initialize
	sentences, sessions, users, task_answers = guidaeta.load_data(DATA_ROOT)

	# get the cognitive load experienced for task two
	task_two_answers = [ta for ta in task_answers if ta.task_no==2]
	cl = sum([
		sum(ta.cl.scores)/len(ta.cl.scores) for ta in task_two_answers
	])/len(task_two_answers)

	# get all sessions associated with the 3rd task
	s = [s for ta in task_answers for s in ta.sessions if ta.task_no==3]

### Contributing

Contributions are welcome!  
Feel free to open issues or pull requests at  
https://github.com/lenxn/apchis-guidaeta

### Acknowledgement

This work was funded by the Austrian Science Fund (FWF) as part of the project 'Human-Centered Interactive Adaptive Visual Approaches in High-Quality Health Information' (A<sup>+</sup>CHIS; Grant No. FG 11-B).

When referring to the dataset please cite:

* S. Lengauer, S.A. von Götz, M.T. Hoesch, F. Steinwidder, M. Tytarenko, M.A. Bedek, T. Schreck, *GUIDÆTA - A Versatile Interactions Dataset with extensive Context Information and Metadata*, 2025, DOI: [10.2312/stag.202513359](https://doi.org/10.2312/stag.202513359).

Bibtex:

	@inproceedings{10.2312:stag.20251335,
		booktitle = {Smart Tools and Applications in Graphics - Eurographics Italian Chapter Conference},
		editor = {Comino Trinidad, Marc and Mancinelli, Claudio and Maggioli, Filippo and Romanengo, Chiara and Cabiddu, Daniela and Giorgi, Daniela},
		title = {{GUIDÆTA - A Versatile Interactions Dataset with extensive Context Information and Metadata}},
		author = {Lengauer, Stefan and Götz, Sarah Annabelle von and Hoesch, Marie-Therese and Steinwidder, Florian and Tytarenko, Mariia and Bedek, Michael A. and Schreck, Tobias},
		year = {2025},
		publisher = {The Eurographics Association},
		ISSN = {2617-4855},
		ISBN = {978-3-03868-296-7},
		DOI = {10.2312/stag.20251335}
	}

