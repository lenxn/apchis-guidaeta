# GUIDÆTA
## A Versatile Interactions Dataset with extensive Context Information and Metadata

The GUIDÆTA interactions dataset was collected within an online user study in the [A+CHIS](https://apchis.cgv.tugraz.at/) project on Human-Centered Interactive Adaptive Visual Approaches in High-Quality Health Information, conducted between May 12 and June 23, 2025.

Extensive information on scope, data collection, metadata, preliminary analysis can be found in the respective [paper]() ("GUIDÆTA – A Versatile Interactions Dataset with extensive Context Information and Metadata"), published at Smart Tools and Applications in Graphics ([STAG](https://www.stag-conference.org/2025/)) 2025, an annual international conference organized by the Italian Chapter of the Eurographics association.

The raw dataset is publicly available on OSF: [https://osf.io/fhvbm/](https://osf.io/fhvbm/). 

### Usage

This repository contains everything necessary for handling the raw dataset. 
I.e., loading, filtering, resampling, analysis and more. 
All of which is implemented in `observations.py`. 
All tools can be used by just importing this file. 
A few selected use cases are given in `main.py`:

	import observations


	# The path pointing to the root of the data
	DATA_ROOT = "../CLT/data"

	# load everything from files and initialize
	sentences, sessions, users, task_answers = observations.load_data(DATA_ROOT)

	# get the cognitive load experienced for task two
	task_two_answers = [ta for ta in task_answers if ta.task_no==2]
	cl = sum([
		sum(ta.cl.scores)/len(ta.cl.scores) for ta in task_two_answers
	])/len(task_two_answers)

	# get all sessions associated with the 3rd task
	s = [s for ta in task_answers for s in ta.sessions if ta.task_no==3]

### Publications

When refering to the dataset please cite:

* S. Lengauer, S.A. von Götz, M.T. Hoesch, F. Steinwidder, M. Tytarenko, M.A. Bedek, T. Schreck, *GUIDÆTA - A Versatile Interactions Dataset with extensive Context Information and Metadata*, 2025.
