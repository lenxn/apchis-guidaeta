# -*- coding: utf-8 -*-
"""
Example file for showcasing the usage of the framework.

"""
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
