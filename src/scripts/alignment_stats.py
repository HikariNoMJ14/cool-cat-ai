import json
import numpy as np
import matplotlib.pyplot as ptl

scores = json.load(open('../../data/alignment_scores/song_scores-28-01-2022-00-37-29.json'))

mean            = [x['mean_score'] for x in scores.values()]
maxes           = [x['max_score'] for x in scores.values()]
outlier_score_1 = [x['outlier_score_1'] for x in scores.values()]
outlier_score_2 = [x['outlier_score_2'] for x in scores.values()]
outlier_score_3 = [x['outlier_score_3'] for x in scores.values()]

ptl.plot(outlier_score_3)
ptl.show()

