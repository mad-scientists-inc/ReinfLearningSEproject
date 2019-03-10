#Upper confidencec bound algorith for ad optimization problem, pretty much the same as multi armed bandit problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
'''
CTR means click through rate
Unlilke the others, this is the simulated dataset of the clicks on each ad by a user
'''

# Implementing Upper confidence Bound Algorithm
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d # number of times the ad was selected upto round n
sums_of_rewards = [0] * d # sum of rewards of ad i upto round n
total_reward = 0

for n in range(0, N): # 10000 people (rounds)
	ad = 0
	max_upper_bound = 0
	for i in range (0, d): # 10 ads
		if (numbers_of_selections[i] > 0): # if ad version i was selected at least once
			# we now compute the average reward and upper confidence bound
			average_reward = sums_of_rewards[i] / numbers_of_selections[i]
			# compute delta to be used in calculating UCB
			delta_i = math.sqrt(3/2 * (math.log(n + 1) / numbers_of_selections[i])) # n+1 as n starts with zero here
			# calculating UCB
			upper_bound = average_reward + delta_i
		else:
			upper_bound = 1e400
		if upper_bound > max_upper_bound:
			max_upper_bound = upper_bound
			ad = i
	ads_selected.append(ad)
	numbers_of_selections[ad] = numbers_of_selections[ad] + 1
	reward = dataset.values[n, ad]
	sums_of_rewards[ad] = sums_of_rewards[ad] + reward
	total_reward = total_reward + reward
	
# Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of ad selections")
plt.xlabel("Ads")
plt.ylabel("number of times  each ad was selected")
plt.show()
	
''''
The ads selected vector will result in selecting each of the ads once at first
Then it starts selecting different ones based on the density function of each
Then as the trials go by one by one, the fourth ad (from zero) is being selected more and more
Finally, nearing 10000, only the fourth ad is being selected
Hence it can be concluded that the fourth ad(from zero), that is the fifth ad is the best
It has the best chance of being clicked by the user
'''
	