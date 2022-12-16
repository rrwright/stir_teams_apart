"""
Assumptions:
- users may enter several entries. Only the last one will be counted.

"""


import numpy as np
# from numpy import *
# from pylab import *
import scipy as sp
from scipy.optimize import fmin_cg
from scipy.io import loadmat
import random, sys, csv
from time import time



#--- Collaborative Filter recommender
def cofi_cost_function(params, Y, R, num_users, num_items, num_features, LAMBDA):
	# Unfold the features (X) and weights (Theta) as matrices from params
	X = params[0:num_items*num_features].reshape(num_items, num_features)
	Theta = params[num_items*num_features:].reshape(num_users, num_features)
	
	hx = Theta.dot(X.T)  # users x items
	diff = hx.T - Y  # items x users
	squErr = diff ** 2
	ratedError = squErr * R
	J = 0.5 * sum(sum(ratedError))
	
	# Regularization
	J += (float(LAMBDA) / 2) * (sum(sum(Theta ** 2)) + sum(sum(X ** 2)))  # J += (float(LAMBDA) / 2) * sum(X ** 2)
	
	return J

def cofi_grad_function(params, Y, R, num_users, num_items, num_features, LAMBDA):
	X = params[0:num_items*num_features].reshape(num_items, num_features)
	Theta = params[num_items*num_features:].reshape(num_users, num_features)
	
	hx = Theta.dot(X.T)  # users x items
	diff = hx.T - Y  # items x users
	
	X_grad = (diff * R).dot(Theta) + LAMBDA * X
	Theta_grad = (diff * R).T.dot(X) + LAMBDA * Theta
	
	grad = np.hstack((X_grad.ravel(), Theta_grad.ravel())).T
	return grad

def normalize_ratings(Y, R):
	Y_mean = np.mean(Y, 1).reshape((Y.shape[0],1))
	Y_norm = Y - Y_mean * R
	return (Y_norm, Y_mean)

def rand_2d_array(height, width):
	M = np.zeros(height * width)
	for i in range(height * width):
		M[i] = random.gauss(0, 1)
	M = M.reshape(height, width)
	return M

class ProgressCounter:
	def __init__(self, Y, R, num_users, num_items, num_features, LAMBDA, max_iter):
		self.i, self.Y, self.R, self.num_users, self.num_items = 0, Y, R, num_users, num_items
		self.num_features, self.LAMBDA, self.max_iter, self.cost_log = num_features, LAMBDA, max_iter, []
	def show_progress(self, xk):
		self.i += 1
		cost = cofi_cost_function(xk, self.Y, self.R, self.num_users, self.num_items, self.num_features, self.LAMBDA)
		self.cost_log.append(cost)
		sys.stdout.write("\rIter:%i   %.2f%% complete.  Cost: %f" % (self.i, (float(self.i) / float(self.max_iter) * 100), cost))
		sys.stdout.flush()
		return

def train_cofi(Y, R, num_features=10, LAMBDA=10.0, max_iter=100):
	num_items, num_users = Y.shape
	X = rand_2d_array(num_items, num_features)
	Theta = rand_2d_array(num_users, num_features);
	
	X_raveled = np.hstack((X.ravel(), Theta.ravel())).T
	
	print("Minimizing cost function...")
	progress_counter = ProgressCounter(Y, R, num_users, num_items, num_features, LAMBDA, max_iter)
	theta = fmin_cg(f=cofi_cost_function,
					fprime=cofi_grad_function,
					x0=X_raveled,
					args=(Y, R, num_users, num_items, num_features, LAMBDA),
					callback=progress_counter.show_progress,
					maxiter=max_iter,
					gtol=0.000000001)
	
	X = theta[0:num_items*num_features].reshape(num_items, num_features)
	Theta = theta[num_items*num_features:].reshape(num_users, num_features)
	
	print("Colaborative filtering recommender system training complete.\n")
	return (X, Theta, progress_counter.cost_log)




#--- AI search functions
def all_unclaimed(results, claimed_users):
	for i in results:
		if i in claimed_users:
			return False
	
	return True

def skip_claimed(idx_set, results, claimed_users):
	from copy import deepcopy
	new_idx = deepcopy(idx_set)
	for idx, val in enumerate(results):
		if val in claimed_users:
			new_idx[idx] += 1
	
	return new_idx

def round_complete(idx_set, winning_teams, winners, user_biases, claimed_users):
	results = [winning_teams[i, val] for i, val in enumerate(idx_set)]
	if all_unclaimed(results, claimed_users):
		if len(set(results)) == len(results):
			return idx_set
	
		else:
			new_idx = update_idx_set(idx_set, winning_teams, winners, user_biases)
			return round_complete(new_idx, winning_teams, winners, user_biases, claimed_users)
	
	else:
		new_idx = skip_claimed(idx_set, results, claimed_users)
		return round_complete(new_idx, winning_teams, winners, user_biases, claimed_users)

def update_idx_set(idx_set, winning_teams, winners, user_biases):
	results = [winning_teams[i, val] for i, val in enumerate(idx_set)]
	from copy import deepcopy
	new_idx = deepcopy(idx_set)
	for i in range(winning_teams.shape[1]):
		if results.count(i) > 1:
			competitors = [idx for idx, val in enumerate(results) if val == i]
			remove_winner(competitors, winners, ranked_list(user_biases[:,i]))
			for comp in competitors:
				new_idx[comp] += 1
	
	return new_idx

def remove_winner(competitors, winners, user_rank):
	to_remove = 99999999999
	for competitor in competitors:
		if user_rank.index(winners[competitor]) < to_remove:
			to_remove = competitor
	
	competitors.pop(competitors.index(to_remove))



#--- Helper functions
def random_ratings(num_users, num_pitches=20):
	ratings = np.zeros((num_pitches, num_users))
	for user in range(num_users):
		a_range = range(num_pitches)
		random.shuffle(a_range)
		for rating in range(1,6):
			ratings[a_range[rating],user] = rating
	return ratings

def ranked_list(one_d_array):
	from copy import deepcopy
	an_array = deepcopy(one_d_array)
	removed = an_array.min() - 1
	ranked_list = []
	for i in range(an_array.shape[0]):
		idx = an_array.argmax()
		ranked_list.append(idx)
		an_array[idx] = removed
	return ranked_list






#--- Read in CSV data
if __name__ == "__main__":
	csv_file = sys.argv[1]
else:
	csv_file = "sample_data.csv"
csv_data = []
with open(csv_file, "r") as fd:
	reader = csv.DictReader(fd)
	for line in reader:
		csv_data.append(line)

csv_keys = list(csv_data[0].keys())
throw_away = csv_keys.pop(csv_keys.index("Timestamp"))
throw_away = csv_keys.pop(csv_keys.index("Username"))
csv_keys = sorted(csv_keys)

#--- Validate input
# One submission per user
user_data = {}
for row in csv_data:
	user_data[row['Username']] = [int(row[x].strip()) if len(row[x].strip()) > 0 else 0 for x in csv_keys]

user_list = user_data.keys()
user_list = sorted(user_list)


# Exactly one of each rating
offenders = []
for user in user_list:
	highest = max(user_data[user])
	lowest = min(user_data[user])
	for i in range(1,6):
		if user_data[user].count(i) != 1:
			print(user, i)
			offenders.append(user)
			break

presenters = [
			  ]
offenders = offenders + presenters

for offender in offenders:
	throw_away = user_list.pop(user_list.index(offender))

if len(offenders) > 0:
	print("Offenders:", offenders)


#--- Create numeric rating matrix
Y = []
for user in user_list:
	Y.append(user_data[user])



Y = np.asarray(Y).T  # One user per column. One pitch per row.
#Y = random_ratings(10, len(csv_keys))
#R = Y != 0
R = np.ones((Y.shape))

# Presenters can have their own rating hard coded... ?
# Y[0,0] = 10

#--- Make the magic happen
Y_norm, Y_mean = normalize_ratings(Y, R)
LAMBDA = 1.0
max_iter = 5000
num_features = 10

X, Theta, cost_log = train_cofi(Y=Y_norm*10000, R=R, LAMBDA=LAMBDA, max_iter=max_iter, num_features=num_features)

user_biases = X.dot(Theta.T) / 10000


#--- Calculate winners and the user's picked by that winner (based on user's ranking)
num_winners = 7
winners = ranked_list(Y_mean)
winning_teams = []
for i in range(num_winners):
	winning_teams.append(ranked_list(user_biases[winners[i],:]))  # sorts all users according to their bias toward that team


	

winning_teams = np.asarray(winning_teams)
claimed_users = set()
round_idx = [0 for x in range(num_winners)]
final_teams = np.zeros((num_winners, 1))


while len(claimed_users) + num_winners <= Y.shape[1]:
	round_idx = round_complete(round_idx, winning_teams, winners, user_biases, claimed_users)
	claimed_users.update( [winning_teams[i, val] for i, val in enumerate(round_idx)] )
	this_round = np.asarray([winning_teams[i, val] for i, val in enumerate(round_idx)])[:,None]  # slice with none to make it 2-dimensional
	final_teams = np.hstack((final_teams, this_round))

#--- Clean up final teams
final_teams = final_teams[:,1:].astype(int)
final_teams = [list(x) for x in list(final_teams)]

remaining_users = list(set(range(Y.shape[1])).difference(claimed_users))
if len(remaining_users) > 0:
	claimed_winners = set()
	winners_list = winners[:num_winners]
	for i in range(Y.shape[0]):
		for user in remaining_users:
			user_pick = ranked_list(user_biases[:,user])[i]
			if user_pick in winners_list:
				team_idx = winners.index(user_pick)
				final_teams[team_idx].append(user)
				winners_list[team_idx] = None
				throw_away = remaining_users.pop(remaining_users.index(user))

"""
print Y
print winners[:num_winners]
from pprint import pprint
pprint(final_teams)	
print ""
"""

#print csv_keys

for i in range(num_winners):
	team = winners[i]
	print(csv_keys[team])
	for member in final_teams[i]:
		print(user_list[member])
	print("")
