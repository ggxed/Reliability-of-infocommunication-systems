import random
import math
import numpy as np
import matplotlib.pyplot as plt

def run_in_period(time, lambda_fail, N, p, del_t):
	tau = []
	lambda_p = []
	tmp = N
	R = []
	R_theor = []
	R_diff = []
	lambda_p_theor = []
	for i in time:
		R_theor.append(float(math.e ** -(lambda_fail[0] * i) * p + math.e ** -(lambda_fail[1] * i) * (1 - p)))
		R_diff.append(-lambda_fail[0] * p * math.e ** -(lambda_fail[0] * i) - (1 - p) * lambda_fail[1] * math.e ** (-lambda_fail[1] * i))  
	for i in range(len(R_theor)):
		lambda_p_theor.append(float(R_diff[i] / R_theor[i]) * -1)
	for i in range(N):
		value = random.random()
		if value < p:
			tau.append(-math.log(random.random()) / lambda_fail[0]) 
		else:
			tau.append(-math.log(random.random()) / lambda_fail[1]) 
	for i in range(len(time) ):
		count1 = 0
		count2 = 0
		for j in range(len(tau)):
			if tau[j] > time[i]:
				count1 += 1
			if tau[j] < time[i] + del_t and tau[j] > time[i]:
				count2 += 1
		R.append(count1 / N)
		lambda_p.append(((count1 - (count1 - count2)) / count1 * (1 / del_t)))
	return lambda_p, R, R_theor, lambda_p_theor

def normal_operation_period(time, lambda_fail, N, p, del_t, k):
	tau = []
	lambda_p = []
	R = []
	R_theor = []
	R_diff = []
	lambda_p_theor = []
	for i in time:
		R_theor.append(math.e ** -(lambda_fail[0] * i) * (math.e ** -(lambda_fail[1] * i)))
		R_diff.append(-(lambda_fail[0] + lambda_fail[1]) * math.e ** (-(lambda_fail[0] + lambda_fail[1]) * i)) 
	for i in range(len(R_theor)):
		lambda_p_theor.append(float(R_diff[i] / R_theor[i]) * -1)
	for i in range(N):
		tmp = []
		for j in range(k):
			tmp.append(-math.log(random.random()) / lambda_fail[j])
		tau.append(min(tmp))
	for i in range(len(time)):
		count1 = 0
		count2 = 0
		for j in range(len(tau)):
			if tau[j] > time[i]:
				count1 += 1
			if tau[j] < time[i] + del_t and tau[j] > time[i]:
				count2 += 1
		R.append(count1 / N)
		lambda_p.append(((count1 - (count1 - count2)) / count1 * (1 / del_t)))
	return lambda_p, R, R_theor, lambda_p_theor

def aging_period(time, lambda_fail, N, p, del_t, k):
	tau = []
	lambda_p = []
	R = []
	R_theor = []
	R_diff = []
	lambda_p_theor = []
	for i in time:
		R_theor.append(math.e ** -(lambda_fail[0] * i) + (math.e ** -(lambda_fail[1] * i)) - ((math.e ** -(lambda_fail[0] * i)) * (math.e ** -(lambda_fail[1] * i))))
		R_diff.append((lambda_fail[0] + lambda_fail[1]) * math.e ** -((lambda_fail[0] + lambda_fail[1]) * i) - lambda_fail[1] * math.e ** -(lambda_fail[1] * i) - lambda_fail[0] * math.e ** -(lambda_fail[0] * i))
	for i in range(len(R_theor)):
		lambda_p_theor.append(float(R_diff[i] / R_theor[i]) * -1)
	for i in range(N):
		tmp = []
		for j in range(k):
			tmp.append(-math.log(random.random()) / lambda_fail[j])
		tau.append(max(tmp))
	for i in range(len(time)):
		count1 = 0
		count2 = 0
		for j in range(len(tau)):
			if tau[j] > time[i]:
				count1 += 1
			if tau[j] < time[i] + del_t and tau[j] > time[i]:
				count2 += 1
		R.append(count1 / N)
		lambda_p.append(((count1 - (count1 - count2)) / count1 * (1 / del_t)))
	return lambda_p, R, R_theor, lambda_p_theor

def Plot(time, practice, theory, label1, label2):
	fig, ax = plt.subplots()
	ax.plot(time, practice,marker = 'o', label = label1)
	ax.plot(time, theory, label = label2)
	plt.legend()
	plt.grid(True)

if __name__ == "__main__":
	N = 500000
	k = 2
	p = 0.6
	lambda_fail = [0.8, 0.9] 
	lambda_fail_theor = 0.5
	nfig = 1
	t = 8
	t_step = t / 50
	del_t = 0.1 * t_step 
	time = np.arange(0, t, t_step)
	lambda_p = [] 
	R_theor = []
	R_pract = []
	lambda_p_theor = []
	lambda_p, R_pract, R_theor, lambda_p_theor = run_in_period(time, lambda_fail, N, p, del_t)
	Plot(time, lambda_p, lambda_p_theor, "practice", "theor")
	Plot(time, R_pract, R_theor, "practice", "theor")
	lambda_p, R_pract, R_theor, lambda_p_theor = normal_operation_period(time, lambda_fail, N, p, del_t, k)
	Plot(time, lambda_p, lambda_p_theor, "practice", "theor")
	Plot(time, R_pract, R_theor, "practice", "theor")
	lambda_p, R_pract, R_theor, lambda_p_theor = aging_period(time, lambda_fail, N, p, del_t, k)
	Plot(time, lambda_p, lambda_p_theor, "practice", "theor")
	Plot(time, R_pract, R_theor, "practice", "theor")
	plt.show()

	