import numpy as np
from matplotlib import pyplot as plt
from random import random


N = 90000
n_repairs = 150
L = 1.1
M = 1.7
delta_t = 0.01
t = np.arange(0, 10, delta_t)


def model_11():
    T_work = [-np.log(random()) / L for i in range(n_repairs)]
    T_repair = [-np.log(random()) / M for j in range(n_repairs)]

    n_t = []
    T = T_work[0]
    is_on = 1
    step = 0
    for ti in t:
        if ti > T:
            if is_on == 1:
                is_on = 0
                T += T_repair[step]
                step += 1
            else:
                is_on = 1
                T += T_work[step]
        n_t.append(is_on)

    return n_t


def model_21():
    free_rem = 1
    T = [-np.log(random()) / L for i in range(2)]
    working_on = [0, 0]
    is_on = [1, 1]
    queue = []
    n_t = [0] * len(t)
    for step in range(len(t)):
        remove_from_queue = []
        for i in range(len(T)):
            if T[i] < t[step]:
                is_on[i] = 0
                if i not in queue:
                    if working_on[i] == 0:
                        queue.append(i)
                    else:
                        free_rem += 1
                        T[i] += -np.log(random()) / L
                        working_on[i] = 0
                        is_on[i] = 1
        for i in range(len(T)):
            if T[i] < t[step]:
                if i in queue:
                    if queue.index(i) < free_rem:
                        T[i] = t[step] -np.log(random()) / M
                        working_on[i] = 1
                        remove_from_queue.append(i)
        free_rem -= len(remove_from_queue)
        for rem in remove_from_queue:
            queue.remove(rem)
        n_t[step] = 1 if (is_on[0] == 1 or is_on[1] == 1) else 0

    return n_t


def model_22():
    T_work = []
    T_repair = []
    for i in range(2):
        T_work.append([])
        T_repair.append([])
        for j in range(n_repairs):
            T_work[i].append(-np.log(random()) / L)
            T_repair[i].append(-np.log(random()) / M)
    n_t = []
    T1 = T_work[0][0]
    T2 = T_work[1][0]
    T1_is_on = 1
    T2_is_on = 1
    i1 = 0
    i2 = 0
    for ti in t:
        if ti > T1:
            if T1_is_on == 1:
                T1_is_on = 0
                T1 += T_repair[0][i1]
                i1 += 1
            else:
                T1_is_on = 1
                T1 += T_work[0][i1]
        if ti > T2:
            if T2_is_on == 1:
                T2_is_on = 0
                T2 += T_repair[1][i2]
                i2 += 1
            else:
                T2_is_on = 1
                T2 += T_work[1][i2]
        n_t.append(T1_is_on + T2_is_on)

    return [1 if i > 0 else 0 for i in n_t]


def model_42():
    free_rem = 2
    T = [-np.log(random()) / L for i in range(4)]
    working_on = [0, 0, 0, 0]
    is_on = [1, 1, 1, 1]
    queue = []
    n_t = [0]*len(t)
    for step in range(len(t)):
        remove_from_queue = []
        for i in range(len(T)):
            if T[i] < t[step]:
                is_on[i] = 0
                if i not in queue:
                    if working_on[i] == 0:
                        queue.append(i)
                    else:
                        free_rem += 1
                        T[i] += -np.log(random()) / L
                        working_on[i] = 0
                        is_on[i] = 1
                if i in queue:
                    if queue.index(i) < free_rem:
                        T[i] = t[step] -np.log(random()) / M
                        working_on[i] = 1
                        remove_from_queue.append(i)
        free_rem -= len(remove_from_queue)
        for rem in remove_from_queue:
            queue.remove(rem)
        n_t[step] = 1 if (is_on[0] == 1 or is_on[1] == 1) and  (is_on[2] == 1 or is_on[3] == 1) else 0
    return n_t


K_11 = [0]*len(t)
for i in range(N):
    i_exp = model_11()
    K_11 = [K_11[i] + i_exp[i] for i in range(len(K_11))]
K_11 = [K_11_i / N for K_11_i in K_11]
K_11_th = M / (L + M)
plt.subplot(3, 2, 1)
plt.plot(t, K_11, label='Экспериментальная оценка')
plt.plot(t, [K_11_th]*len(t), 'r--', label='Теоретическая оценка')
plt.legend()

K_21 = [0]*len(t)
for i in range(N):
    i_exp = model_21()
    K_21 = [K_21[i] + i_exp[i] for i in range(len(K_21))]
K_21 = [K_21_i / N for K_21_i in K_21]
K_21_th = (2*M*L + M**2) / (2*(L**2) + 2*M*L + M**2)
plt.subplot(3, 2, 3)
plt.plot(t, K_21, label='Экспериментальная оценка')
plt.plot(t, [K_21_th]*len(t), 'r--', label='Теоретическая оценка')
plt.legend()

K_22 = [0]*len(t)
for i in range(N):
    i_exp = model_22()
    K_22 = [K_22[i] + i_exp[i] for i in range(len(K_22))]
K_22 = [K_22_i / N for K_22_i in K_22]
K_22_th = 1 - (1 - M / (M + L))**2
plt.subplot(3, 2, 5)
plt.plot(t, K_22, label='Экспериментальная оценка')
plt.plot(t, [K_22_th]*len(t), 'r--', label='Теоретическая оценка')
plt.legend()

K_42_1 = K_21_th * K_21_th
K_42_2 = K_11_th * K_11_th
K_42_up = K_22_th *  K_22_th
K_42_exp = [0]*len(t)
for i in range(N):
    i_exp = model_42()
    K_42_exp = [K_42_exp[i] + i_exp[i] for i in range(len(K_42_exp))]
K_42_exp = [K_42_exp_i / N for K_42_exp_i in K_42_exp]
plt.subplot(1, 2, 2)
plt.plot(t, [K_42_up]*len(t), label='верхняя граница')
plt.plot(t, [K_42_1]*len(t), label='Нижняя граница, рассчитанная 1 способом')
plt.plot(t, [K_42_2]*len(t), label='Нижняя граница, рассчитанная 2 способом')
plt.plot(t, K_42_exp, label='Экспериментальная оценка')
plt.legend()

plt.show()
