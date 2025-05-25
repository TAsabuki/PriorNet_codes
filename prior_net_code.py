"""
Copyright 2024 Toshitake Asabuki

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
import pylab as pl
import numba
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from warnings import simplefilter

# Matplotlib settings
simplefilter(action='ignore', category=FutureWarning)
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42

# ----------------------------------------------------------------
# Model Parameters & Initialization
# ----------------------------------------------------------------

np.random.seed()

alpha = 1
theta = 1
beta = 5

@numba.njit(fastmath=True, nogil=True)
def g(x):
    return 1 / (1 + alpha * np.exp(beta * (-x + theta)))


@numba.njit(parallel=True, fastmath=True, nogil=True)
def learning(w, g_V_star, PSP_star, eps, g_V_som, mask):
    for i in numba.prange(len(w[:, 0])):
        for l in numba.prange(len(w[0, :])):
            delta = (-(g_V_star[i]) + g_V_som[i]) * PSP_star[l]
            w[i, l] += eps * delta * beta * (1 - g_V_star[i])
            w[i, l] *= mask[i, l]
    return w


@numba.njit(parallel=True, fastmath=True, nogil=True)
def learning_inh(w, g_V_star, PSP_star, eps, g_V_som, mask):
    for i in numba.prange(len(w[:, 0])):
        for l in numba.prange(len(w[0, :])):
            delta = (-(g_V_star[i]) + g_V_som[i]) * PSP_star[l]
            w[i, l] += eps * delta * beta * (1 - g_V_star[i])
            w[i, l] *= mask[i, l]
            w[i, l] *= (w[i, l] >= 0)
    return w

def calc_cor(x):
    xv = x - x.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    result = np.matmul(xv.transpose(), xv) / np.sqrt(np.outer(xvss, xvss))
    result[np.eye(N, dtype=bool)] = 0
    return result

raeter_size = 0.6

dt = 1
second = 10
test_len = int(20 * 1000 / dt)
plot_len = int(second * 1000 / dt)
N = 500

max_rate = np.ones(N) * 0.05
gain = 1
width = 100

eps_E = 10 ** -4
eps_I = 10 ** -4
msecs_learning = 1000 * 1000
simtime = np.arange(0, msecs_learning, dt)
simtime_len = len(simtime)

tau_m = 15
n_input = 500

pat_color = plt.cm.get_cmap('tab10').colors

W_rec = np.random.randn(N, N) / np.sqrt(N)
W_I = np.random.rand(N, N) / np.sqrt(N)
W_rec_mask = np.ones((N, N))
W_rec_mask[np.eye(N, dtype=bool)] = 0
W_I_mask = np.ones((N, N))
W_I_mask[np.eye(N, dtype=bool)] = 0

w_input = np.random.randn(N, n_input) / np.sqrt(n_input) * 0.1
W_input_mask = np.ones((N, n_input))

I_syn_input = np.random.rand(n_input)
I_syn = np.random.rand(N)
id_rec = np.zeros(N, dtype=bool)

prob_ratio = 2  # the probability ratio

n_pat = 5  # num of stim patterns
pat_list = [0] * prob_ratio + list(range(1, n_pat))
n_pat = len(set(pat_list))
r_input = 0.05
noise_rate = 0.2 * (r_input * 100) / n_input
pat_mat = np.zeros((n_input, n_pat))
for i in range(n_pat):
    pat_mat[i * 100:(i + 1) * 100, i] = r_input

r_input_vec = np.zeros(n_input)
max_trace = np.zeros(N) + 1
tau_max = 10 * 1000

# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------

pat_start = 0
prob_stim = np.zeros(n_pat)
pat_start_list = [[] for _ in range(n_pat)]

for i in tqdm(range(simtime_len), desc="[training]"):
    if i == pat_start:
        count = 0
        pattern_id = pat_list[np.random.randint(len(pat_list))]
        pat_start_list[pattern_id].append(i)
        r_input_vec = pat_mat[:, pattern_id]
        pat_start += 2 * width
        prob_stim[pattern_id] += 1

    if count == width:
        r_input_vec = np.zeros(n_input) + noise_rate

    id_input = (np.random.rand(n_input) < r_input_vec)
    I_syn_input = (1.0 - dt / tau_m) * I_syn_input
    I_syn_input += id_input
    I_syn = (1.0 - dt / tau_m) * I_syn
    I_syn += id_rec

    M_term = np.dot(W_I, I_syn)
    input_term = np.dot(w_input, I_syn_input)
    rec_term = np.dot(W_rec, I_syn)
    x = input_term + rec_term - M_term

    max_trace = (1.0 - dt / tau_max) * max_trace
    max_trace[max_trace < x * gain] = gain * x[max_trace < x * gain]
    y = x / max_trace
    f = g(3 * y)

    id_rec = (np.random.rand(N) < f * dt * max_rate)

    W_rec = learning(W_rec, g(rec_term), I_syn, eps_E, f, W_rec_mask)
    w_input = learning(w_input, g(input_term), I_syn_input, eps_E, f, W_input_mask)
    W_I = learning_inh(W_I, g(M_term), I_syn, eps_I, f, W_I_mask)

# ----------------------------------------------------------------
# Evoked Activity
# ----------------------------------------------------------------

id_list_E = np.zeros((test_len, N), dtype=bool)
id_list_input = np.zeros((test_len, n_input), dtype=bool)
pat_start = 0
pat_start_list = [[] for _ in range(n_pat)]
count = 0
f_list_E = np.zeros((N, test_len))
max_trace_list_evk = np.zeros(test_len)

for i in tqdm(range(test_len), desc="[evoked]"):
    if i == pat_start:
        count = 0
        pattern_id = pat_list[np.random.randint(len(pat_list))]
        r_input_vec = pat_mat[:, pattern_id]
        pat_start_list[pattern_id].append(i)
        pat_start += 2 * width

    if count == width:
        r_input_vec = np.zeros(n_input) + noise_rate

    id_input = (np.random.rand(n_input) < r_input_vec)
    id_list_input[i, :] = id_input
    I_syn_input = (1.0 - dt / tau_m) * I_syn_input
    I_syn_input += id_input
    I_syn = (1.0 - dt / tau_m) * I_syn
    I_syn += id_rec

    M_term = np.dot(W_I, I_syn)
    input_term = np.dot(w_input, I_syn_input)
    rec_term = np.dot(W_rec, I_syn)
    x = input_term + rec_term - M_term

    max_trace = (1.0 - dt / tau_max) * max_trace
    max_trace[max_trace < x * gain] = gain * x[max_trace < x * gain]
    max_trace_list_evk[i] = np.mean(max_trace)
    y = x / max_trace
    f = g(3 * y)

    f_list_E[:, i] = f[0:N]
    id_rec = (np.random.rand(N) < f * dt * max_rate)
    id_list_E[i, :] = id_rec[0:N]

    count += 1

# ----------------------------------------------------------------
# Pattern Sorting/Grouping
# ----------------------------------------------------------------

targets = np.zeros((n_pat, test_len))
for mm in range(n_pat):
    for nn in pat_start_list[mm]:
        targets[mm, nn:min(nn + width, test_len)] = 1

groups_E = [[] for _ in range(n_pat + 1)]

for i in range(N):
    correlations = np.zeros(n_pat)
    for j in range(n_pat):
        correlations[j] = np.corrcoef(f_list_E[i, :], targets[j, :])[0][1]
    if np.max(correlations) > -1:
        groups_E[np.argmax(correlations)].append(i)
    else:
        groups_E[-1].append(i)

index_E = []
for s in groups_E:
    index_E.extend(s)

id_sorted_E_evk = np.zeros((plot_len, N), dtype=bool)
for i in range(N):
    id_sorted_E_evk[:, i] = id_list_E[0:plot_len, int(index_E[i])]

assembly_size = np.zeros(n_pat)
for i in range(n_pat):
    assembly_size[i] = len(groups_E[i])

# ----------------------------------------------------------------
# Evoked Raster Plot
# ----------------------------------------------------------------

tspk, nspk = pl.nonzero(id_sorted_E_evk[0:4000, 0:int(np.sum(assembly_size))] == 1)
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(tspk, nspk, c='k', s=raeter_size, linewidth=0)
for i in range(n_pat):
    for j in pat_start_list[i]:
        if j < 4000:
            ax.axvspan(j, min(j + width, 4000), facecolor=pat_color[i], alpha=0.3, linewidth=0)
ax.set_ylabel("Neuron id", fontsize=11)
ax.set_xlabel("Time [ms]", fontsize=11)
ax.set_xlim([0, 4000])
ax.tick_params(direction='out')
fig.subplots_adjust(bottom=0.25, left=0.15)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('raster_evoked.pdf', format='pdf', dpi=100)

# ----------------------------------------------------------------
# Spontaneous Activity
# ----------------------------------------------------------------

plot_len = 1000 * 1000
f_list = np.zeros((N, plot_len))
id_list_E = np.zeros((plot_len, N), dtype=bool)

for i in tqdm(range(plot_len), desc="[spontaneous]"):
    I_syn = (1.0 - dt / tau_m) * I_syn
    I_syn += id_rec
    M_term = np.dot(W_I, I_syn)
    rec_term = np.dot(W_rec, I_syn)
    x = rec_term - M_term
    max_trace = (1.0 - dt / tau_max) * max_trace
    max_trace[max_trace < x * gain] = gain * x[max_trace < x * gain]
    y = x / max_trace
    f = g(3 * y)
    f_list[:, i] = f
    id_rec = (np.random.rand(N) < f * dt * max_rate)
    id_list_E[i, :] = id_rec[0:N]

plot_len = int(200 * 1000 / dt)
id_list_E = np.zeros((plot_len, N), dtype=bool)
f_list = np.zeros((N, plot_len))
V_list = np.zeros((N, plot_len))

for i in tqdm(range(plot_len), desc="[spontaneous]"):
    I_syn = (1.0 - dt / tau_m) * I_syn
    I_syn += id_rec
    M_term = np.dot(W_I, I_syn)
    rec_term = np.dot(W_rec, I_syn)
    x = rec_term - M_term
    max_trace = (1.0 - dt / tau_max) * max_trace
    max_trace[max_trace < x * gain] = gain * x[max_trace < x * gain]
    y = x / max_trace
    f = g(3 * y)
    f_list[:, i] = f
    V_list[:, i] = y
    id_rec = (np.random.rand(N) < f * dt * max_rate)
    id_list_E[i, :] = id_rec[0:N]

f_sorted = np.zeros((N, plot_len))
V_sorted = np.zeros((N, plot_len))
for i in range(N):
    f_sorted[i, :] = f_list[int(index_E[i]), :]
    V_sorted[i, :] = V_list[int(index_E[i]), :]

mean_firing_rate = np.zeros((n_pat, plot_len))
count = 0
for i in range(n_pat):
    if int(assembly_size[i]) != 0:
        mean_firing_rate[i, :] = np.mean(f_sorted[count:count + int(assembly_size[i]), :], axis=0)
    count += int(assembly_size[i])

cor_mat = calc_cor(f_sorted.T)

fig, ax = plt.subplots(figsize=(4, 3))
cax = ax.imshow(cor_mat, interpolation='nearest', aspect="auto", cmap='jet')
fig.colorbar(cax, orientation='vertical')
ax.set_xlabel("Neuron id", fontsize=10)
ax.set_ylabel("Neuron id", fontsize=10)
ax.tick_params(length=1.3, width=0.05, labelsize=11)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
fig.subplots_adjust(left=0.15, bottom=0.25, right=0.8)
plt.savefig('correlation_spontaneous.pdf', format='pdf', dpi=350)

id_sorted_E = np.zeros((plot_len, N), dtype=bool)
for i in range(N):
    id_sorted_E[:, i] = id_list_E[:, int(index_E[i])]

mean_firing_rate = np.zeros((n_pat, plot_len))
spike_count = np.zeros(n_pat)
mean_V = np.zeros((n_pat, plot_len))
count = 0
for i in range(n_pat):
    if int(assembly_size[i]) != 0:
        mean_firing_rate[i, :] = np.mean(f_sorted[count:count + int(assembly_size[i]), :], axis=0)
        mean_V[i, :] = np.mean(V_sorted[count:count + int(assembly_size[i]), :], axis=0)
        spike_count[i] = np.sum(id_sorted_E[:, count:count + int(assembly_size[i])])
    count += int(assembly_size[i])

fig, ax = plt.subplots(figsize=(3, 3))
LABEL = [n + 1 for n in range(n_pat)]
ax.bar([n + 1 for n in range(n_pat)], spike_count / assembly_size, tick_label=LABEL)
ax.tick_params(direction='out')
fig.subplots_adjust(bottom=0.25, left=0.12)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('spike_count_mean.pdf', dpi=350)

fig, ax = plt.subplots(figsize=(3, 3))
ax.bar([n + 1 for n in range(n_pat)], spike_count, tick_label=LABEL)
ax.tick_params(direction='out')
fig.subplots_adjust(bottom=0.25, left=0.12)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('spike_count_total.pdf', dpi=350)

fig, ax = plt.subplots(figsize=(3, 3))
ax.bar([n for n in range(n_pat)], assembly_size, tick_label=[n + 1 for n in range(n_pat)])
ax.tick_params(direction='out')
fig.subplots_adjust(bottom=0.25, left=0.12)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('assembly_size.pdf', dpi=350)

plot_len2 = 20 * 1000
tspk, nspk = pl.nonzero(id_sorted_E[0:plot_len, 0:int(np.sum(assembly_size))] == 1)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(tspk, nspk, c='k', s=raeter_size, linewidth=0)
ax.set_ylabel("Neuron id", fontsize=11)
ax.set_xlabel("Time [ms]", fontsize=11)
ax.set_xlim([0, plot_len2])
ax.tick_params(direction='out')
fig.subplots_adjust(bottom=0.25, left=0.15)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('raster_spont.pdf', dpi=350)

plt.close('all')
