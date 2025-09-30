import numpy as np
from gridworld import example41_env

def iterative_policy_evaluation(env, v0=None, k=1):
    v = {s: 0.0 for s in env.all_states()} if v0 is None else dict(v0)
    for _ in range(k):
        v_new = dict(v)
        for s in env.nonterminal_states():
            total = 0.0
            acts = list(env.actions(s))
            for a in acts:
                ns, r, done = env.step(s,a)
                total += (1.0/len(acts)) * (r + (0.0 if done else env.gamma * v[ns]))
            v_new[s] = total
        v = v_new
    return v

def policy_grid_from_v(env, v):
    arrows = {0:'↑',1:'↓',2:'←',3:'→'}
    pi = env.greedy_policy_from_v(v)
    grid = [['' for _ in range(env.W)] for _ in range(env.H)]
    for r in range(env.H):
        for c in range(env.W):
            s = (r,c)
            grid[r][c] = 'T' if s in env.terminal else arrows[pi[s]]
    return grid

def format_grid(arr):
    return '\n'.join(' '.join(f'{x:>4}' for x in row) for row in arr)

def main():
    env = example41_env(4)
    ks = [0,1,2,3,10]
    v = {s: 0.0 for s in env.all_states()}
    for k in ks:
        if k>0: v = iterative_policy_evaluation(env, v0=v, k=1)
        print(f'k={k}  V:')
        print(env.to_value_grid(v).round(1))
        print('Greedy policy wrt V_k:')
        print(format_grid(policy_grid_from_v(env, v)))
        print('-'*40)


    states = env.nonterminal_states()
    idx = {s:i for i,s in enumerate(states)}
    n = len(states)
    A = np.eye(n); b = -1*np.ones(n)
    for s in states:
        i = idx[s]
        A[i,i] = 1.0
        tot = np.zeros(n)
        for a in env.actions(s):
            ns, _, done = env.step(s,a)
            if not done: tot[idx[ns]] += (1/4)
        A[i,:] -= tot
    vsol = np.linalg.solve(A, b)
    v_inf = {s: vsol[idx[s]] for s in states}
    for t in env.terminal: v_inf[t] = 0.0
    print('k=∞ exact V for random policy:')
    print(env.to_value_grid(v_inf).round(1))
    print('Greedy wrt V_∞:')
    print(format_grid(policy_grid_from_v(env, v_inf)))

if __name__ == '__main__':
    main()
