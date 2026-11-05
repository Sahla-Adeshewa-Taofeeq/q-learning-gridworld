import numpy as np
import random

random.seed(42)
np.random.seed(42)
import matplotlib.pyplot as plt
steps_per_episode = []

num_states = 5
num_actions = 2
goal_state = 4
gamma = 0.9
theta = 1e-6


def step(state, action):
    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(goal_state, state + 1)

    if next_state == goal_state:
        reward = 10
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done

alpha =0.1
gamma = 0.9
epsilon = 0.2
episodes = 100
max_steps = 50

q_table = np.zeros((num_states, num_actions))

for episode in range(episodes):
    state = 0
    done = False
    steps = 0

    while not done and steps < max_steps:

        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, done = step(state, action)

        if done:
            td_target = reward
        else:
            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q

        td_error = td_target - q_table[state, action]

        q_table[state, action] += alpha * td_error

        state = next_state
        steps += 1
    steps_per_episode.append(steps)


print("==========TRAINING RESULTS==========")

print("Q-Table:")
print("Columns: Left (0), Right (1)")
print("Rows: States 0 to 4")
print()
print(np.round(q_table, 2))
print()
print("========TESTING THE AGENT========")
print()
state = 0
route = [state]
done = False
print()
while not done and len(route) <= max_steps:
    action = int(np.argmax(q_table[state]))
    next_state, reward, done = step(state, action)
    route.append(next_state)
    state = next_state
print()
print("Starting position:", 0)
print("Goal position:", goal_state)
print("Learned route:", " ->".join(map(str, route)))

if done:
    print("Result: The agent reached the goal!")
else:
    print("Result: The agent did not reach the goal.")

print("Project completed.")
plt.figure(figsize=(8, 4))
plt.plot(steps_per_episode, color='blue')
plt.xlabel("Episode")
plt.ylabel("Steps to goal")
plt.title("Q-Learning Convergence Curve")
plt.grid(True)
plt.savefig("learning_curve.png")
plt.close()

V = np.zeros(num_states)
gamma, theta = 0.9, 1e-6

while True:
    delta = 0
    for s in range(goal_state):
        values = []
        for a in (0, 1):
            s2, r, done = step(s, a)
            values.append(r if done else r + gamma * V[s2])
        best = max(values)
        delta = max(delta, abs(best - V[s]))
        V[s]= best
    if delta < theta:
        break
print("Value iteration result:", np.round(V, 2))
