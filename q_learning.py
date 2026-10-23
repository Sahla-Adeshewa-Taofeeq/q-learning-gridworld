import numpy as np
import random

num_states = 5
num_actions = 2
goal_state = 4

def step(state, acton):
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

print("==========TRAINING RESULTS==========")

print("Q-Table:")
print("Colums: Left (0), Rght (1)")
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
    next_state, rewards, done = step(state, action)
    route.append(next_state)
    state = next_state
print()
print("Startng position:", 0)
print("Goal position:", goal_state)
print("Learned route:", " ->".join(map(str, route)))

if done:
    print("Result: The agent reached the goal!")
else:
    print("Result: The agent did not reach the goal.")

print("Project completed.")
