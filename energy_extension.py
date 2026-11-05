import random
import numpy as np

random.seed(42)
np.random.seed(42)

NUM_POS = 5
MAX_ENERGY = 5
FOOD_POS = 4
ACTIONS = ["left", "right", "stay"]

alpha = 0.1       
gamma = 0.9       
epsilon = 0.2    
episodes = 3000
max_steps = 30

Q = np.zeros((NUM_POS, MAX_ENERGY + 1, len(ACTIONS)))

def step(pos, energy, action):
    if action == 0:
        pos = max(0, pos - 1)
    elif action == 1:
        pos = min(NUM_POS - 1, pos + 1)
    energy -= 1
    
    if pos == FOOD_POS:
        energy = min(MAX_ENERGY, energy + 3)
  
    if energy <= 0:
        return pos, 0, -10, True
    reward = -(MAX_ENERGY - energy)
    return pos, energy, reward, False

for episode in range(episodes):
    pos = random.randint(0, NUM_POS - 1)
    energy = random.randint(1, MAX_ENERGY)
    for _ in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = int(np.argmax(Q[pos, energy]))
        new_pos, new_energy, reward, done = step(pos, energy, action)
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(Q[new_pos, new_energy])
        Q[pos, energy, action] += alpha * (target - Q[pos, energy, action])
        pos, energy = new_pos, new_energy
        if done:
            break

print("Learned action for each (position, energy):")
print("energy ->  1       2       3       4       5")
for p in range(NUM_POS):
    row = [ACTIONS[int(np.argmax(Q[p, e]))].ljust(7) for e in range(1, MAX_ENERGY + 1)]
    print(f"pos {p}:   " + " ".join(row))

pos, energy = 0, 5
energies = [energy]
for _ in range(15):
    action = int(np.argmax(Q[pos, energy]))
    pos, energy, reward, done = step(pos, energy, action)
    energies.append(energy)
    if done:
        break
print("Energy over time:", energies)