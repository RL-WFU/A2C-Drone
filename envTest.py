from env import *

env = Env()

actions = []
for l in range(200):
    actions.append(0)
    actions.append(4)
    actions.append(1)

for a in actions:
    classifiedImage, next_row, next_col, reward = env.step(action=a)
    print("Action:", a)
    print("New Position:", next_row, ",", next_col)
    print("Reward:", reward)

    print("==========================")




