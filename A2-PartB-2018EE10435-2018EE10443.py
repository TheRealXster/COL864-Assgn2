# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import random
import matplotlib.pyplot as plt

# %% [markdown]
# ## Plot Function

# %%
def plot(values, policy, goalState, flag, walls, fileName = None, title = None):
    cmap = plt.cm.Blues
    norm = plt.Normalize(np.min(values), np.max(values))
    rgba = cmap(norm(values))

    rgba[12, 48] = 1.0, 0.0, 0.0, 1.0

    for i in range(0, 25):
        for j in range(0, 50):
            if(walls[j][i] == 1):
                rgba[i][j] = 0.0, 0.0, 0.0, 1.0

    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(rgba, interpolation = 'nearest')
    ax.set_ylim(ax.get_ylim()[::-1])
    #print(np.min(values), np.max(values))

    if(flag):
        for i in range(0, 25):
            for j in range(0, 50):
                if(walls[j][i] == 1):
                    continue

                num = int(policy[i][j])
                text = ''
                if(num == 1): text = 'L'
                elif(num == 2): text = 'U'
                elif(num == 3): text = 'R'
                else: text = 'D'
                color = 'red'
                if(i == 12 and j == 48): continue
                text = ax.text(j, i, text, ha = 'center', va = 'center', color = 'black', size = 'small')

    plt.axis('on')

    for i in range(1, 26, 1):
        ax.axhline(i - 0.5, color = 'gray', linewidth = 0.4)
        
    for i in range(1, 51, 1):
        ax.axvline(i - 0.5, color = 'gray', linewidth = 0.4)

    if(title): ax.set_title(title)

    if(fileName):
        plt.savefig(fileName)
        plt.close()
    
    plt.show()

# %% [markdown]
# ## Part a - Implement Q - Learning

# %%
def chooseAction(values, x, y, epsilon):
    num = random.uniform(0.0, 1.0)
    
    if(num <= 1 - epsilon):
        currMax = -np.Inf
        actionTaken = -1
        for action in range(0, 4):
            if(values[x][y][action] > currMax):
                actionTaken = action
                currMax = values[x][y][action]
        
        return actionTaken
    
    else:
        return random.randint(0, 3)


# %%
def takeAction(x, y, action, walls, goalState):
    # 0 is left, 1 is up, 2 is right, 3 is down
    num = random.uniform(0.0, 1.0)

    newStateX = -5
    newStateY = -5

    if(action == 0):
        if(num <= 0.8): newStateX = x - 1
        elif(num <= 0.8 + 0.2 / 3): newStateY = y + 1
        elif(num <= 1.0 - 0.2 / 3): newStateX = x + 1
        else: newStateY = y - 1
    if(action == 1):
        if(num <= 0.8): newStateY = y + 1
        elif(num <= 0.8 + 0.2 / 3): newStateX = x - 1
        elif(num <= 1.0 - 0.2 / 3): newStateX = x + 1
        else: newStateY = y - 1
    if(action == 2):
        if(num <= 0.8): newStateX = x + 1
        elif(num <= 0.8 + 0.2 / 3): newStateY = y + 1
        elif(num <= 1.0 - 0.2 / 3): newStateX = x - 1
        else: newStateY = y - 1
    if(action == 3):
        if(num <= 0.8): newStateY = y - 1
        elif(num <= 0.8 + 0.2 / 3): newStateY = y + 1
        elif(num <= 1.0 - 0.2 / 3): newStateX = x + 1
        else: newStateX = x - 1
    
    if(newStateX == -5): newStateX = x
    if(newStateY == -5): newStateY = y

    if(walls[newStateX][newStateY] == 1):
        return -1, x, y
    if(newStateX == goalState[0] and newStateY == goalState[1]):
        return 100, goalState[0], goalState[1]
    return 0, newStateX, newStateY


# %%
def qLearn(values, walls, goalState, alpha, gamma, epsilon, maxIter, xInit, yInit, rewards, episode):

    xCurr = xInit
    yCurr = yInit
    aCurr = chooseAction(values, xInit, yInit, epsilon)

    iter = 0
    rewardAcc = 0
    while(iter < maxIter):
        if(xCurr == 48 and yCurr == 12): break

        reward, xNew, yNew = takeAction(xCurr, yCurr, aCurr, walls, goalState)
        rewardAcc += reward

        quantity = -np.Inf
        for action in range(0, 4):
            if(values[xNew][yNew][action] > quantity):
                quantity = values[xNew][yNew][action]

        values[xCurr][yCurr][aCurr] = values[xCurr][yCurr][aCurr] + alpha * (reward + gamma * quantity - values[xCurr][yCurr][aCurr])

        xCurr = xNew
        yCurr = yNew
        aCurr = chooseAction(values, xCurr, yCurr, epsilon)
        iter += 1
    
    rewards[episode] = rewardAcc

    return values, rewards

# %%
def plotPolicy(values, walls, fileName = None, title = None):
    policy = np.zeros((50, 25))
    toPlot = np.zeros((50, 25))

    for i in range(0, 50):
        for j in range(0, 25):
            if(walls[i][j] == 1): continue
            if(i == 48 and j == 12): continue
            
            currMax = -np.Inf
            actionTaken = -1
            for action in range(0, 4):
                if(values[i][j][action] > currMax):
                    actionTaken = action + 1
                    currMax = values[i][j][action]
            
            policy[i][j] = actionTaken
            toPlot[i][j] = currMax
    #print(title)
    if(fileName and title):
        plot(toPlot.transpose(), policy.transpose(), (48, 12), True, walls, fileName, title)
        return
    if(fileName):
        plot(toPlot.transpose(), policy.transpose(), (48, 12), True, walls, fileName)
        return
    if(title):
        plot(toPlot.transpose(), policy.transpose(), (48, 12), True, walls, None, title)
        return
    else:
        plot(toPlot.transpose(), policy.transpose(), (48, 12), True, walls)


# %%
values = np.zeros((50, 25, 4))
rewards = np.zeros((20000))
walls = np.zeros((50, 25))

for i in range(0, 50):
    for j in range(0, 25):
        if(i == 0 or j == 0 or i == 49 or j == 24): walls[i][j] = 1
        if(i == 25 or i == 26):
            if(j <= 11 or j >= 13): walls[i][j] = 1

for i in range(0, 50):
    for j in range(0, 25):
        for k in range(0, 4):
            if(walls[i][j] == 1): break
            if(i == 48 and j == 12): break
            values[i][j][k] = random.uniform(0.0, 1.0)

for episode in range(0, 20000):
    xInit = random.randint(1, 48)
    yInit = random.randint(1, 23)

    if(xInit == 48 and yInit == 12): xInit = random.randint(1, 47)
    if(walls[xInit][yInit] == 1): yInit = 12

    values, rewards = qLearn(values, walls, (48, 12), 0.25, 0.99, 0.05, 1000, xInit, yInit, rewards, episode)
    if(episode == 999 or episode == 3999 or episode == 9999 or episode == 19999):
        plotPolicy(values, walls, None, 'For ' + str(episode + 1) + ' episodes')

# %% [markdown]
# ## Part b - Visualization of state-value pairs and optimal policy




# %%
plotPolicy(values, walls)

# %% [markdown]
# ## Part c - Comparing Q - Learning for different values of epsilon

# %%
#Q-Learning for epsilon = 0.005
values_C1 = np.zeros((50, 25, 4))
rewards_C1 = np.zeros((20000))

for i in range(0, 50):
    for j in range(0, 25):
        if(i == 0 or j == 0 or i == 49 or j == 24): walls[i][j] = 1
        if(i == 25 or i == 26):
            if(j <= 11 or j >= 13): walls[i][j] = 1

for i in range(0, 50):
    for j in range(0, 25):
        for k in range(0, 4):
            if(walls[i][j] == 1): break
            if(i == 48 and j == 12): break
            values_C1[i][j][k] = random.uniform(0.0, 1.0)

for episode in range(0, 20000):
    xInit = random.randint(1, 48)
    yInit = random.randint(1, 23)
    
    if(xInit == 48 and yInit == 12): xInit = random.randint(1, 47)
    if(walls[xInit][yInit] == 1): yInit = 12

    values_C1, rewards_C1 = qLearn(values_C1, walls, (48, 12), 0.25, 0.99, 0.005, 1000, xInit, yInit, rewards_C1, episode)
    if(episode == 999 or episode == 3999 or episode == 9999 or episode == 19999): plotPolicy(values_C1, walls, str(episode) + 'c1', str(episode + 1) + ' Episodes')



#Q-Learning for epsilon = 0.5 ---------------------------------------------------------------------------------------------------------------------
values_C3 = np.zeros((50, 25, 4))
rewards_C3 = np.zeros((20000))

for i in range(0, 50):
    for j in range(0, 25):
        if(i == 0 or j == 0 or i == 49 or j == 24): walls[i][j] = 1
        if(i == 25 or i == 26):
            if(j <= 11 or j >= 13): walls[i][j] = 1

for i in range(0, 50):
    for j in range(0, 25):
        for k in range(0, 4):
            if(walls[i][j] == 1): break
            if(i == 48 and j == 12): break
            values_C3[i][j][k] = random.uniform(0.0, 1.0)

for episode in range(0, 20000):
    xInit = random.randint(1, 48)
    yInit = random.randint(1, 23)
    
    if(xInit == 48 and yInit == 12): xInit = random.randint(1, 47)
    if(walls[xInit][yInit] == 1): yInit = 12

    values_C3, rewards_C3 = qLearn(values_C3, walls, (48, 12), 0.25, 0.99, 0.5, 1000, xInit, yInit, rewards_C3, episode)
    if(episode == 999 or episode == 3999 or episode == 9999 or episode == 19999): plotPolicy(values_C3, walls, str(episode) + 'c2', str(episode + 1) + ' Episodes')

# %% [markdown]
# ## Part d - Plotting reward accumulated per episode

# %%
plt.plot(rewards_C3[0:100])
plt.savefig('more100')


