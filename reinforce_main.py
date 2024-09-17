import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from reinforce import PolicyGradientAgent

def plot_learning_curve(scores,x,figure_file ) :
    running_avg = np.zeros(len(scores))
    
    for i in range(len(running_avg)) : 
        running_avg[i] = np.mean(scores[-100:])
        plt.plot(x,running_avg)
        plt.title("Running average of previous 100 scores")
        
        plt.save(figure_file)
        
        
if __name__ == "__main__"  :
    env = gym.make("CartPole-v1",render_mode = "human")
    n_games = 3000
    agent = PolicyGradientAgent(lr = 0.0005, 
                                gamma = 0.99,
                                input_dims=(4,),
                                n_actions = 2)
    
    fname = 'REINFORCE_' + 'Walker2D' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots_reinforce/' + fname + '.png'
    
    best_score = 0
    
    #agent.policy.load_state_dict(torch.load('Plots/saved'))
    scores = []
    for i in range(n_games) : 
        terminal = False
        state,info = env.reset()
        env.render()
        score = 0
        
        while not terminal : 
            action = agent.choose_action(state)
            #print(action)
            next_state,reward,terminal,truncated,info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            state = next_state

        agent.learn()
        
        if(score>best_score):
            best_score = score
            print("Saving model")
            #torch.save(agent.policy.state_dict(),'Plots/saved')
            
        
        scores.append(score)
            
        avg_score = np.mean(scores[-100:])
        print("Episode : ",i," Score : ",score," Average score : ",avg_score)
        

    x = [i+1 for i in range(len(score))] 
    
    plot_learning_curve(scores = scores, x=x, figure_file=figure_file)
        