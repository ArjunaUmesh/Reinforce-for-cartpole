import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym


#class for generating the policy network
class PolicyNetwork(nn.Module) : 
    
    def __init__(self,lr,input_dims,n_actions) :
        super(PolicyNetwork,self).__init__()
        self.fc1 = nn.Linear(*input_dims,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
    def forward(self,state) : 
        
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
#The policy gradient agent
#has them memory function to store actions and rewards
#policy netwrok for predicting the action given a state
class PolicyGradientAgent() : 
    
    def __init__(self,lr,input_dims,gamma=0.99,n_actions = 6):
        self.gamma = gamma
        self.n_actions = n_actions
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []
        
        self.policy = PolicyNetwork(self.lr,input_dims,n_actions)
        
    #choose action : 
        
    def choose_action(self,state) : 
        state = torch.tensor([state]).to(self.device)
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = torch.distributions.Categorical(probabilities)
        #SELECTING THE ACTION BASE ON THE DEFINED PROBABILITIES
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        
        self.action_memory.append(log_probs)
        
        #DEREFERENCE THE ACTION FROM A TRNSOR 
        return action.item()
            
            
    def store_rewards(self,reward) : 
        self.reward_memory.append(reward)\
            
                   
    
    #Learn function for the reinforce Monte Carlo algorithm (episodic)
    def learn(self) : 

        self.policy_optimizer.zero_grad()
    
        G = np.zeros_like(self.reward)
        #For each step of episode calculate the returns for each step of the episode
        for t in range(len(self.reward_memory)) : 
            G_sum = 0
            discount = 1
            for k in range(t,len(self.reward_memory)) : 
                G_sum += self.reward_memory[k] * discount 
                discount = discount*self.gamma
            G[t] = G_sum
            
        
        loss = 0
        for g,log_prob in zip(G,self.action_memory) : 
            loss += -g * log_prob
            
        loss.backward()
        self.policy.optimizer.step()
        
        self.action_memory = []
        self.reward_memory = []
        
        
            
            
            
        
        
                
    
    
        
         
        
        
        
        
    

