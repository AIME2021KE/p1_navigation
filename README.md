# Introduction to my Udacity Navigation Project
This readme describes what you need to run my solution in ADDITION / SUPPLIMENTAL to the basic Udacity 1st Project for the Reinforcement Learning class Navigation project P1_Naviation readme information.

Briefly the project uses the Unity (MS Visual Studios) pre-defined environment (Bananas.exe) which is a simple game to get the yellow bananas for a reward of +1, avoid the blue bananas for a reward of -1 based on actions (forward, backward, left, right) and forward perception for a total of 37 states. The further details of this project is contained in this directory in the readme1st.md file, which was the Udacity original readme file for this project, which I renamed to avoid conflict/confusion. 

# Project environment details 
From the readme1st.md file, repeated here as a condition of the project submission:
"A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
This project is considered complete when the 100 average of the training scores exceed 13"

I've also included the Udacity .gitignore, CODEOWNERS, and LICENCE files for the entire Reinforcement Learning class as well as the class readme file, renamed to readme2nd.md again to avoid conflicts with this file (project) readme file.

# Brief description of setting up the environment
This development was performed in Windows 64bit environment, so if you have a different computer environment you may need (slightly) different instructions, particularly with regards to the Unity Banana.zip file.

Although details of setting up this environment can be found in the Readme1st.md and Readme2nd.md(Dependencies section), briefly it involves:

1) downloading the Banana .zip file containing the self-contained unity environment
2) put the resulting directory in the p1_navigation folder; we further placed the Banana.exe file in the p1_navigation
3) we also followed the udacity README.md file concerning the setup of the (CONDA) environment for the dqn alone:

	a) conda create --name drlnd python=3.6 
	
	b) activate drlnd  
	
	c) use the drlnd kernel in the Jupyter notebook when running this project  
	
4) We installed MS Visual Studios 2017 & 2022. We did not find the "Build Tools for Visual Studio 2019" on the link provided (https://visualstudio.microsoft.com/downloads/) as indicated in the provided instructions, but rather mostly VS 2022 (VS_Community.exe) and some other things. We selected Python and Unity aspects of the download to hopefully cover our bases there and that seemed to work.
5) Clone the repository locally (if you haven't already; I had) and pip install the python requirements (last line, c)):

	a)git clone https://github.com/udacity/deep-reinforcement-learning.git  
	
	b) cd deep-reinforcement-learning/python  
	
	c) pip install .  
	
6) pip install unityagents  

	a) which may require downloading the unity app for personal individual use: https://store.unity.com/front-page?check_logged_in=1#plans-individual

We have provided the Banana.exe and the python directory within the repository for convenience

# My model description
Briefly my model is strongly based on the Udacity DQN (Deep Q-learning Network) mini-project for the AIGym LunarLander-v2 environment, which in turn uses a 3-layer fully connected set of neural network to match the states to the action responses, and a double-Q-learning agent and a large memory buffer. The double-Q-learning agent allows the learning to occur with a fixed set of weights while changing the weights in the other set, and then once the training is done with a batch (=64 in this case) of data, it then updates the fixed weights NN and starts again. This tends to avoid overtuning/overfitting.

I had originally planned to start with this model and then modify as needed, but it turned out that my default settings for this case was able to train quite rapidly in about 440-460 episodes. 

However, I did get rather slowed down/distracted trying to get a reloaded "test" version of the trained dictionary to do as well as the trained version (eventually). Turns out this was not absolutely required but it did confuse me (asked a mentor question asking for help on the topic) as the best I seemed able to do was to essentially take as long again to train.

## PRELIMINARY NOTES:
Ran the random actions provided section of code (now commented out) above (for 10 episodes) and get a score of 0.0. We assume this is normal as the video fails to provide this key element but it "makes sense"

Our expectation based in the instructions is to start with our DQN model we used prior and to adjust to fit into this different (Unity) paradym for starters and to meet the performance requirement as needed. We didn't find this necessary however.

We started with just importing the model and agent from DQN as before. Locally we've renamed the python files dqn_model.py and dqn_agent.py. We found this model to work adequately and so didn't try anything else, but will see about implementing some of the other techniques as well. 

## APPROACH
We started with the default (intial) values for the DQN internal or hyperparameters, expecting to make further refinements and network adjustments as needed to meet the requirements, but none were needed because the defaults trained quickly (< 500 episodes). 

The two included python files are dqn_model.py, which contains the QNetwork implementation (only seen by the agent) from DQN, and dqn_agent.py, which contains the Agent class as well as a supporting ReplayBuffer class to store the experiences in tuples for use by the Qnetworks.

QNetwork is composed of 3 fully connected layers with two NN internal sizes (defaults: fc1_size=64 and fc2_size=64) using RELU activation functions along with an initial state_size and a final action_size to map into the bananas input (state) and output (action) environment. The Qnetwork has an __init__ function to be invoked on class creation and the forward method using the NN's to convert the current state into an action.

Agent: the initial agent solution used in the DQN mini project was used as-is with the following Hyperparamters:
Buffer size: 100,000
batch size: 64
gamma: 0.99 # discount factor for subsequent rewards
tau: 1e-3, # soft update of the target parameters
LR: 5e-4, # learning rate
UPDATE_EVERY: 4, how often to update the network

The Agent class itself is composed of an __init__ fuction for construction, which creates the two qnetworks, one that is local and one that is the target network, along with the optimizer and memory buffer from the ReplayBuffer class to store experiences. 

The Agent step method adds the current experience into the memory buffer, and every UPDATE_EVERY steps stores the experience into memory and exectutes the learn fucntion.

The Agent act method returns actions for a given state given current policy. It does this by evaluating (eval) the local qnetwork, get new actions from the local qnetwork, train the local network, and finally select actions either randomly (if a random toss is bigger than the hyperparameter eps, which for our setup has a start (max), end(min) and a decay (multiplier to determine new eps value) to allow it to start pretty randomly but (slowly) select more from the train policy actions.

The Agent learn method was the one for which we had to provide the appropriate solutions previously with the DQN mini-project. Here we unpack the tuple experiences into states, actions, rewards, next_states, and dones. The next_states are used in the target (NOT local) qnetwork to get the next target actions. These are then detached from the resulting tensor to make a true copy, access Qtable for the next action, and hence the rewards of the target network. The resulting tensor has to be (carefully) unpacked to get it into the correct form to be used in subsequent calculations. We then get the next action results from the local qnetwork and then determine the MSE loss between the target and local network fits. We then zero_grad the optimizer, propagate the loss backwards through the network, and perform a step in the optimizer. Finally a soft update is performed on the target network, using TAU times the local network parameters and (1-TAU) times the target network parameters to update the target network parameters.

As indicated the original DQN agent has a helper class ReplayBuffer, with methods add, to add experiences to the buffer, and sample, to sample experiences from the buffer, and is used extensively in the step method for the Agent class.

Originally we expected to look at some of the post-dqn example approaches, especially the dueling networks and the prioritized experience replay. However since these were mainly modifications of the internal workings of the agents and the like. we felt that it was best to first get the baseline DQN running and then see if there are problems about possibly making these modifications. 

So we start with our original agent and model, which we've imported locally and import the (slightly modified) dqn functionfor the unity setup. 

# Running the model  
To run the code, download my entire repository onto your local drive, which includes the banana.zip file that you'll want to unpack locally, and copy the Banana.exe into the project top folder for the self-contained unity environment. You will probably want to make sure you have a recent version of MS Visual Studios (2017 to 2022 seemed to be OK, I already had 2017 installed and installed 2022) and use your Anaconda powershell to create the drlnd anaconda environment as briefly indicated above and in the Readme1st.md. In Anaconda,  click on the "Applications on " pull-down menu and select your newly created drlnd environment (drlnd) and once that loads then launch the Jupyter notebook from that particular environment. 

Once in the notebook, you'll want to go the the kernel and at the bottom change your kernel from python to your newly created drlnd. At this point you are ready to run the notebook. For reference I still included the initial "train" "test" cases, now commented out, which basically just selected random actions, as this was in the original notebook script. 

The user can select kernel on the top of the notebook and then select restart and run all to run all the cells. First time I found I had to have the section that is commented out, but after that we didn't need to install the pyvirtualdisplay package. 

The way we set the notebook up, the user just needs to run the dqn function having defined the agent, env from unityagents, and the resulting brain_name. When you train set the train_mode=True, and setting eps start, end, and decay values to desired (we used 1.0, 0.01, and 0.995, respectively). I like a lot more feedback so I set the dprint value to only 10 and we captured both the raw scores as well as the 100 average

To demonstrate the trained network, we load the defined dictionary into the local qnetwork, which we've seen how to do previously, critically we also need to eval() the local network and finally reset the environment but now with train_mode=False and finally run dqn function but train_mode=False and set eps to 0.0
