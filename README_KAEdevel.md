# Introduction to my Udacity Navigation Project
This readme attempts to describe what you need to run my solution in ADDITION / SUPPLIMENTAL to the basic Udacity 1st Project for the Reinforcement Learning class Navigation project P1_Naviation readme information.

Briefly the project uses the Unity (MS Visual Studios) pre-defined environment (Bananas.exe) which is a simple game to get the yellow bananas for a reward of +1, avoid the blue bananas for a reward of -1 based on actions (forward, backward, left, right) and forward perception for a total of 37 states. The further details of this project is contained in this directory in the readme1st.md file, which was the Udacity original readme file for this project, which I renamed to avoid conflict/confusion.

I've also included the Udacity .gitignore, CODEOWNERS, and LICENCE files for the entire Reinforcement Learning class as well as the class readme file, renamed to readme2nd.md again to avoid conflicts with this file (project) readme file.

# Brief description of setting up the environment
This development was performed in Windows 64bit environment, so if you have a different computer environment you may need different instructions

Although details of setting up this environment can be found in the Readme1st and Readme2nd(Dependencies section), briefly it involves:

1) downloading the Banana .zip file containing the self-contained unity environment
2) put the resulting directory in the p1_navigation folder; we further placed the Banana.exe file in the p1_navigation
3) we also followed the udacity README.md file concerning the setup of the (CONDA) environment for the dqn alone:
	a) conda create --name drlnd python=3.6 
	b) activate drlnd
	c) use the drlnd kernel in the Jupyter notebook when running this project
4) We installed MS Visual Studios 2017 & 2022. We did not find the "Build Tools for Visual Studio 2019" on the link provided (https://visualstudio.microsoft.com/downloads/) but rather mostly VS 2022 (VS_Community.exe) and some other things. We selected Python and Unity aspects of the download to hopefully cover our bases there.
5) Clone the repository locally (if you haven't already; I had) and pip install the python requirements (last line, c)):
	a)git clone https://github.com/udacity/deep-reinforcement-learning.git
	b) cd deep-reinforcement-learning/python
	c) pip install .
6) pip install unityagents
	a) which may require downloading the unity app for personal individual use: https://store.unity.com/front-page?check_logged_in=1#plans-individual


