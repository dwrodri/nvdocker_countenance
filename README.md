# Using Nvidia Containers on Perceptron
The goal of this document is to give users a quick tutorial demonstrating how 
to take an existing deep learning project and run it on Perceptron in a GPU-
accelerated Docker container.

## Getting Started
Containers are like VMs with a lot less low-level isolation and security 
guarantees. Instead of keeping multiple deep-learning libraries up to date, 
you pull all your dependencies into an "image" which is then executed in 
Nvidia's runtime environment. "Containerizing" an existing project can be 
broken down into three main steps:

1. Find the correct base container on Nvidia's Container Registry
2. Configure the build process for your project
3. Load your image into the Container runtime

As an example, I'll be walking through this process with my own project

## Step 1: Getting Pull Access to NGC Container Registry (NCR)
Many high level programming languages have some library/module index on 
the Internet to facilitate code sharing (Python's PyPI, Ruby's Gem system, etc.). 
In the world of containers, the eqiuvalent service is provided through container
registries. The registry that interests us the most is the [Nvidia GPU Cloud 
Container Registry](https://ngc.nvidia.com/containers) (NCR). 

Unfortunately, to access these containers requires an NGC account.

This is where Nvidia provides all of the "Base images" 
