# Using Nvidia Containers on Perceptron
The goal of this document is to give users a quick tutorial demonstrating how 
to take an existing deep learning project and run it on Perceptron in a GPU-
accelerated Docker container.


## TL;DR
* Pull a base image off of Nvidia's container registry. Registry Web UI [here](https://ngc.nvidia.com/catalog/containers)
* Go to the top level directory of your project and create a Dockerfile. Documentation on Dockerfiles [here](https://docs.docker.com/engine/reference/builder/)
* Call `nvidia-docker build .` in your shell. 
* Execute `nvidia-docker run` with the right argument flags. Run command docks [here](https://docs.docker.com/engine/reference/run/#general-form)

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
In the world of containers, the equivalent service is provided through container
registries. The registry that interests us the most is the [Nvidia GPU Cloud Container Registry](https://ngc.nvidia.com/containers) (NCR). 


**NOTE**: You may have to create an NGC account and add your API key to get
pull access from the registry. Also, if you're tech stack is the same as the 
one in the tutorial, you won't have to run any commands in this step.


Nvidia provides all of the "Base images". You can search for the right 
container through the web UI. Since my project is written in Python 3 and 
depends on Tensorflow, I'll be using [this](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
container. Note that there are different versions of similar builds in the 
"Tags" tab on the WebUI. I've gone ahead and pulled the latest version of 
my desired image with the command:


`docker pull nvcr.io/nvidia/tensorflow:19.01-py3`


Some of these base images are quite large and take some time to download, so 
if you want a different image I highly recommend you go pull it now while 
you're reading this. If you'd like to see a list of docker base images that 
have already been pulled to Perceptron, go ahead and run:


`docker image ls` 


Once you have the base image extracted into our local container registry, we're
ready to move on to the next step. In case you were curious, pulling a base image
that has already been downloaded to the machine just updates the image. 

--- 

## Step 2: Configuring the Build Process For Your Project's Container

If you've got this far, you ought to know what base image has the dependencies
you need. If it hasn't been pulled, it will get pulled when your container
gets build for the first time. Now that we have a desired base image for our 
project, we can start specifying our build configuration through a Dockerfile.
A Dockerfile is a text file that lists all the commands in sequential order that 
the Docker daemon must execute to build your container. The complete set of 
Docker build commands is quite extensive, so I've gone ahead and included my
annotated copy of my Dockerfile to cover the basics:

```
# Set Docker Base Image to pull from
FROM nvcr.io/nvidia/tensorflow

# Create working directory for my application
WORKDIR /app

# Copy the contents of the folder containing the Dockerfile into container's 
# working dir 
COPY . /app

# Set an environment variable called CUDA_VISIBLE_DEVICES in the container
ENV CUDA_VISIBLE_DEVICES=PCI_BUS_ID

# Open Port 8000 for Tensorboard
EXPOSE 8000

# Use RUN to execute shell commands that change the state of the container
RUN pip install requirements.txt
```

For your convenience, I've put this Dockerfile in this repository. Here is a
table of useful Dockerfile instructions for quick reference:

| Command 	| Calling Convention                             	        | Purpose                                                                                                                    	|
|---------	|--------------------------------------------------------	|----------------------------------------------------------------------------------------------------------------------------	|
| ARG     	| ARG \<Name\>                                             	| Add a custom argument to pass at build time                                                                                	|
| CMD     	| CMD [\<executable\>, \<arg1\>, \<arg2\>, ... \<argv\>]    | At Runtime, set default behavior to Launch executable with arguments.                                                         |
| COPY    	| COPY \<Source\> \<Container Destination\>              	| Copy files over from host to container                                                                                     	|
| EXPOSE  	| EXPOSE \<Port Number\>                           			| Expose a virtual port from inside the container. To access from outside the Runtime, call `docker run -p \<Port Number\>` 	|
| FROM    	| FROM \<Image Name\>                              			| Set base image                                                                                                             	|
| RUN     	| RUN \<Shell Command\>                            			| Run shell command inside container at build time and commit changes.                                                       	|
| VOLUME  	| VOLUME \<Directory Name\>                        			| Mount Docker volume in directory name                                                                                     	|
| WORKDIR 	| WORKDIR \<Folder Name\>                          			| Make container boot dir                                                                                                   	|
                                                                		
You can find more information about creating Dockerfiles that fit your needs 
in [this](https://docs.docker.com/engine/reference/builder/) part of 
the official documentation.

Once this is set up, you can build the container with the command:
`nvidia-docker build .`

---

## Step 3: Executing your Container in the Runtime

You're finally ready to run the container. This section will be much more 
brief, since I'll just be focusing around specific argument flags that 
need to be passed to the docker runtime. Here is a table of some relevant
argument flags that I suspect will get used quite often on Perceptron:

|                Flag               	|                                       Function                                       	|
|:---------------------------------:	|:------------------------------------------------------------------------------------:	|
|                 -i                	|                                 Attach to /dev/stdin                                 	|
|                 -t                	| Allocate a pseudo tty. Often seen as -it for interactive processes like shells. 	|
|                 -d                	|          When container launches, it is detached from stdin, stdout, stderr.         	|
| --ulimit \<Attribute\> \<Number\> 	| configure container-level usage limits (see /etc/security/limits.conf)               	|
| --rm                              	| delete the container once the process has exited                                     	|
| -p \<Port Number\>                 	| Make port available outside the container.                                        	|

I also highly recommend you check out 
[this](https://docs.docker.com/engine/reference/run/#general-form) section of 
the documentation. Specifically, the `--mount` flag, and the flags pertaining 
to restart policies and resource allocation. By default, the Docker engine is 
extremely conservative with resource allocation for containers.  Deep learning 
applications tend to be much more memory-hungry than your average web app,
so you'll be overriding Nvidia's container runtime defaults quite often. For 
an example runtime command, here's mine:

`nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=0 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000 --rm -it adf7c790e9b1`

Here is what each part of the command means, in order:
1. `nvidia-docker run`: Call Nvidia's docker runtime
2. `-e NVIDIA_VISIBLE_DEVICES=0`: Set container environment variable to only see the GPU in the first PCI Bus.
3. `--shm-size=1g`: Allocate 1GB of RAM for the container to use during execution
4. `--ulimit memlock=-1`: Remove hard limit on maximum virtual address, allowing the container to continuously request more RAM from the host.
5. `--ulimit stack=67108864`: Make the container's maximum stack size larger
6. `-p 8000`: Allow me to access container's port 8000 so I can see Tensorboard in a browser in my host machine. 
7. `--rm`: Delete the container upon exit
8. `-it`: attach the container to /dev/stdin and allocate a ptty to instantiate a shell.
9. `adf7c790e9b1`: The UUID of the container to be launched. 
