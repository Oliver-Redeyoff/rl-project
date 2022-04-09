# Docker Stuff

## Pre-reqs
1. Clone the repo onto the cluster (or where you want to build it)
    - May be easier to use `scp`
2. Replace their Dockerfile with my one

## Dockerfile
In this directory.
Last line (starting with `CMD`) is the command that will be run
This is annoying as the docker needs to be rebuilt on command change
You can change this to `CMD /bin/bash` in order to get a bash prompt


## On the cluster (weatherwax.cs.bath.ac.uk)

### Building
Build using `hare build -t BUCS_USERNAME/openai-baseline`

### Running
Run using `hare run -it --rm BUCS_USERNAME/openai-baseline`
This should run the `CMD` command and attach the container to your terminal
**REMOVE THE --rm if you want to save the container**

Alternatively you can run it using `hare run -it --rm  -v `pwd`:/scratch --user $(id -u):$(id -g) --workdir=/scratch BUCS_USERNAME/openai-baseline YOUR_COMMAND` 
This mounts the current directory you are in to `/scratch` in the container allowing you to output results to your directory **USE THIS IF YOU WANT TO SAVE LOGS OR THE MODEL OR WHATEVER**
