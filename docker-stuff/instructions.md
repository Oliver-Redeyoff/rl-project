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
In future I want to fix this but I am lazy

## On the cluster (weatherwax.cs.bath.ac.uk)

### Building
Build using `hare build -t BUCS_USERNAME/openai-baseline`

### Running
Run using `hare run -it --rm BUCS_USERNAME/openai-baseline`
This should run the `CMD` command and attach the container to your terminal
