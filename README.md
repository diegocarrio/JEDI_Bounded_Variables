# Instructions how to compile JEDI/oops in a MacBook Laptop 💻:

One of the easiest way to compile JEDI on a MacBook laptop is using Docker containers, which is a packaged user environment that can be “unpacked” and used across different systems, from laptops to cloud to HPC. From a JEDI perspective, the main purpose of containers is Portability: they provide a uniform computing environment (software tools, libraries, compilers, etc) across different systems. So, users and developers can focus on working with the JEDI code without worrying about whether their version of cmake is up to date or whether or not NetCDF was configured with parallel HDF5 support (for example). Further details can be found following this link:

https://jointcenterforsatellitedataassimilation-jedi-docs.readthedocs-hosted.com/en/latest/using/jedi_environment/containers/container_overview.html#docker

One distinguishing feature of Docker is that it does not rely on the linux user namespaces and other features. So, in short, Docker can run natively on laptops and PCs running macOS or Windows.

These are the steps to follow to compile JEDI/oops:

**Step 1:** docker run -u nonroot --rm -it -v /Users/diegocarrio/Dropbox/JEDI:/home/nonroot/shared jcsda/docker-gnu-openmpi-dev:latest  
**Step 2:** cd shared/  
**Step 3:** git clone -b 1.8.0 https://github.com/jcsda/oops  
**Step 4:** cd oops/  
**Step 5:** mkdir build  
**Step 6:** cd build  
**Step 7:** ecbuild ..  
**Step 8:** make  
