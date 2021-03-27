# Introduction


The full (and build) project documentation is available in the ``./docs`` directory.

This documentation gives an overview over my code for the final project in the EPP- and the scientific computing course, University of Bonn, winter term 2020/21.

The emphasis is on four aspects:

- Develop a working, stable and relatively fast python implementation for the adaptive lasso, which is currently not available to the best of my knowledge
- Run a simulation study to benchmark several post-model-selection inference procedures, including the adaptive lasso
- Apply the adaptive lasso to a real-world dataset (LISS) to improve the understanding of vaccination attitudes
- Conduct a small latent profile analysis to identify latent subpopulations (regarding vaccination attitudes) based on various sets of variables from the LISS panel


Some information for building this project:

- For the python part, the project makes use of a conda environment, as specified in the ``environment.yml``. To create an equivalent new conda environment on your local machine, run ``conda env create --name my_local_env_name --file=environment.yml`` in the root of the cloned repository.
- You have to activate the conda environment and run ``conda develop .`` before proceeding.
- In order to replicate all parts of this project, you will need an R interpreter in version 4.0.4, and the *renv* package installed, since it manages the R dependencies needed for this project, which means the ``renv.lock`` records the exact versions of R packages used (``https://rstudio.github.io/renv/articles/renv.html``). Afer cloning the repository, navigate to the root of the project and open the R interpreter in the terminal.
- Then type ``library("renv")`` to load the package (or ``install.packages("renv")`` if not installed yet), and subsequently ``renv::init(bare = TRUE)``, restart the R interpreter and then type ``renv::restore()`` to automatically install the packages declared in ``renv.lock`` into your own private project library.
- You will also need to import the files that are specified within ``./src/original_data/file_list.txt`` into the folder ``./src/original_data/``.
- Since I am following the ``Gaudecker, H. M. V. (2014). Templates for reproducible research projects in economics`` template, I am also making use of the `pytask` build system. After having set up your conda environment and your R environment, you should be able to run `pytask` in a terminal opened in the root of the project.
- If you are just interested in the results, you do not have to bother yourself with pytask. However, if full replication is the goal, running pytask is necessary. In this case, have a look into the file ``pytask.ini`` in the root of the project. You will see there that some of the tasks are skipped / ignored by default, due to their long runtime. If you want to run those, this is possible by commenting out the respective lines.
- To understand what each task is doing, have a look into the relevant section of the documentation.
- Running the tasks (22) that are not ignored by default takes me around 2 minutes (Ubuntu 20.04, CPU: Ryzen 9 5950X with 32 Threads at 3.4GHz Base Clock)
- You might consider setting the ``number_simulations`` variable in ``src/analysis/task_sim_real_data_benchmarking.py`` and ``src/analysis/task_sim_data_benchmarking.py`` to a lower number, if you decide to run those, to save time.
- To make it easier to access the final results, also without running all tasks, I have added the necessary final outputs into the ``./src/final`` folder as well. So if you do not want to run all the tasks and just have a look at the results, then you can just open the respective notebook within ``./src/final``, which by default loads the final outputs from the same folder.