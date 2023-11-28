# mariobros_v2

The classes to build the agent are defined in `agent_model.py`.
    *  The `Agent::train` method uses `runCounter.txt` to keep track of the number for the next run.
         *  This is used to create a new folder for the checkpoints and name the output files.
    *  The output text and data is saved in `output_text/`
    *  The replay buffer is saved every 200 episodes in `output_replay_memory/`. These files can get pretty big (3GB+)

You can configure a run using `run_agent.ipynb`.
    *  A run can pick up training from a previous run by:
        1. Loading the checkpoints (to configure the model parameters)
        2. Copying the replay memory (so the model remembers and keeps training on moves from the previous run)
        3. Setting the `start_ep` and `start_step` paramaters for `agent.train` to where the last run left off (this affects output files names and the epsilon schedule)
