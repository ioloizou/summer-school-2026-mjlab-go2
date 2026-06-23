# summer-school-2026-mjlab
RL Locomotion tutorial with Unitree Go2 in The AI for Human–Robot Interaction summer school which will be held at the Loria and Inria Center at the Université de Lorraine, in Nancy (France) from July 6 to 10.

In this tutorial you will train a Unitree Go2 to walk from scratch in a flat terrain. You will configure parts of the algorithm and the enviroment which later you will train to obtain your policy.

The files you have which have an uncompleted part for you to complete are 

- `src/unitree_go2/go2_constants.py` : Used to configure most of the parameters related to the robot.

- `src/unitree_go2_velocity/env_cfg.py`: Everything related to the enviroments are configured here (rewards, domain randomization, etc.)

- `src/unitree_go2_velocity/rl_cfg.py`: Everything related to the PPO algorithm are configured here (actor, critic, PPO parameters)

- `src/unitree_go2_velocity/custom_mdp.py`: Hand-written reward/termination math functions (using
  torch directly), as opposed to the pre-built terms imported from mjlab elsewhere.

Each uncompleted part is marked with a `# TODO(Exercise N - ...)` comment, and each
TODO explains what's needed and points to a similar, already-completed piece of code
right next to it to use as a reference.

**Exercises 1-6 and 9** are config-wiring exercises: fill in a missing dict entry or
config object. Each one ends in `raise NotImplementedError(...)` placed directly in the
config-building code, so the code will fail to import until you **replace** that `raise`
with your own implementation. You'll hit exercises 1-6 one at a time, in this order:

1. **Init state** (`go2_constants.py`) — define `INIT_STATE`, the robot's default
   standing pose (position + joint angles) used at the start of every episode.
2. **Contact sensor** (`env_cfg.py`) — define `feet_ground_cfg`, the sensor that
   detects feet touching the ground.
3. **Observation** (`env_cfg.py`) — add a `joint_vel` term to the actor's observations.
4. **Event** (`env_cfg.py`) — add a `push_robot` domain-randomization event.
5. **Reward** (`env_cfg.py`) — add a `track_angular_velocity` reward term.
6. **Curriculum** (`env_cfg.py`) — add a `command_vel` curriculum term that widens the
   commanded velocity range over the course of training.

Once 1-6 are done, the package will still fail to import — `rl_cfg.py` is built right
after `env_cfg.py` whenever the task is registered, so you'll land straight on the last
exercise next:

9. **Network architecture** (`rl_cfg.py`) — choose `hidden_dims` and `activation`,
   shared by both the actor and critic MLPs.

**Exercises 7-8** are different: You will write the actual MDP math in
`custom_mdp.py`, using torch directly on the simulation state, rather than just wiring
together pre-built functions. The setup code (fetching the robot/command/sensor data)
is already written for you — you'll find a few intermediate variables set to `None`
with a comment hint above each one (e.g. `z_error = None  # ...`), replace those with
the real computation, and a final `raise NotImplementedError(...)` will keep failing
loudly until you do. Unlike the other exercises, this `raise` lives inside a function
body, so it won't block the config from importing — it'll only show up once that
function is actually called during training/play. You can tackle them whenever you
like, but exercise 5 and "fell_over" (in the Terminations section of `env_cfg.py`)
already wire them in for you, so once the package imports (exercises 1-6 and 9 done)
the env still won't actually run until both are done:

7. **Angular velocity reward** (`custom_mdp.py`) — implement `track_angular_velocity`.
8. **Fell-over termination** (`custom_mdp.py`) — implement `bad_orientation`.

## Train the policy
After having completed the scripts, you will be able to launch you training with the following command

```bash
uv run train Mjlab-Velocity-Flat-Unitree-Go2 --env.scene.num-envs 2048 --agent-max-iterations 3_000
```

## Play the policy's latest checkpoint
```bash
uv run play Mjlab-Velocity-Flat-Unitree-Go2 --checkpoint_file <path-to-repository>/summer-school-2026-mjlab-go2/logs/rsl_rl/go2_velocity/<run-name>/model_<latest-number>.pt 
```

## Play an already trained policy
Due to time limatation of the tutorial and given many of you may not have a capable gpu. You can visualise the end result with this command with an already trained policy with the same configuration.
```bash
uv run play Mjlab-Velocity-Flat-Unitree-Go2 --checkpoint_file <path-to-repository>/summer-school-2026-mjlab-go2/model_3000.pt 
```