# summer-school-2026-mjlab
RL Locomotion tutorial with Unitree Go2 in The AI for Human–Robot Interaction summer school which will be held at the Loria and Inria Center at the Université de Lorraine, in Nancy (France) from July 6 to 10.

In this tutorial you will train a Unitree Go2 to walk from scratch in a flat terrain. You will configure parts of the algorithm and the enviroment which later you will train to obtain your policy.

The files you have which have an uncompleted part for you to complete are 

- `src/unitree_go2/go2_constants.py` : Used to configure most of the parameters related to the robot.

- `src/unitree_go2_velocity/env_cfg.py`: Everything related to the enviroments are configured (rewards, domain randomization, etc.)

- `src/unitree_go2_velocity/rl_cfg.py`: Everything related to the PPO algorithm is configured (actor, critic, PPO parameters)

## Train the policy
After having completed the scripts, you will be able to launch you training with the following command

```bash
uv run train Mjlab-Velocity-Flat-Unitree-Go2 --env.scene.num-envs 4096 --agent-max-iterations 50_000
```

## Play the policy's latest checkpoint
```bash
uv run play Mjlab-Velocity-Flat-Unitree-Go2 --checkpoint_file <path-to-repository>/summer-school-2026-mjlab-go2/logs/rsl_rl/go2_velocity/<run-name>/model_<latest-number>.pt 
```

## Deployment
After the training is finished the training script will export and save an `<run-name>.onnx` in the `/logs` which will be used for deploying the actor on the real robot. Due to the limitation of time an already trained onnx is being provided. In case you managed to finish training in time you can try your own.

