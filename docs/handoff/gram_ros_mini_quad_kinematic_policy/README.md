# gram-ros_mini Quad Kinematic Policy Template

Copy this folder's logic into:

`gram-ros_mini/src/locomotion/policy`

The actual ROS repo was not present on this machine, so these files are a
drop-in implementation template rather than an already-applied patch.

Runtime files named in the deployment note:

- `observation_builder.py`: replace quad-test obs `[0:3] = zeros` with the
  `QuadKinematicBaseVelocityEstimator`.
- `action_processor.py`: clip 12D raw quad-test actions to `[-1, 1]` before
  applying `target = default_active_position + 0.5 * action`.
- `policy_node.py`: expose/select the new model config.
- `joint_mapping.py`: keep the same 18D publish order and 12D active order.

Config examples:

- `policy_config_quad_forward_kinematic.yaml`
- `policy_config_quad_backward_kinematic.yaml`

Installer:

- `install_into_gram_ros_mini.py`

Run it in dry-run mode first:

```bash
python3 install_into_gram_ros_mini.py /path/to/gram-ros_mini/src/locomotion/policy
```

Apply only after reviewing the copy plan:

```bash
python3 install_into_gram_ros_mini.py /path/to/gram-ros_mini/src/locomotion/policy --apply
```

Model files to copy into `gram-ros_mini/src/locomotion/policy/models/`:

- `model_1798_forward_deploy_kinematic.pt`
- `model_2615_backward_deploy_kinematic.pt`

Do not deploy these models with the old `quad_test` behavior that sets base
linear velocity to zeros.
