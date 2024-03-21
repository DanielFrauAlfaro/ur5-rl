/usr/local/lib/python3.10/dist-packages/torch/nn/modules/instancenorm.py:80:
UserWarning: input's size at dim=1 does not match num_features. You can 
silence this warning by not passing in num_features, which is not used 
because affine=False
  warnings.warn(f"input's size at dim={feature_dim} does not match 
num_features. "
Traceback (most recent call last):
  File "/daniel/Desktop/ur5-rl/train.py", line 125, in <module>
    model.learn(total_timesteps=60000, log_interval=5, tb_log_name= "Test", 
callback = [checkpoint_callback], progress_bar = True)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/sac/sac.py", line
307, in learn
    return super().learn(
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/common/off_policy
_algorithm.py", line 328, in learn
    rollout = self.collect_rollouts(
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/common/off_policy
_algorithm.py", line 557, in collect_rollouts
    actions, buffer_actions = self._sample_action(learning_starts, 
action_noise, env.num_envs)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/common/off_policy
_algorithm.py", line 390, in _sample_action
    unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/common/base_class
.py", line 553, in predict
    return self.policy.predict(observation, state, episode_start, 
deterministic)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/common/policies.p
y", line 366, in predict
    actions = self._predict(obs_tensor, deterministic=deterministic)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/sac/policies.py",
line 353, in _predict
    return self.actor(observation, deterministic)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/sac/policies.py",
line 168, in forward
    mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/sac/policies.py",
line 155, in get_action_dist_params
    features = self.extract_features(obs, self.features_extractor)
  File 
"/usr/local/lib/python3.10/dist-packages/stable_baselines3/common/policies.p
y", line 131, in extract_features
    return features_extractor(preprocessed_obs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/daniel/Desktop/ur5-rl/networks_SB.py", line 173, in forward
    image_features_2 = self.image_extractor_2(image_tensor_2)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File 
"/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", 
line 217, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/daniel/Desktop/ur5-rl/networks_SB.py", line 104, in forward
    x = self.conv1(x)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File 
"/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", 
line 217, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py",
line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", 
line 460, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", 
line 456, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Given groups=1, weight of size [16, 2, 3, 3], expected 
input[1, 0, 160, 160] to have 2 channels, but got 0 channels instead
