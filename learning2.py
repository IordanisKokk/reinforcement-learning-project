from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    # for agent in env.agents:
    #     print(f"{agent} obs: {env.observation_space(agent)}")
    # for agent in env.agents:
    #     print(f"{agent} action_space: {env.action_space(agent)}")

    
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    for agent in env.agents:
        print(rewards[agent])
    
env.close()