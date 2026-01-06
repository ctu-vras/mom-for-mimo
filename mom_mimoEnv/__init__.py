from gymnasium.envs.registration import register

register(id='MIMoTest-v0',
         entry_point='mom_mimoEnv.envs:MIMoTestEnv',
         kwargs={"render_mode": "human",
                 },
         )

register(id='MIMoAdultRetargeting-v0',
         entry_point='mom_mimoEnv.envs:MIMoRetargeting',
         kwargs={"render_mode": "human",
                 },
         )
