import os 
import itertools 

# gaits = ('trot', 'walk')
# gaits = ('canter', 'pace')
# gfs = (1.5, 2.0, 2.5)
# dfs = (0.5, 0.6, 0.75)

settings = (
    ('trot', 2.0, 0.5),
    ('walk', 2.0, 0.75),
    ('canter', 2.0, 0.66),
    ('pace', 2.0, 0.5), 
)

if __name__ == "__main__":
    for setting in settings:
        gait, gf, df = setting
        cmd = f"""
            python scripts/enjoy_with_logging.py
            --algo ppo
            --env A1GymEnv-v0 
            -f logs
            --record
            --no-render
            --exp-id 11
            --env-kwargs 
                gait_name:"'{gait}'"
                gait_frequency:{gf}
                duty_factor:{df}
            --stats-dir {gait}-{gf}Hz-{df}df
        """.replace("\n", "").replace("\t", "")
        cmd = " ".join(list(filter(lambda x: len(x) > 0, cmd.split(" "))))
        print(cmd)
        os.system(cmd)