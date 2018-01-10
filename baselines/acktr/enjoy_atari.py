from baselines.acktr import acktr_disc
from baselines.acktr import policies
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import argparse
from baselines.common.misc_util import (
    boolean_flag,
)
from gym.monitoring import VideoRecorder

def main():
    args = parse_args()
    
    env = make_atari(args.env)
    env = wrap_deepmind(env)
    
    # setup the model to process actions for one environment and one step at a time
    model = acktr_disc.Model(policies.CnnPolicy, env.observation_space, env.action_space, 1, 1)
    # load the trainable parameters from our trained file
    model.load(args.model_path)
    # keep track of the last 4 frames of observations
    env_width = env.observation_space.shape[0]
    env_height = env.observation_space.shape[1]
    obs_history = np.zeros((1, env_width, env_height, 4), dtype=np.uint8)

    # if we're supposed to show how the model sees the game
    if args.show_observation:
        obs = env.reset()
        import pygame
        from pygame import surfarray
        # the default size is too small, scale it up
        scale_factor = args.scale_factor
        screen = pygame.display.set_mode((env_width*scale_factor, env_height*scale_factor), 0, 8)
        # setup a gray palette
        pygame.display.set_palette(tuple([(i, i, i) for i in range(256)]))
        
    # if we're supposed to record video
    video_path = args.video_path
    if video_path is not None:
        video_recorder = VideoRecorder(
        env, base_path=video_path, enabled=video_path is not None)
        
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            if args.show_observation:
                # use the Kronecker product to scale the array up for display, and also transpose x/y axes because pygame
                # displays as column/row instead of gym's row/column
                transposed = obs_history[0,:,:,-1].transpose((1,0))
                scaled_array = np.uint8(np.kron(transposed, np.ones((scale_factor, scale_factor))))
                surfarray.blit_array(screen, scaled_array)
                pygame.display.flip()
            if video_path is not None:
                video_recorder.capture_frame()
            # add the current observation onto our history list
            obs_history = np.roll(obs_history, shift=-1, axis=3)
            obs_history[:, :, :, -1] = obs[None][:, :, :, 0]
            # get the suggested action for the current observation history
            action, v, _ = model.step(obs_history)
            
            obs, rew, done, info = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
        # if we're taking video, stop it now and clear video path so no more frames are added if we're out of lives or there are no lives in this game
        if video_path is not None and ('ale.lives' not in info or info['ale.lives'] == 0):
            video_path = None
            video_recorder.close()

def parse_args():
    parser = argparse.ArgumentParser("Enjoy model results for Atari games")
    # model's save file is required
    parser.add_argument("--model-path", type=str, default=None, required=True, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="name of the game (needs to be NoFrameskip type)")
    # other config
    boolean_flag(parser, "show-observation", default=False, help="whether or not to show an illustration of what the model sees (helps to debug)")
    parser.add_argument("--scale-factor", type=int, default=2, help="only used when show-observation is set, scales up display size of observation")
    parser.add_argument("--video-path", type=str, default=None, help="supply to record video (e.g. /tmp/breakout will create /tmp/breakout.mp4)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
