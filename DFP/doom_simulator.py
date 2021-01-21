'''
ViZDoom wrapper
'''
from __future__ import print_function
import sys
import os

vizdoom_path = '../../../../toolboxes/ViZDoom_2017_03_31'
sys.path = [os.path.join(vizdoom_path,'bin/python3')] + sys.path

import vizdoom 
print(vizdoom.__file__)
import random
import time
import numpy as np
import re
import cv2

def get_random_color():
    return np.random.randint(0, 255, 3, dtype=np.int32)



name_to_color_map = dict({0: [0, 0, 0]})
id_to_color_map = dict({0: [128, 128, 128]})

ammo_color = [0, 0, 255]
medikit_color = [0, 255, 0]
armor_color = [0, 128, 0]
monstor_color = [128, 128, 128]
poison_color = [128, 0, 0]
misc_color = [0, 0, 128]

random_monster_color = lambda: [randint(100, 255), 0, randint(0, 40)]

name_to_color_map['CustomMedikit'] = medikit_color
name_to_color_map['DoomImp'] = monstor_color
name_to_color_map['Clip'] = ammo_color
name_to_color_map['Poison'] = poison_color

wall_color = [128, 40, 40]
floor_color = [40, 40, 128]
wall_id = 0
floor_id = 1

class DoomSimulator:
    
    def __init__(self, args):
        self.config = args['config']
        self.resolution = args['resolution']
        self.frame_skip = args['frame_skip']
        self.color_mode = args['color_mode']
        self.switch_maps = args['switch_maps']
        self.maps = args['maps']
        self.game_args = args['game_args']
        
        self._game = vizdoom.DoomGame()
        self._game.set_vizdoom_path(os.path.join(vizdoom_path,'bin/vizdoom'))
        self._game.set_doom_game_path(os.path.join(vizdoom_path,'bin/freedoom2.wad'))
        self._game.load_config(self.config)
        self._game.set_labels_buffer_enabled(True)
        self._game.add_game_args(self.game_args)
        self.curr_map = 0
        self._game.set_doom_map(self.maps[self.curr_map])

        if 'navigation' in self.config:
            self.object_names = ['CustomMedikit', 'Poison']
        else:
            self.object_names = ['Clip', 'CustomMedikit', "DoomImp"]


        
        # set resolution
        try:
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_%dX%d' % self.resolution))
            self.resize = False
        except:
            print("Requested resolution not supported:", sys.exc_info()[0], ". Setting to 160x120 and resizing")
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_160X120'))
            self.resize = True

        # set color mode
        if self.color_mode == 'RGB':
            self._game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
            self.num_channels = 3
        elif self.color_mode == 'GRAY':
            self._game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
            self.num_channels = 1
        else:
            print("Unknown color mode")
            raise

        self.available_controls, self.continuous_controls, self.discrete_controls = self.analyze_controls(self.config)
        self.num_buttons = self._game.get_available_buttons_size()
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        self.num_meas = self._game.get_available_game_variables_size()
            
        self.meas_tags = []
        for nm in range(self.num_meas):
            self.meas_tags.append('meas' + str(nm))
            
        self.episode_count = 0
        self.game_initialized = False

    def color_labels(labels):
        """
        Walls are blue, floor/ceiling are red (OpenCV uses BGR).
        """
        tmp = np.stack([labels] * 3, -1)
        tmp[labels == 0] = [255, 0, 0]
        tmp[labels == 1] = [0, 0, 255]

        return tmp
        
    def analyze_controls(self, config_file):
        with open(config_file, 'r') as myfile:
            config = myfile.read()
        m = re.search('available_buttons[\s]*\=[\s]*\{([^\}]*)\}', config)
        avail_controls = m.group(1).split()
        cont_controls = np.array([bool(re.match('.*_DELTA', c)) for c in avail_controls])
        discr_controls = np.invert(cont_controls)
        return avail_controls, np.squeeze(np.nonzero(cont_controls)), np.squeeze(np.nonzero(discr_controls))
        
    def init_game(self):
        if not self.game_initialized:
            self._game.init()
            self.game_initialized = True
            
    def close_game(self):
        if self.game_initialized:
            self._game.close()
            self.game_initialized = False

    def transform_labels(self, labels,
                     buffer):
        rgb_buffer = np.stack([buffer] * 3, axis=2)
        mask = np.zeros((rgb_buffer.shape[0], rgb_buffer.shape[1], len(self.object_names)+3))

        # Walls and floor
        rgb_buffer[buffer == wall_id] = wall_color
        rgb_buffer[buffer == floor_id] = floor_color
        mask[buffer==wall_id, 0] = 1.
        mask[buffer==floor_id, 1] = 1.

        for l in labels:
            name = l.object_name
            if name not in self.object_names:
                mask[buffer == l.value, 2] = 1.
                rgb_buffer[buffer == l.value, :] = misc_color
            else:
                color = name_to_color_map[name]
                rgb_buffer[buffer == l.value, :] = color
                mask[buffer == l.value, 3+self.object_names.index(name)] = 1.
        return rgb_buffer, mask
            
    def step(self, action=0):
        """
        Action can be either the number of action or the actual list defining the action
        
        Args:
            action - action encoded either as an int (index of the action) or as a bool vector
        Returns:
            img  - image after the step
            meas - numpy array of returned additional measurements (e.g. health, ammo) after the step
            rwrd - reward after the step
            term - if the state after the step is terminal
        """
        self.init_game()
        
        rwrd = self._game.make_action(action, self.frame_skip)        
        state = self._game.get_state()
        if 'navigation' in self.config:
            obj_labels = np.zeros(2)
        else:
            obj_labels = np.zeros(3)
        if state is None:
            img = None
            meas = None
            seg = None
        else:
            labels = state.labels
            for l in labels:
                if 'navigation' in self.config:
                    if l.object_name == 'CustomMedikit':
                        obj_labels[0] = 1.0
                    if l.object_name == 'Poison':
                        obj_labels[1] = 1.0
                else:
                    if l.object_name == 'Clip':
                        obj_labels[0] = 1.0
                    if l.object_name == 'CustomMedikit':
                        obj_labels[1] = 1.0
                    if l.object_name == "DoomImp":
                        obj_labels[2] = 1.0

            # ViZDoom 1.0
            #raw_img = state.image_buffer
                
            ## ViZDoom 1.1 
            if self.color_mode == 'RGB':
                raw_img = state.screen_buffer
            elif self.color_mode == 'GRAY':
                raw_img = np.expand_dims(state.screen_buffer,0)
                
            if self.resize:
                if self.num_channels == 1:
                    if raw_img is None:
                        img = None
                    else:
                        img = cv2.resize(raw_img[0], (self.resolution[0], self.resolution[1]))[None,:,:]
                else:
                    raise NotImplementedError('not implemented for non-Grayscale images')
            else:
                img = raw_img
                
            meas = state.game_variables # this is a numpy array of game variables specified by the scenario
            buffer = state.labels_buffer
            mask, seg_target = self.transform_labels(labels, buffer)
            mask = cv2.resize(mask, (self.resolution[0], self.resolution[1]))
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)[None, :, :]
            mask = mask.transpose((2, 0, 1))
            seg_target = cv2.resize(seg_target, (self.resolution[0], self.resolution[1]))
            seg_target = seg_target.transpose((2, 0, 1))
            # if obj_labels[2] == 1:
            #     cv2.imwrite('mask.png', mask.transpose((1, 2, 0)))
            #     cv2.imwrite('gray.png', img.transpose((1, 2, 0)))
            
        term = self._game.is_episode_finished() or self._game.is_player_dead()
        
        if term:
            self.new_episode() # in multiplayer multi_simulator takes care of this            
            img = np.zeros((self.num_channels, self.resolution[1], self.resolution[0]), dtype=np.uint8) # should ideally put nan here, but since it's an int...
            meas = np.zeros(self.num_meas, dtype=np.uint32) # should ideally put nan here, but since it's an int...
            seg_target = np.zeros((3+len(self.object_names), self.resolution[1], self.resolution[0]), dtype=np.uint8)
            mask = np.zeros((3, self.resolution[1], self.resolution[0]), dtype=np.uint8)
            gray_mask = np.zeros((1, self.resolution[1], self.resolution[0]), dtype=np.uint8)

        return img, meas, rwrd, term, obj_labels, gray_mask, mask, seg_target
    
    def get_random_action(self):
        return [(random.random() >= .5) for i in range(self.num_buttons)]
        
    def is_new_episode(self):
        return self._game.is_new_episode()
    
    def next_map(self):     
        if self.switch_maps:
            self.curr_map = (self.curr_map+1) % len(self.maps)
            self._game.set_doom_map(self.maps[self.curr_map])
    
    def new_episode(self):
        self.next_map()
        self.episode_count += 1
        self._game.new_episode()
