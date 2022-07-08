from turtle import speed
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle

END_DISTANCE = 500.
SPEED_LIMITS = [120.]#KpH
TARGET_SPEED=[10., 30., 50., 70., 100.,120.]#KpH

class ACCEnv(AbstractEnv):

    """
    A highway merge negotiation environment.
    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    def __init__(self, target_speed=-1,speed_limit=120):
        if target_speed == -1:
            self.target_speed = np.random.choice(TARGET_SPEED) / 3.6 #m/s
        else:
            self.target_speed=target_speed/3.6
        self.speed_limit=speed_limit/3.6
        print("Initializing ACC Env with Speed Limit: {}".format(self.speed_limit))
        super().__init__()
        
        self.config.update({
            "speed_limit": self.speed_limit #m/s
        })
        self.config["action"].update({
            "speed_range": [0, self.speed_limit]
        })    
        self.config.update({"max_ep_len":230})


    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "time_reward": -0.001,#-4e-5,#
             "action": {
                "type": "ContinuousAction",
                "lateral": False,
                "speed_range": [0, 30.]
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 2,  # [Hz],
            "speed_limit": 30. #m/s
        })
        cfg["action"].update({
            "speed_range": [0, cfg["speed_limit"]]
        })
        cfg["observation"].update({
            #"features":["presence","x","vx"],
            "vehicles_count": 2,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = self.config["collision_reward"] * self.vehicle.crashed+self.config["time_reward"]#*(self.target_speed+6)#+self.vehicle.speed*0.01#
        #print(abs(self.vehicle.speed-self.target_speed)/self.target_speed)
        #if self.vehicle.crashed or self.vehicle.position[0] > END_DISTANCE:
        if self.vehicle.position[0] >= END_DISTANCE:
            reward = reward + 0.5
            self.success=True
        elif self.steps>self.config["max_ep_len"]:
            reward=reward+(self.vehicle.position[0]-320)*0.001
        #    print('abstad_reward:',(self.vehicle.position[0]-320)*0.001)
        #print(reward)
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.steps>self.config["max_ep_len"] or self.vehicle.position[0] > END_DISTANCE 

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.success=False

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 350]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        speed_limit = self.config["speed_limit"]
        ego_v = speed_limit * (np.random.normal() * 0.3 + 0.7)
        ego_v = np.clip(ego_v, 0., speed_limit)
        #ego_v=speed_limit*np.random.uniform()
        other_v = speed_limit * (np.random.normal() * 0.3 + 0.7)
        other_v = np.clip(other_v, 0., speed_limit)
        #other_v=speed_limit*np.random.uniform()
        #self.target_speed=0 
        
        #min_safe_distanse=speed_limit*speed_limit/14+6
        if ego_v>=other_v:
            # ####
            #min_safe_distanse=np.power(ego_v-other_v,2)/13+np.power(other_v-self.target_speed,2)/15+np.power(ego_v-self.target_speed,2)/15+7
            #### new_env
            #min_safe_distanse=np.power(ego_v-other_v,2)/13+np.clip(other_v-self.target_speed,0,speed_limit)*1.2+np.clip(ego_v-self.target_speed,0,speed_limit)*1.5+8
            #new_new_env
            #min_safe_distanse=np.power(ego_v-other_v,2)/12+np.power(other_v-self.target_speed,2)/14+np.power(ego_v-self.target_speed,2)/15+5
            # env 5
            #min_safe_distanse=np.power(ego_v-other_v,2)/14+np.power(other_v-self.target_speed,2)/14+np.power(ego_v-self.target_speed,2)/14+2
            # env 6
            #min_safe_distanse=np.power(ego_v-other_v,2)/13+np.power(other_v-self.target_speed,2)/13+np.power(ego_v-self.target_speed,2)/13
            # env 7
            #min_safe_distanse=np.power(ego_v-other_v,2)/13+np.power(other_v-self.target_speed,2)/13+np.power(ego_v-self.target_speed,2)/13+5
            #env 8
            #min_safe_distanse=np.power(ego_v-other_v,2)/5+np.power(other_v-self.target_speed,2)/10+10+np.random.uniform(0., 20.)
            #env 9
            min_safe_distanse=np.power(ego_v-other_v,2)/4+np.power(other_v-self.target_speed,2)/10+8+np.random.uniform(0., 10.)
            
            
            
        else:
            min_safe_distanse=20#25

        
        ego_distance = 90
        other_distance = ego_distance  +min_safe_distanse #+np.random.uniform(0., 20.)
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("b", "c", 1)).position(ego_distance, 0),
                                                     speed=ego_v)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("b", "c", 1)).position(other_distance, 0), speed=other_v, target_speed = self.target_speed))
        print("Ego init speed: {} other speed: {} target speed:{}".format(ego_v, other_v,self.target_speed))
        self.vehicle = ego_vehicle
        #print(self.vehicle.position[0])


register(
    id='acc-v0',
    entry_point='highway_env.envs:ACCEnv'
)
