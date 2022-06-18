import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle

END_DISTANCE = 450.
SPEED_LIMITS = [7., 30., 50., 70., 120.]#KpH

class ACCEnv(AbstractEnv):

    """
    A highway merge negotiation environment.
    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    def __init__(self, speed_limit=-1):
        if speed_limit == -1:
            speed_limit = np.random.choice(SPEED_LIMITS) / 3.6 #m/s
        print("Initializing ACC Env with Speed Limit: {}".format(speed_limit))
        super().__init__()
        self.config.update({
            "speed_limit": speed_limit #m/s
        })
        self.config["action"].update({
            "speed_range": [0, speed_limit]
        })    

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "time_reward": -0.001,
             "action": {
                "type": "ContinuousAction",
                "lateral": False,
                "speed_range": [0, 30.]
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 10,  # [Hz],
            "speed_limit": 30. #m/s
        })
        cfg["action"].update({
            "speed_range": [0, cfg["speed_limit"]]
        })
        cfg["observation"].update({
            "vehicles_count": 2
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = self.config["collision_reward"] * self.vehicle.crashed + self.config["time_reward"]
        if self.vehicle.crashed or self.vehicle.position[0] > END_DISTANCE:
            reward = reward + 0.2
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > END_DISTANCE

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 250]  # Before, converging, merge, after
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
        other_v = speed_limit * (np.random.normal() * 0.3 + 0.7)
        other_v = np.clip(other_v, 0., speed_limit)       
        
        ego_distance = 90
        other_distance = ego_distance + 120. + np.random.uniform(0., 20.)
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("b", "c", 1)).position(ego_distance, 0),
                                                     speed=ego_v)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("b", "c", 1)).position(other_distance, 0), speed=other_v, target_speed = speed_limit))
        #road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        #road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        # merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        # merging_v.target_speed = 30
        # road.vehicles.append(merging_v)
        print("Ego init speed: {} other speed: {}".format(ego_v, other_v))
        self.vehicle = ego_vehicle


register(
    id='acc-v0',
    entry_point='highway_env.envs:ACCEnv'
)