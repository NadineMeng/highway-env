import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.utils import near_split


class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    COLLISION_REWARD: float = -1
    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.2
    MERGING_SPEED_REWARD: float = -0.5
    LANE_CHANGE_REWARD: float = -0.05

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.RIGHT_LANE_REWARD * self.vehicle.lane_index[2] / 1 \
                 + self.HIGH_SPEED_REWARD * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_SPEED_REWARD * \
                          (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        return utils.lmap(action_reward[action] + reward,
                          [self.COLLISION_REWARD + self.MERGING_SPEED_REWARD,
                            self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > 320

    def _reset(self) -> None:
        #high_speed
        vehicles_density=np.random.uniform(0.6,1.5)
        avg_speed = 30.
        if np.random.random()<0.5:#low_speed
            vehicles_density=np.random.uniform(0.5,1.)
            avg_speed = 15.
        self.config.update({"vehicles_density": vehicles_density,})
        self.config.update({"avg_speed": avg_speed,})

        self._make_road()
        self._make_vehicles()

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "vehicles_density": 1,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "normalize":False,
                "order": "sorted"
            },
            "vehicles_count":20,
            "policy_frequency": 2,
            "duration": 40,
            'real_time_rendering': True,
        })
        return config

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
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
        road = self.road


        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        print(self.config["other_vehicles_type"])
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(140, 0), speed=30)
        for _ in range(self.config["vehicles_count"]):

            lanes = np.arange(2)
            lane_id = self.road.np_random.choice(lanes, size=1).astype(int)[0]
            lane = self.road.network.get_lane(("a", "b", lane_id))
            speed=np.random.normal(self.config["avg_speed"], 3.)
            speed=min(speed, lane.speed_limit)
            new_vehicle = other_vehicles_type.create_random(self.road,
                                                  lane_from="a",
                                                  lane_to="b",
                                                  lane_id=lane_id,
                                                  speed=speed,
                                                  spacing=1 / self.config["vehicles_density"],
                                                  ).plan_route_to("d")


            #
            #new_vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"],
            #                                                speed=speed)
            new_vehicle.enable_lane_change = np.random.random()<0.5
            self.road.vehicles.append(new_vehicle)

        #IMPORTANT: Ego vehicle should be added after others!
        road.vehicles.append(ego_vehicle)

        self.vehicle = ego_vehicle

register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)
