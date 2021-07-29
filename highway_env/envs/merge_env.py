import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.frenet_vehicle import FrenetVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.utils import near_split

START_DIS = 540
END_DIS = 720
class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    SUCCESS_REWARD: float = 1.
    COLLISION_REWARD: float = -1.
    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.2
    MERGING_SPEED_REWARD: float = -0.5
    LANE_CHANGE_REWARD: float = -0.05

    def __init__(self, avg_speed=-1, min_density=0., max_density=1., cooperative_prob=0., observation="LIST", negative_cost=False, sample_vehicles_count=0, random_vehicles_count=20, force_render=False, seed=123, frames_per_decision=1):
        self.avg_speed = avg_speed
        self.min_density = min_density,
        self.max_density = max_density
        self.config = self.default_config()
        self.config.update({"cooperative_prob": cooperative_prob,})
        self.config.update({"negative_cost": negative_cost,})
        self.config.update({"sample_vehicles_count": sample_vehicles_count,})
        self.config.update({"random_vehicles_count": random_vehicles_count,})
        self.config.update({"force_render": force_render,})
        self.config.update({"policy_frequency": self.config["policy_frequency"]*frames_per_decision,})

        if observation == "GRID":
            self.config.update({
                "observation": {
                    "type": "OccupancyGrid",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                    "grid_step": [5, 5],
                    "absolute": False
                }})
        elif observation == "LIST":
            self.config.update({
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-1000, 1000],
                        "y": [-100, 100],
                        "vx": [-40, 40],
                        "vy": [-40, 40]
                    },
                    "absolute": False,
                    "normalize":True,
                    "see_behind":True,
                    "clip":False,
                    "order": "sorted"
                }})
        else:
            raise ValueError('Observation {} not implemented'.format(observation))
        super().__init__(self.config)
        self.seed(seed)
        np.random.seed(seed)

    def _cost(self, action: int) -> float:
        cost = 0.
        if self.vehicle.crashed:
            cost = 1.
        elif self.vehicle.position[0] > END_DIS and self.config["negative_cost"] is True:
            cost = -1.
        return cost


    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        # action_reward = {0: self.LANE_CHANGE_REWARD,
        #                  1: 0,
        #                  2: self.LANE_CHANGE_REWARD,
        #                  3: 0,
        #                  4: 0}
        # reward = self.COLLISION_REWARD * self.vehicle.crashed \
        #          + self.RIGHT_LANE_REWARD * self.vehicle.lane_index[2] / 1 \
        #          + self.HIGH_SPEED_REWARD * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1)
        #
        # # Altruistic penalty
        # for vehicle in self.road.vehicles:
        #     if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
        #         reward += self.MERGING_SPEED_REWARD * \
        #                   (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
        #
        # return utils.lmap(action_reward[action] + reward,
        #                   [self.COLLISION_REWARD + self.MERGING_SPEED_REWARD,
        #                     self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
        #                   [0, 1])
        r = -0.001#Time penalty

        if self.vehicle.position[0] > END_DIS and self.config["negative_cost"] is False:
            r = 0.01
        #print("Reward: {}".format(r))
        return r

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        terminal = self.vehicle.crashed or self.vehicle.position[0] > END_DIS or self.road.any_crash
        #if terminal:
        #    print("Terminal......................")
        return terminal

    def _reset(self) -> None:
        # #high_speed
        if self.avg_speed == -1:
            avg_speed = 30.
            vehicles_density=np.random.uniform(0.6,1.5)
            if np.random.random()<0.5:#by 50 percent switch to low_speed config
                avg_speed = 10.
                vehicles_density=np.random.uniform(0.3,0.6)
        else:
            avg_speed = self.avg_speed
            vehicles_density=np.random.uniform(self.min_density, self.max_density)
        self.config.update({"vehicles_density": vehicles_density,})
        self.config.update({"avg_speed": avg_speed,})

        self._make_road()
        self._make_vehicles()

    @classmethod
    def default_config(cls) -> dict:
        print("default config")
        config = super().default_config()
        config.update({
            "vehicles_density": 1,
            "cooperative_prob": 0.,
            "negative_cost": False,
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False
            },
            "policy_frequency": 2,
            "duration": 70,
            'real_time_rendering': False,
        })
        return config

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [550, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [350, y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([350, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("d", "e", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], -StraightLane.DEFAULT_WIDTH), lkb.position(ends[1], -StraightLane.DEFAULT_WIDTH) + [ends[2], 0.],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("bb", "cc", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        #road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road


        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        #print(self.config["other_vehicles_type"])
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))
        #Make init velocity of ego vehicle a random value
        ego_init_speed = np.random.uniform(10.,18.)
        ego_init_speed=np.clip(ego_init_speed, 5., road.network.get_lane(("j", "k", 0)).speed_limit)

        ego_vehicle = FrenetVehicle(road,
                                                     road.network.get_lane(("j", "k", 0)).position(START_DIS, 0), speed=ego_init_speed)



        for _ in range(self.config["sample_vehicles_count"]):
            lanes = np.arange(2)
            lane_id = self.road.np_random.choice(lanes, size=1).astype(int)[0]
            lane = self.road.network.get_lane(("b", "c", lane_id))
            distance_back = np.random.uniform(600, 620.)
            speed=np.random.normal(self.config["avg_speed"], 3.)
            speed=np.clip(speed, 5., lane.speed_limit)
            sample_vehicle = other_vehicles_type.create_random(self.road,
                                                            lane_from="b",
                                                            lane_to="c",
                                                            lane_id=lane_id,
                                                            speed=speed,
                                                            spacing=1 / self.config["vehicles_density"],
                                                            cooperative=np.random.uniform()<self.config["cooperative_prob"],
                                                            )
            #sample_vehicle = other_vehicles_type(road, road.network.get_lane(("a", "b", lane_id)).position(distance_back, 0), speed=speed, cooperative=np.random.uniform()<self.config["cooperative_prob"])
            sample_vehicle.enable_lane_change = False#np.random.random()<0.5
            self.road.vehicles.append(sample_vehicle)


        for _ in range(self.config["random_vehicles_count"]):
            lanes = np.arange(2)
            lane_id = self.road.np_random.choice(lanes, size=1).astype(int)[0]
            if np.random.uniform()<0.5:
                lane = self.road.network.get_lane(("a", "b", lane_id))
            else:
                lane = self.road.network.get_lane(("b", "c", lane_id))
            speed=np.random.normal(self.config["avg_speed"], 3.)
            speed=np.clip(speed, 5., lane.speed_limit)
            cooperative = np.random.uniform()<self.config["cooperative_prob"]
            new_vehicle = other_vehicles_type.create_random(self.road,
                                                  lane_from="a",
                                                  lane_to="b",
                                                  lane_id=lane_id,
                                                  speed=speed,
                                                  spacing=1 / self.config["vehicles_density"],
                                                  cooperative=cooperative,
                                                  )


            #
            #new_vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"],
            #                                                speed=speed)
            new_vehicle.enable_lane_change = False#np.random.random()<0.5
            self.road.vehicles.append(new_vehicle)

        #IMPORTANT: Ego vehicle should be added after others!
        road.append_ego_vehicle(ego_vehicle)

        self.vehicle = ego_vehicle

register(
    id='mergefast-v0',
    entry_point='highway_env.envs:MergeEnv',
    kwargs={'avg_speed' : 30, 'min_density' : 0.6, 'max_density' : 1.5},
)
register(
    id='mergeslow-v0',
    entry_point='highway_env.envs:MergeEnv',
    kwargs={'avg_speed' : 10, 'min_density' : 0.3, 'max_density' : 0.6},
)

register(
    id='mergemixed-v0',
    entry_point='highway_env.envs:MergeEnv',
    kwargs={'avg_speed' : -1},
)

register(
    id='mergesample-v0',
    entry_point='highway_env.envs:MergeEnv',
    kwargs={'avg_speed' : 10, 'min_density' : 0.3, 'max_density' : 0.6, 'sample_vehicles_count' : 5, 'random_vehicles_count' : 0},
)

