from typing import List, Tuple, Union

import numpy as np
import os
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle
from intersection_behavior import Policy, Action, PolicyParams, RiskParam, MotionConstraints, FrenetTrajectory, FrenetState
from highway_env.trajectory_vis.visualizer import Visualizer
USE_EASIER_POLICY = False
SAVE_VIDEO = True
class FrenetVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]
    POLICY_DT = 0.25
    SIM_DT = 0.05
    MAX_ACC = 8.
    DEC_LIMIT = 16.
    MAX_JERK = 20.
    MIN_JERK = -40.
    MAX_SPEED = 30.
    frenet_action = 0
    COMFORT_JERK = 1.
    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        self.frenet_action = 0
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

        #Policy Interface for Frenet Planner
        motion_constraints = MotionConstraints(speed_limit=self.MAX_SPEED, acceleration_limit=self.MAX_ACC,
                                               deceleration_limit=self.DEC_LIMIT, max_jerk_limit=self.MAX_JERK, min_jerk_limit=self.MIN_JERK,comfort_jerk_limit=self.COMFORT_JERK)

        self.risk_param = RiskParam(veh_length=5.,veh_width=2., safety_distance=1.0, time_headway=0.)
        self.risk_param_easier = RiskParam(veh_length=5.,veh_width=2., safety_distance=0.5, time_headway=0.)

        self.policy_params = PolicyParams(max_acc_agent=4., dt=self.POLICY_DT, history_length=4, max_real_agents=20, max_occl_agents=4, motion_constraints=motion_constraints, risk_param=self.risk_param)
        self.policy_params_easier = PolicyParams(max_acc_agent=1., dt=self.POLICY_DT, history_length=4, max_real_agents=20, max_occl_agents=4, motion_constraints=motion_constraints, risk_param=self.risk_param_easier)

        self.policy = Policy(self.policy_params)
        self.policy_easier = Policy(self.policy_params_easier)

        self.control_freq=1
        self.counter = 0


        if True:
            record_path = "/home/kamran/helsinki_dir/tmp/"


            self.vis = Visualizer(step_size = self.SIM_DT, hist_size = 100, controller=self, save_fig=SAVE_VIDEO, record_path=record_path)
            vis_inp = {'q_values': None, 'attention_weights': None, 'action': None, 'alpha':None}
    @classmethod
    def create_from(cls, vehicle: "FrenetVehicle") -> "FrenetVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination: str) -> "FrenetVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action is None:
            return
        print("action here in act: {}".format(action))
        if action == "PROG":
            self.frenet_action = 1
        elif action == "DEF":
            self.frenet_action = 0
        else:
            self.frenet_action = 1
        # if action == "FASTER":
        #     self.target_speed += self.DELTA_SPEED
        # elif action == "SLOWER":
        #     self.target_speed -= self.DELTA_SPEED
        #
        # elif action == "LANE_RIGHT":
        #     _from, _to, _id = self.target_lane_index
        #     target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
        #     if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
        #         self.target_lane_index = target_lane_index
        # elif action == "LANE_LEFT":
        #     _from, _to, _id = self.target_lane_index
        #     target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
        #     if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
        #         self.target_lane_index = target_lane_index
        # action = {"steering": self.steering_control(self.target_lane_index),
        #           "acceleration": -1}
        # action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        super().act(self.action)

    def step(self, dt):
        print("dt: {}".format(dt))
        self.follow_road()
        self.counter += 1
        if self.frenet_action is None:
            self.frenet_action = 0
        if self.counter==self.control_freq:
            self.counter = 0
            # try:
            self.feed_state_to_policy_merging(self.policy)
            #h_acceleration = self.policy.get_helsinki_acceleration(v0=float(self.speed), a0=float(self.action['acceleration']), action=ACTION)
            frenet_trajectory_pr = self.policy.get_helsinki_trajectory(v0=float(self.speed), a0=float(self.action['acceleration']), action=1)
            frenet_trajectory_def = self.policy.get_helsinki_trajectory(v0=float(self.speed), a0=float(self.action['acceleration']), action=0)
            trajectoris = [frenet_trajectory_def, frenet_trajectory_pr]
            h_acceleration = trajectoris[self.frenet_action].get_FState(1).s_dd
            vis_inp = {'frenet_trajectories': trajectoris, 'vel_history': [self.speed], 'accl_history' : [self.action['acceleration']], 'jerk_history' : [h_acceleration - self.action['acceleration']]}
            self.vis.visualize(vis_inp)
            print("RL action: {}".format(self.frenet_action))
            # # except:
            # #     h_acceleration = -100.
            # if h_acceleration == -100.:
            #     if USE_EASIER_POLICY:
            #         print("Normal policy failed, trying to apply easier safety policy")
            #         self.feed_state_to_policy_merging(self.policy_easier)
            #         h_acceleration = self.policy_easier.get_helsinki_acceleration(v0=float(self.speed), a0=float(self.action['acceleration']), action=ACTION)
            #     else:
            #         raise ValueError('No Maneuver was Generated.')

            # print("Old acc: {}".format(self.action['acceleration']))
            # print("Helsinki acc: {}".format(new_acceleration))
            #acceleration = self.action['acceleration'] + dt*(new_acceleration-self.action['acceleration'])

            print("h_acceleration: {}".format(h_acceleration))

            new_acceleration = h_acceleration#self.action['acceleration'] + (h_acceleration-self.action['acceleration'])*(self.SIM_DT/dt)
            print("new_acceleration: {}".format(new_acceleration))

            new_acceleration = max(min((self.road.get_vehicle_max_lane_speed(self) - self.speed) / dt, new_acceleration), -self.speed / dt)

            print("new acceleration cliped: {}".format(new_acceleration))

            jerk = (new_acceleration - self.action['acceleration'])
            if self.speed>1. and jerk<-0.5:
                print("Jerk: *********************************************************************************************************************************************************** {}".format(jerk))
            action = {"steering": self.steering_control(self.target_lane_index),
                      "acceleration": new_acceleration}
            action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
            super().act(action)
        super().step(dt)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)
    def get_distance(self, position):
        current_lane = self.target_lane_index
        while self.road.network.get_lane(current_lane).after_end(position):
            current_lane = self.road.network.next_lane(current_lane,
                                                       route=self.route,
                                                       position=position,
                                                       np_random=self.road.np_random)

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)

        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                                           -1, 1))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [self.route[0:index+1] + [(self.route[index][1], destination, self.route[index][2])]
                  for destination in next_destinations_from]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.randint(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                           for t in times]))



    def feed_state_to_policy_merging(self, policy):
        ego_s = self.road.get_ego_vehicle_distance()
        front_vehicles = self.road.get_front_vehicles()
        merging_vehicles = self.road.get_merging_vehicles()
        policy.reset_situation()
        lane_to_stl_distance = 0.
        #v_max_ego = self.road.get_vehicle_max_lane_speed(self)
        #print("Max lane speed ego: {}".format(v_max_ego))
        policy.update_speed_limit(self.MAX_SPEED)#assume ego lane speed limit is same as other lane speed limit
        v_max_lane_speed_other = 30.#Streight lane
        lane_id=policy.add_lane_situation(ego_d=float(ego_s), ego_v=float(self.speed), v_max_lane=v_max_lane_speed_other, stl_d=lane_to_stl_distance)
        # #print("ego d: {} v: {}".format(ego_d, ego_v))
        i=0
        for merging_v in merging_vehicles:
            policy.add_agent(agent_id=i, agent_d=merging_v[0], agent_v=float(merging_v[1]), lane_id=int(lane_id), occlusion_agent=False, merging_lane=True)
            i +=1
        #
        for front_v in front_vehicles:
            policy.add_front_agent(agent_id=i, agent_v=float(front_v[1]), v_max_lane=v_max_lane_speed_other, ego_d=front_v[0], ego_v=float(self.speed))
            i +=1



        # CZ_W = 6.
        # num_cars = len(merging_vehicles)
        # if num_cars>0:
        #     for i in range(num_cars):
        #         obj_d = float(cur_obs['other_cars']['positions'][i][0])
        #         if obj_d<-CZ_W - 2.:#if object has passed the conflict zone
        #             obj_d+=CZ_W#for cars after the CZ, the distance reference should be from left edge of CZ
        #             if obj_d < ego_d: #if the object who passed the conflict zone is in front of ego
        #                 #print("obj d {}".format(obj_d))
        #                 #print("front object at distance: {} v: {}".format(((-obj_d)+(ego_d)), cur_obs['other_cars']['velocities'][i][0]))
        #                 self.policy.add_front_agent(agent_id=i, agent_v=float(abs(cur_obs['other_cars']['velocities'][i][0])), v_max_lane=v_max_lane, ego_d=((-obj_d)+(ego_d)), ego_v=float(ego_v))
        #         elif obj_d>-CZ_W - 2.:# if object is inside or behind conflict zone
        #             if obj_d<0:
        #                 obj_d = 0.
        #             #print("Merging object at distance: {} v: {}".format(obj_d, cur_obs['other_cars']['velocities'][i][0]))
        #             self.policy.add_agent(agent_id=i, agent_d=obj_d, agent_v=float(abs(cur_obs['other_cars']['velocities'][i][0])), lane_id=int(lane_id), occlusion_agent=False, merging_lane=True)
        maximum_visible_distance = 500.
        policy.add_agent(agent_id=-1, agent_d=abs(maximum_visible_distance), agent_v=self.MAX_SPEED, lane_id=int(lane_id), occlusion_agent=True, merging_lane=True)
        return