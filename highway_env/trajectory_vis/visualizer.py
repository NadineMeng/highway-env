from collections import deque
import os
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from intersection_behavior import FrenetTrajectory

plt.ioff()
sns.set()
sns.set_style('whitegrid')
sns.set_palette('husl', 4)
sns.set_context('paper', font_scale=0.8)
class Visualizer(object):
    def __init__(
            self, hist_size, controller, save_fig, record_path
    ):
        self.plotters = [
            FrenetTrajectoryPlotter(controller, save_fig, record_path),
            AgentPlotter(hist_size, controller, save_fig, record_path)]


    def visualize(self, plotter_inp):
        for plotter in self.plotters:
            plotter.draw(**plotter_inp)
    # self.save_frame()
    # self.step += 1


# def save_frame(self):
# 	screen_buffer = (
# 	    np.frombuffer(self.screen.get_buffer(), dtype=np.int32)
# 	    .reshape(self.screen_height, self.screen_width))
# 	sb = screen_buffer[:, 0:self.screen_width]
# 	self.record_frame[..., 2] = sb % 256
# 	self.record_frame[..., 1] = (sb >> 8) % 256
# 	self.record_frame[..., 0] = (sb >> 16) % 256
# 	frame_number = self.step
# 	for file_type in self.file_types:
# 		if not file_type:
# 			continue
# 		filename = (self.filename_format.format(frame_number) + '.{}'.format(file_type))
# 		im = Image.fromarray(self.record_frame)
# 		im.save(os.path.join(self.record_path, filename))


# def generate_video(self, video_file='video_mp4'):
# 	if 'png' not in self.file_types:
# 		return
# 	os.chdir(self.record_path)
# 	file_regex = self.filename_format.replace('{:', '%').replace('}', '')
# 	file_regex += '.png'
# 	subprocess.call(['ffmpeg', '-r', '30', '-f', 'image2', '-s', '1920x1080',
#                   		 '-i', file_regex, '-vcodec', 'libx264', '-crf', '25',
#                    	 '-pix_fmt', 'yuv420p', video_file])


class FrenetTrajectoryPlotter(object):
    def __init__(self, controller, save_fig, record_path):
        self.save_fig = save_fig
        self.record_path = record_path
        self.controller = controller
        matplotlib.pyplot.close("all")
        #plt.ion()
        #plt.show()
        self.dt = 0.25
        self.colors = ["g", "r"]
        self.counter = 1
    def draw(self, frenet_trajectories, img_index, **kwargs):
        self.fig, self.ax = plt.subplots(3, 1, figsize=[3.8, 3.0])

        for a in self.ax:
            a.cla()
        self.ax[0].set(ylim=(0, 100.))
        self.ax[0].set(xlim=(0, 10.))
        self.ax[1].set(ylim=(0., self.controller.MAX_SPEED))
        self.ax[1].set(xlim=(0, 10.))
        self.ax[2].set(ylim=(-self.controller.DEC_LIMIT, self.controller.MAX_ACC))
        self.ax[2].set(xlim=(0, 10.))
        for i in range(len(frenet_trajectories)):
            self.draw_ft(frenet_trajectories[i], self.colors[i])

        plt.draw()
        #plt.pause(1.)
        if self.save_fig:
            plt.savefig(self.record_path + "/frenet/frame_{:0>4d}.png".format(img_index), dpi=150)
        self.counter+=1#Todo remove it
        plt.close()
    def draw_ft(self, frenet_trajectory, color):
        #
        length = frenet_trajectory.get_FStates_length()
        Tf = frenet_trajectory.get_time(length-1)
        time = np.arange(0, Tf, self.dt)
        dt_tr = Tf/length
        distances = []
        velocities = []
        accelerations = []
        for t in time:
            index = int(t/dt_tr)
            distances.append(frenet_trajectory.get_FState(index).s)
            velocities.append(frenet_trajectory.get_FState(index).vel)
            accelerations.append(frenet_trajectory.get_FState(index).s_dd)

        self.ax[0].plot(time, distances, linewidth=2, color=color)
        self.ax[1].plot(time, velocities, linewidth=2, color=color)
        self.ax[2].plot(time, accelerations, linewidth=2, color=color)

class EnvPlotter(object):
    def __init__(self, env):
        self.env = env

    def draw(self, attention_weights, **kwargs):
        a=1
    #obs = self.env.render('rgb_array', attention_weights).astype(np.int32)


class AgentPlotter(object):
    def __init__(self, hist_size, controller, save_fig, record_path):
        self.save_fig = save_fig
        self.record_path = record_path
        self.controller = controller
        init_val = [0.] * hist_size
        self.vel = deque(init_val, maxlen=hist_size)
        self.acc = deque(init_val, maxlen=hist_size)
        self.jerk = deque(init_val, maxlen=hist_size)
        self.data_time = deque(init_val, maxlen=hist_size)
        self.counter=1
        #plt.ion()
        #plt.show()
    def draw(self, time_history, vel_history, accl_history, jerk_history, img_index, **kwargs):
        self.fig, self.ax = plt.subplots(3, 1, figsize=[3.8, 3.0], sharex=True)


        for a in self.ax:
            a.cla()
        self.vel.extend(vel_history)
        self.acc.extend(accl_history)
        self.jerk.extend(jerk_history)
        self.data_time.extend(time_history)
        st_time = r"$[s]$"
        st_jerk = r"$[m/s^3]$"

        st_acc = r"$[m/s^2]$"
        st_vel = r"$[m/s]$"

        self.ax[0].set(ylim=(0, self.controller.MAX_SPEED + 1))
        self.ax[1].set(ylim=(-self.controller.DEC_LIMIT, self.controller.MAX_ACC))
        self.ax[2].set(ylim=(-12.0, 12.0))

        self.ax[0].set_ylabel("Velocity \n" + st_vel, fontsize=10)
        self.ax[1].set_ylabel("Acceleration \n"+st_acc, fontsize=10)
        self.ax[2].set_ylabel("Jerk \n"+st_jerk, fontsize=10)

        self.ax[2].set_xlabel("Time "+st_time, fontsize=16)
        # self.ax[0].tick_params(axis='both', which='major', labelsize=9)
        # self.ax[1].tick_params(axis='both', which='major', labelsize=9)
        # self.ax[2].tick_params(axis='both', which='major', labelsize=9)

        #current_time_list = [x+self.counter*self.step_size for x in self.time]

        self.ax[0].plot(self.data_time, self.vel, linewidth=2, color="g")
        self.ax[1].plot(self.data_time, self.acc, linewidth=2, color="b")
        self.ax[2].plot(self.data_time, self.jerk, linewidth=2, color="r")

        # self.ax[0].set_yticks([0,5,10])
        # self.ax[1].set_yticks([-8,0,4])
        # self.ax[2].set_yticks([-10,-5,0,5,10])

        # plt.rcParams.update({'font.size': 14})
        # self.fig.tight_layout()
        # self.fig.canvas.draw()
        plt.draw()
        #plt.pause(0.1)
        if self.save_fig:
            plt.savefig(self.record_path +"/agent/frame_{:0>4d}.png".format(img_index), dpi=150)
        # plt.savefig("/home/kamran/helsinki_data/tmp/agent/frame_{:0>4d}.svg".format(self.counter), dpi=150)
        # plt.close()
        self.counter+=1#TODO remove it
        plt.close()

class QValuePlotter(object):
    def __init__(self, controller, limits):
        self.controller = controller
        self.limits = limits
        self.names=['decelerate', 'accelerate', 'idle']
        self.colors=['red','blue','green']
        self.counter=1
    def draw(self, q_values, action, alpha=0, **kwargs):
        self.fig, self.ax = plt.subplots(figsize=[3.8, 3.0])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.95)
        labels_font=12
        ticks_font=9
        legend_font=8
        self.ax.cla()
        if len(q_values.shape) > 1:#IQN
            self.ax.set_ylabel("Density", fontsize=labels_font)
            self.ax.set_xlabel(r"$\mathcal{R}$", fontsize=labels_font)
            self.ax.tick_params(axis='both', which='minor', labelsize=ticks_font)
            self.ax.set_yticks([0, 2., 4.])
            self.ax.set_xticks([-1.5,-0.5 ,0.5, 1.5])
            q_values = np.transpose(q_values)
            for a, q in enumerate(q_values):
                sns.kdeplot(q, ax=self.ax, shade=a==action, label=self.names[a],color=self.colors[a], legend=False)
            self.ax.set(ylim=(0, 4.0))
            self.ax.set(xlim=(-1.5, 1.5))
            self.ax.text(0.32, 1.15, "IQN Policy "+ r"$\alpha: $"+"{}".format(alpha), transform=self.ax.transAxes, fontsize=labels_font,
                         verticalalignment='top', bbox=props)
            plt.legend(fontsize=legend_font)

        elif q_values[-1]!=100:#DQN
            self.ax.set_ylabel("Q", fontsize=labels_font)
            self.ax.set_xlabel("RL actions ", fontsize=labels_font) #+ r"$[m/s^3]$"
            self.ax.tick_params(axis='both', which='major', labelsize=ticks_font)
            self.ax.set_yticks([-0.5, 0, 0.5])

            mq=[q_values[1], q_values[2], q_values[0]]
            bar=sns.barplot(y=mq, x=np.arange(3), ax=self.ax, palette=self.colors)
            self.ax.set(ylim=(-2, 2))
            self.ax.set(xlim=(-1, 3))
            self.ax.set_xticklabels(["Progressive","Coop","Conservative"])
            plt.rcParams['xtick.labelsize'] = 4

            max_q =  max(q_values)
            for i,thisbar in enumerate(bar.patches):
                if mq[i]==max_q:
                    thisbar.set_hatch('x')
            self.ax.text(0.15, 0.99, "DQN Policy", transform=self.ax.transAxes, fontsize=labels_font,
                         verticalalignment='top', bbox=props)
        else:#Rule-based
            final_x = np.arange(0,3)
            final_y = [0]*3
            # for d in q_values[:-1]:
            # 	final_y[int(int(d)+5)] = 1
            self.ax.set_xlabel("Motion Planner Maneuver Mode", fontsize=labels_font)

            final_y[int(int(q_values[-2]))] = 1

            bar=sns.barplot(y=final_y, x=final_x, ax=self.ax, palette=sns.diverging_palette(10, 230,n=11,s=100, l=40,center="dark"))#palette="flare")
            self.ax.set(ylim=(0, 3.1))
            self.ax.set_yticks([0,1])#set(xticks=np.arange(-5,6))
            self.ax.set_yticklabels(["",'selected'])
            self.ax.set_xticks([0,1,2])#set(xticks=np.arange(-5,6))
            self.ax.set_xticklabels(["Progressive","Coop","Conservative"])
            self.ax.tick_params(axis='both', which='major', labelsize=ticks_font)

            bar.patches[int(q_values[-2])].set_hatch('x')

            self.ax.text(0.05, 0.99, "MPC Agent", transform=self.ax.transAxes, fontsize=14,
                         verticalalignment='top', bbox=props)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.savefig("/home/kamran/helsinki_data/tmp/qplots/frame_{:0>4d}.png".format(self.counter), dpi=150)
        self.counter+=1
        plt.close()