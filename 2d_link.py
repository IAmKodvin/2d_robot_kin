import numpy as np
from matplotlib import pyplot as plt, animation
from math import *


class Rob2d:

    origin = np.array([0.0, 0.0])
    L = np.array([15.0, 10.0])
    q = np.array([0.0, 0.0])
    ee = np.array([0.0, 0.0])

    def __init__(self, q_init=None):
        if q_init is None:
            self.q = np.array([0, 0])
        if np.any(q_init):
            self.q = q_init

        # Calculate start position
        self.fwd(self.q)

    def fwd(self, q):
        self.q = q
        self.ee[0] = self.L[1] * cos(q[1]) + self.L[0] * cos(q[0])
        self.ee[1] = self.L[1] * sin(q[1]) + self.L[0] * sin(q[0])

    def get_ee(self):
        return self.ee

    def get_q(self):
        return self.q

    def get_joint_positions(self):
        joints = np.zeros([len(self.q)+1, 2])
        origin = np.array([0, 0])
        for i in range(0, len(self.q)):
            joints[i+1, 0] = np.array([self.L[i] * cos(self.q[i]) + origin[0]])
            joints[i+1, 1] = np.array([self.L[i] * sin(self.q[i]) + origin[1]])
            origin = joints[i+1]

        return joints

    def rob_plt(self, ax = None):
        plt.rcParams["figure.figsize"] = [7.50, 7.50]
        plt.rcParams["figure.autolayout"] = True

        joints = self.get_joint_positions()

        if ax is None:
            fig, ax = plt.subplots(1, 1)

            max_L = np.sum(self.L) + 5
            ax.set_xlim(-max_L, max_L)
            ax.set_ylim(-max_L, max_L)

        ax.plot(joints[:, 0], joints[:, 1])

        return ax

    def inv_remove_faulty_sols(self, j_x, j_y, x, y):
        # remove impossible solution (L0 > L1 for this robot)
        tmp = np.array([0.0, 0.0])
        for i in range(0, 2):
            if abs(x - j_x[i]) <= self.L[1]:
                tmp[0] = j_x[i]
        for i in range(2, 4):
            if abs(x - j_x[i]) <= self.L[1]:
                tmp[1] = j_x[i]
        j_x = tmp
        return j_x

    def inv(self, x, y):
        j_y = [0.0, 0.0] # position of joint 1
        j_x = [0.0, 0.0, 0.0, 0.0]

        # constant
        a = (x ** 2 + y ** 2 + self.L[0] ** 2 - self.L[1] ** 2)

        sqrt_expr = (4 * y * a) ** 2 - 4 * (4 * y ** 2 + 4 * x ** 2) * (a ** 2 - 4 * self.L[0] ** 2 * x ** 2)
        if sqrt_expr < 0:
            print("Imaginary solution?")
        elif sqrt_expr == 0:
            print("single solution")
            try:
                j_y[0] = 4 * y * a / (2 * (4 * y ** 2 + 4 * x ** 2))
            except RuntimeWarning:
                raise ValueError("Impossible configuration")
            j_y[1] = j_y[0]
        else:
            j_y[0] = (4 * y * a + sqrt(
                (4 * y * a) ** 2 - 4 * (4 * y ** 2 + 4 * x ** 2) * (a ** 2 - 4 * self.L[0] ** 2 * x ** 2))) / (
                                 2 * (4 * y ** 2 + 4 * x ** 2))

            j_y[1] = (4 * y * a - sqrt(
                (4 * y * a) ** 2 - 4 * (4 * y ** 2 + 4 * x ** 2) * (a ** 2 - 4 * self.L[0] ** 2 * x ** 2))) / (
                                 2 * (4 * y ** 2 + 4 * x ** 2))

        j_x[0] = sqrt(self.L[0] ** 2 - j_y[0] ** 2)
        j_x[1] = - sqrt(self.L[0] ** 2 - j_y[0] ** 2)
        j_x[2] = sqrt(self.L[0] ** 2 - j_y[1] ** 2)
        j_x[3] = - sqrt(self.L[0] ** 2 - j_y[1] ** 2)

        # config
        j_x = self.inv_remove_faulty_sols(j_x, j_y, x, y)

        # Find end-link axis solution
        theta_j1 = np.array([0.0, 0.0])

        dx = np.array([x - j_x[0], x - j_x[1]])
        dy = np.array([y - j_y[0], y - j_y[1]])
        for i in range(0, len(j_x)):
            if dx[i] > 0 and dy[i] > 0:
                theta_j1[i] = atan(dy[i] / dx[i])
                #print("case 1: i=", i)
            elif dx[i] < 0 and dy[i] > 0:
                theta_j1[i] = pi + atan(dy[i] / dx[i])
                #print("case 2: i=", i)
            elif dx[i] < 0 and dy[i] < 0:
                theta_j1[i] = pi + atan(dy[i] / dx[i])
                #print("case 3: i=", i)
            else:
                theta_j1[i] = 2 * pi + atan(dy[i] / dx[i])
                #print("case 4: i=", i)

        #print("config 1: j_x = ", j_x[0], " j_y = ", j_y[0])
        #print("config 2: j_x = ", j_x[1], " j_y = ", j_y[1])
        #print("thetas = ", degrees(theta_j1[0]), degrees(theta_j1[1]))

        q_sol = np.array([0.0, 0.0])
        #i_delta_theta_min = np.argmin(np.abs([self.q[0] - theta_j1[0], self.q[0] - theta_j1[1]]))

        # chosen config
        #i_config = i_delta_theta_min
        #q_sol[1] = theta_j1[i_config]
        #print("q sol: ", degrees(q_sol[i_config]))

        # find base link
        theta_j0 = np.array([0.0, 0.0])
        for i in range(0, len(theta_j1)):
            try:
                if j_x[i] > 0 and j_y[i] > 0:
                    theta_j0[i] = acos(j_x[i] / self.L[0])
                elif j_x[i] < 0 and j_y[i] > 0:
                    theta_j0[i] = acos(j_x[i] / self.L[0])
                elif j_x[i] < 0 and j_y[i] < 0:
                    theta_j0[i] = 2*pi - acos(j_x[i] / self.L[0])
                else:
                    theta_j0[i] = 2*pi - acos(j_x[i] / self.L[0])

                #print("base theta: ", degrees(q_sol[0]))
            except ValueError:
                print("j_x=", j_x, "j_x/L[0]=", j_x/self.L[0], "theta=", degrees(q_sol[1]))

        i_delta_theta_min = np.argmin(np.abs([self.q[0] - theta_j0[0], self.q[0] - theta_j0[1]]))

        # chosen config
        i_config = i_delta_theta_min
        q_sol = np.array([theta_j0[i_config], theta_j1[i_config]])

        self.ee[0] = x
        self.ee[1] = y
        self.q = q_sol

        #print("\n")


def antimate_rob_jspace(rob):
    plt.rcParams["figure.figsize"] = [7.50, 7.50]
    plt.rcParams["figure.autolayout"] = True

    joints = rob.get_joint_positions()

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)

    line, = plt.plot(joints[:, 0], joints[:, 1], linestyle='-', color = 'blue')

    frames = 100
    trajectory = trajectory_circle_arc(20, 0, 2*pi, frames)
    #trajectory = trajectory_square(10, 10, 10, frames)
    plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='--', color = 'red' )

    # plot workspace
    wspace_inner = trajectory_circle_arc(rob.L[0] - rob.L[1], 0, 2*pi, frames)
    plt.plot(wspace_inner[:, 0], wspace_inner[:, 1], linestyle='--', color='pink')

    wspace_outer = trajectory_circle_arc(rob.L[0] + rob.L[1], 0, 2 * pi, frames)
    plt.plot(wspace_outer[:, 0], wspace_outer[:, 1], linestyle='--', color='pink')

    def animate(i):
        rob.inv(trajectory[i, 0], trajectory[i, 1])
        joints = rob.get_joint_positions()

        ax.clear()
        ax.plot(joints[:, 0], joints[:, 1], linestyle='-', color = 'blue')
        plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='--', color = 'red' )

        plt.plot(wspace_inner[:, 0], wspace_inner[:, 1], linestyle='--', color='pink')
        plt.plot(wspace_outer[:, 0], wspace_outer[:, 1], linestyle='--', color='pink')

        plt.scatter(rob.ee[0], rob.ee[1], color='red')
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, repeat=True)
    plt.show()


def inv_2d_math(x, y):
    L_1 = 15
    L_2 = 10

    a = (x**2+y**2+L_1**2-L_2**2)
    q_y = [0, 0]
    q_y[0] = (4 * y * a + sqrt((4 * y * a) ** 2 - 4 * (4 * y ** 2 + 4*x**2) * (a ** 2 - 4 * L_1 ** 2 * x**2)))/(2*(4*y**2+4*x**2))
    q_y[1] = (4 * y * a - sqrt((4 * y * a) ** 2 - 4 * (4 * y ** 2 + 4*x**2) * (a ** 2 - 4 * L_1 ** 2 * x**2)))/(2*(4*y**2+4*x**2))
    q_x = [sqrt(L_1**2 - q_y[0]**2), sqrt(L_1**2 - q_y[1]**2)]

    # choose correct y based on closest

    q = np.zeros(2)
    q[0] = degrees(asin((y-q_y[1]) / L_2))
    q[1] = degrees(asin(q_y[1] / L_1))

    return q_y, q_x, q


def trajectory_circle_arc(r, start, end, n):
    n_steps = n
    traj = np.zeros([n_steps, 2])
    steps = (end-start) / n_steps
    angles = np.linspace(start, end, n_steps)

    for i in range(0, n_steps):
        traj[i, 0] = r * cos(angles[i]) #cos(i * steps) # x
        traj[i, 1] = r * sin(angles[i]) #sin(i * steps) # y

    return traj


def trajectory_square(x, y, l, n):
    n_steps = n
    traj = np.zeros([n_steps, 2])

    # segment start
    steps = np.round(np.linspace(0, n_steps-1, 5)).astype(int)

    # seg 1
    traj[0:steps[1], 0] = np.linspace(x - l / 2, x + l / 2, steps[1]-steps[0])
    traj[0:steps[1], 1].fill(y-l/2)
    # seg 2
    traj[steps[1]: steps[2], 0].fill(x + l/2)
    traj[steps[1]: steps[2], 1] = np.linspace(y - l / 2, y + l / 2, steps[2]-steps[1])
    # seg 3
    traj[steps[2]: steps[3], 0] = np.linspace(x + l / 2, x - l / 2, steps[3]-steps[2])
    traj[steps[2]: steps[3], 1].fill(y + l/2)
    # seg 4
    traj[steps[3]:steps[4], 0].fill(x-l/2)
    traj[steps[3]:steps[4], 1] = np.linspace(y + l / 2, y - l / 2, steps[4]-steps[3])

    return traj


def test_all_qrts(rob):
    rob.inv(15, 15)
    ax = rob.rob_plt()

    rob.inv(-15, 15)
    rob.rob_plt(ax)

    rob.inv(-15, -15)
    rob.rob_plt(ax)

    rob.inv(15, -15)
    rob.rob_plt(ax)


if __name__ == "__main__":
    rob = Rob2d(q_init=np.array([radians(90), radians(0)]))
    antimate_rob_jspace(rob)
    rob.rob_plt()
    #print(rob.get_q(), rob.get_ee())
    #test_all_qrts(rob)

    plt.show()
