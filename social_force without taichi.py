"""
version :1.0 
无邻域搜索
"""


import taichi as ti
import numpy as np
import random
import time

#TODO:add group_force
# desired relaxation_time 0.5
# desired goal_threshold 0.2
# desired factor 1.0
# social force factor 5.1
# obstacle sigma: 0.2
# obstacle threshold: 3.0
# obstacle factor 10.0


class Config:
    def __init__(self):  # TODO: 传入DICT[str]int，动态指定某些参数
        self.sim_step = 0.01  # 仿真步长
        self.WINDOW_HEIGHT = 200
        self.WINDOW_WIDTH = 800
        self.window_size = 20  # 每行/列有多少个小格子 对正方形地图更友好
        self.pixel_number = self.window_size**2
        self.grid_space = 1 / self.window_size

        # render
        self.people_radius = 10
        self.base_radius = 0.2

        # desired force
        self.relaxation_time = 0.5
        self.goal_threshold = 0.01
        self.stop_radius = 0.1  # 比较接近目标就停下来
        self.desired_factor = 10

        # fij
        self.social_radius = 0.05
        self.lambda_importance = 2.0
        self.gamma = 0.35
        self.n = 2
        self.n_prime = 3
        self.fij_factor = 10

        # obstacle force
        self.shoulder_radius = 0.001
        self.resolution = 1000
        # self.obstacles = [[0, 1, 0, 0],[0, 1, 1, 1],[0,0,0,1],[1,1,1,0],[0.5,0.5,0,0.45],[0.5,0.5,0.55,1]]
        # self.obstacles_x = np.array([[0, 1],[0, 1],[0,0],[1,1],[0.5,0],[0.5,0.55]])
        # self.obstacles_y = np.array([[0, 0],[1, 1],[1,0],[1,0],[0.5,0.45],[0.5,1]])
        self.obstacles = [[0, 1, 0, 0],[0, 1, 1, 1],[0,0,0,1],[1,1,1,0]]
        self.obstacles_x = np.array([[0, 1],[0, 1],[0,0],[1,1]])
        self.obstacles_y = np.array([[0, 0],[1, 1],[1,0],[1,0]])
        self.sigma = 0.2
        self.obstacle_threshold = 0.01
        self.obstacle_factor = 10

        self.max_speed_factor = 0.5

        self.N = 10000  # people number TODO:odd
        self.vel_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
        self.fill_vel_0()
        self.pos_0 = np.zeros(shape=(self.N, 2), dtype=np.float32)
        self.fill_pos_0()
        self.desiredpos_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
        self.fill_desiredpos_0()
        self.desiredvel = np.zeros(dtype=np.float32, shape=self.N)  # 期望速度大小m/s
        self.fill_desiredvel()

    def fill_desiredvel(self):
        for i in range(self.N):
            self.desiredvel[i] = (60+random.randint(0, 15))/100

    def fill_pos_0(self):
        for i in range(self.N):
            self.pos_0[i][1] = 0.5 + random.uniform(-0.5, 0.5)
            if i >= self.N/2:
                self.pos_0[i][0] = 1

    def fill_vel_0(self):
        vel_01 = np.array([(0.01, 0.)]*int(self.N/2), dtype=np.float32)
        vel_02 = np.array([(-0.01, 0.)]*int(self.N/2), dtype=np.float32)
        self.vel_0 = np.vstack((vel_01, vel_02))

    def fill_desiredpos_0(self):
        desiredpos_01 = np.array([(1., 0.5)]*int(self.N/2), dtype=np.float32)
        desiredpos_02 = np.array([(0., 0.5)]*int(self.N/2), dtype=np.float32)
        self.desiredpos_0 = np.vstack((desiredpos_01, desiredpos_02))



def normalize(vec):
    """向量标准化"""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec / norm
    return new_vec, norm



def capped(vec, limit):
    """Scale down a desired velocity to its capped speed."""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec * min(1, limit/norm)
    return new_vec


def vector_angles(vec):
    """Calculate angles for an array of vectors  = atan2(y, x)"""
    return ti.atan2(vec[1], vec[0])


class Scene:
    def __init__(self, obstacles, resolution):
        self.obstacle_number = 0
        self._obstacles = []
        self.init_obstacles(obstacles, resolution)
        self._obstacles = np.vstack(self._obstacles)
        self.obstacles = ti.Vector.field(2, ti.f32)
        ti.root.dense(ti.i, len(self._obstacles)).place(self.obstacles)
        self.obstacles.from_numpy(self._obstacles)
        self.obstacle_number = self._obstacles.shape[0]
        print("obstacles=", self.obstacles)
        print("obstacle number = ", self.obstacle_number)

    def init_obstacles(self, obstacles, resolution):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacles is None:
            self._obstacles = []
        else:
            self._obstacles = []
            for startx, endx, starty, endy in obstacles:
                samples = int(np.linalg.norm(
                    (startx - endx, starty - endy)) * resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples),
                            np.linspace(starty, endy, samples))
                    )
                )
                self._obstacles.append(line)
                self.obstacle_number += 1



class People:
    def __init__(self, config):
        self.config = config
        self.scene = Scene(config.obstacles, config.resolution)

        self.vel = ti.Vector.field(2, ti.f32)
        self.pos = ti.Vector.field(2, ti.f32)
        self.forces = ti.Vector.field(2, ti.f32)
        self.desiredpos = ti.Vector.field(2, ti.f32)
        self.max_vel = ti.field(ti.f32, self.config.N)
        self.vel_value = ti.field(ti.f32, self.config.N)
        ti.root.dense(ti.i, self.config.N).place(
            self.vel, self.pos, self.desiredpos, self.forces)
        self.grid_matrix = ti.field(ti.i32)
        self.grid_count = ti.field(ti.i32, self.config.pixel_number)
        self.block = ti.root.dense(ti.i, self.config.pixel_number)
        pixel = self.block.dynamic(ti.j, self.config.N)
        pixel.place(self.grid_matrix)

        self.fill_parm()

    def fill_parm(self):
        """init时调用的函数 实现numpy到taichi field的转换 初始化完成后的数据存访都使用taichi field"""
        self.vel.from_numpy(self.config.vel_0)
        self.pos.from_numpy(self.config.pos_0)
        self.desiredpos.from_numpy(self.config.desiredpos_0)
        self.max_vel.from_numpy(
            self.config.desiredvel * self.config.max_speed_factor)

    def render(self, gui):
        """taichi gui渲染 只渲染people"""
        people_centers = self.pos.to_numpy()
        self.make_vel()
        people_radius = self.vel_value.to_numpy()*self.config.people_radius
        gui.circles(np.array([[0.01, 0.5], [1, 0.5]]), color=0x0000ff,
                    radius=self.config.people_radius)
        gui.lines(begin=self.config.obstacles_x,
                  end=self.config.obstacles_y, radius=2, color=0xff0000)
        gui.circles(people_centers, color=0x000000,
                    radius=people_radius)


    def make_vel(self):
        for i in range(self.config.N):
            self.vel_value[i] = self.vel[i].norm() + self.config.base_radius

    def clean_forces(self):
        for i in range(self.config.N):
            self.forces[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_desired_force(self):
        """Calculates the force between this agent and the next assigned waypoint.
        If the waypoint has been reached, the next waypoint in the list will be selected.
        把force添加到self.forces 整体做截断"""
        for i in range(self.config.N):
            direction, dist = normalize(self.desiredpos[i]-self.pos[i])
            desired_force = ti.Vector([0.0, 0.0])
            if dist > self.config.goal_threshold:
                desired_force = direction * self.max_vel[i] - self.vel[i]
            else:
                desired_force = -1.0 * self.vel[i]
            desired_force /= self.config.relaxation_time
            self.forces[i] += desired_force * self.config.desired_factor
            # if i==99:
            #     print("desired_force=",desired_force * self.config.desired_factor)


    def update_grid(self):
        """邻域更新-更新Grid_matrix ; grid_count 
        TODO:在GPU上动态扩展内存比较耗时 可以改成静态分配内存 https://zhuanlan.zhihu.com/p/563182093
        """
        for i in range(self.config.pixel_number):
            self.grid_count[i] = 0

        for i in range(self.config.N):
            x = ti.floor(self.pos[i][0] / self.config.grid_space)
            y = ti.floor(self.pos[i][1] / self.config.grid_space)            
            if x < 0:
                x = 0
            elif x >= self.config.window_size:
                x = self.config.window_size-1
            if y < 0:
                y = 0
            elif y >= self.config.window_size:
                y = self.config.window_size-1
            index = int(x*self.config.window_size+y)
            ti.append(self.grid_matrix.parent(), index, i)
            self.grid_count[index] += 1

    def compute_fij_force(self):
        """计算当前场景中该个体与其他所有人的社会力 
        把force添加到self.forces 整体做截断
        TODO:目前的算法是O(N^2)的 改邻域搜索可以O(N)
        """
        for i in range (self.config.N):
            for j in range(self.config.N):
                if i<j:#O(N^2)
                    pos_diff = self.pos[i]-self.pos[j]
                    diff_direction, diff_length = normalize(pos_diff)
                    if diff_length >= self.config.social_radius:
                        continue
                    vel_diff = self.vel[j]-self.vel[i]

                    # compute interaction direction t_ij
                    interaction_vec = self.config.lambda_importance * vel_diff + diff_direction
                    interaction_direction, interaction_length = normalize(interaction_vec)

                    # compute angle theta (between interaction and position difference vector)
                    theta = vector_angles(interaction_direction) - vector_angles(diff_direction)
                    # compute model parameter B = gamma * ||D||
                    B = self.config.gamma * interaction_length

                    force_velocity_amount = ti.exp(-1.0 * diff_length / B - (self.config.n_prime * B * theta)**2)
                    sign_theta = 0.0
                    if theta > 0:
                        sign_theta = 1.0
                    elif theta < 0:
                        sign_theta = -1.0
                    force_angle_amount = -sign_theta * ti.exp(-1.0 * diff_length / B - (self.config.n * B * theta)**2)

                    force_velocity = force_velocity_amount * interaction_direction
                    force_angle = ti.Vector([0.0,0.0])
                    force_angle[0] = -force_angle_amount * interaction_direction[1]
                    force_angle[1] = force_angle_amount * interaction_direction[0]
                    fij = force_velocity + force_angle

                    self.forces[i] += fij*self.config.fij_factor
                    self.forces[j] -= fij*self.config.fij_factor


    def compute_obstacle_force(self):
        """Calculates the force between this agent and the nearest obstacle in this scene"""
        #TODO: 改邻域搜索
        threshold = self.config.obstacle_threshold + self.config.shoulder_radius
        for i in range(self.config.N):
            # a=ti.Vector([0.0,0.0])
            for j in range(self.scene.obstacle_number):
                diff = self.pos[i] - self.scene.obstacles[j]
                directions, dist = normalize(diff)
                dist += -self.config.shoulder_radius
                if dist >= threshold:
                    continue
                directions = directions * ti.exp(-dist / self.config.sigma)
                self.forces[i] += directions*self.config.obstacle_factor
                # if i==0:
                #     a+=directions*self.config.obstacle_factor
            # if i==99:             
            #     print("obstacle_force=",a)

    def make_force(self):
        self.clean_forces()
        # self.block.deactivate_all()
        # self.update_grid()
        # print(self.grid_count)
        self.compute_fij_force()
        self.compute_desired_force()
        self.compute_obstacle_force()


    def update(self):
        for i in range(self.config.N):
            new_vel = self.vel[i] + self.config.sim_step * self.forces[i]
            new_vel = capped(new_vel, self.max_vel[i])
            # # if close enough to goal, stop
            # destination_vector = self.desiredpos[i] - self.pos[i]
            # _, dist = normalize(destination_vector)
            # if dist < self.config.stop_radius:
            #     new_vel = ti.Vector([0.0,0.0])

            self.pos[i] += new_vel * self.config.sim_step
            self.vel[i] = new_vel


if __name__ == "__main__":
    ti.init(arch=ti.cpu, debug=True, kernel_profiler=True,cpu_max_num_threads=1)
    start = time.time()
    config = Config()
    gui = ti.GUI("social force model", res=(
        config.WINDOW_WIDTH, config.WINDOW_HEIGHT))

    people = People(config)
    print("------------")
    print(people.pos)
    print(people.max_vel)
    print("------------")


    end = time.time()
    print("initialize time = ", end-start)


    #while gui.running:
    for i in range (100):
        gui.clear(0xffffff)
        people.make_force()
        people.update()
        # print(i)
        # print(people.pos)
        # print(people.vel)
        people.render(gui)
        gui.show()

    endend = time.time()
    print("simulation time = ", endend-end)

