import numpy as np
import taichi as ti
import random
import math
from PIL import Image

@ti.data_oriented
class Config:
    def __init__(self,path = None):
        self.Astar = 1 # 是否使用A*预渲染地图 1代表是
        self.im = None
        self.sim_step = 0.01  # 仿真步长
        self.WINDOW_HEIGHT = 200
        self.WINDOW_WIDTH = 800
        self.window_size = 50  # 这个值调大了之后影响邻域搜索
        self.pixel_number = self.window_size ** 2
        self.search_radius = 50 # 搜索半径 2代表在5*5的网格中进行搜索
        self.dynamic_search = 1 #邻域搜索时使用静态分配内存还是动态分配内存 1代表用dynamic node 0代表用静态分配内存方法

        # render
        self.people_radius = 5
        self.base_radius = 0.5

        # desired force
        self.relaxation_time = 0.5
        self.goal_threshold = 0.01 # 减速的threshold
        self.stop_radius = 0.1  # 比较接近目标就停下来 废弃
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
        self.resolution = self.window_size+5 #一条障碍线段会被分成多少个障碍点 改邻域搜索后这个值比window_size大其实就可以
        self.obstacles = [[0, 0, 1, 0],[0, 1, 1, 1],[1,0,1,1],[0,0,0,1]] # 线段(startx, starty, endx, endy),[0.5,0,0.5,0.45],[0.5,1,0.5,0.55]
        self.obstacles_x = np.array(self.obstacles)[:,0:2]
        self.obstacles_y = np.array(self.obstacles)[:,2:4]
        self.sigma = 0.2
        self.obstacle_threshold = 0.03
        self.obstacle_factor = 10

        if not (path is None):
            # read img 
            self.path = path
            
            im = np.array(Image.open(path).convert('L'))
            im = np.rot90(im,-1) # PIL读入的图片坐标系和taichi不同 需要逆时针旋转90°
            self.WINDOW_WIDTH = im.shape[0]
            self.WINDOW_HEIGHT = im.shape[1]
            self.im = ti.field(ti.f32, shape=im.shape)
            self.im.from_numpy(im)
            print("reading pic from {}, pic resolution in pixel is {}*{}".format(path,self.WINDOW_WIDTH,self.WINDOW_HEIGHT))
            self.window_size = max(self.WINDOW_WIDTH,self.WINDOW_HEIGHT) +50
            self.pixel_number = self.window_size ** 2

        self.N = 50  # people number
        self.max_speed_factor = 0.5
        self.group_target = np.array([[0.1, 0.5],[0.9, 0.5]]) #每批人的目的地
        self.group_vel = np.array([[-0.01, 0.],[0.01, 0.]]) #每批人的初始速度
        
        self.batch = self.group_target.shape[0] #有几批人
        self.group = [] #每批的人数
        self.fill_group()

        self.vel_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
        self.pos_0 = np.zeros(shape=(self.N, 2), dtype=np.float32)
        self.desiredpos_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
        self.desiredvel = np.zeros(dtype=np.float32, shape=self.N)  # 期望速度大小m/s
        self.fill_vel_target()
        self.fill_pos_0()
        self.fill_desiredvel()
        
    def fill_group(self):
        sum = 0
        batch = math.floor(self.N/self.batch)
        for _ in range (self.batch-1):
            self.group.append(batch)
            sum+=batch
        self.group.append(self.N-sum)

    def fill_desiredvel(self):
        for i in range(self.N):
            self.desiredvel[i] = (60+random.randint(0, 15))/100

    def fill_pos_0(self):
        for i in range(self.N):
            self.pos_0[i][1] = 0.5 + random.uniform(-0.4, 0.4)
            if i < self.group[0]:
                self.pos_0[i][0] = 0.9
            else:
                self.pos_0[i][0] = 0.1

    def fill_vel_target(self):
        _vels = []
        _desiredpos = []
        for i in range (self.batch):
            for _ in range (self.group[i]):
                _vels.append(self.group_vel[i])
                _desiredpos.append(self.group_target[i])
        self.vel_0 = np.vstack(np.array(_vels))
        self.desiredpos_0 = np.vstack(np.array(_desiredpos))