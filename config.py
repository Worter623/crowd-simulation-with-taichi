import numpy as np
import taichi as ti
import random
import math
from PIL import Image

@ti.data_oriented
class Config:
    def __init__(self,path = None):
        self.Astar = 0 # 是否使用A*预渲染地图 1代表是
        self.export_csv = 0 # 是否导出csv文件
        self.im = None
        self.sim_step = 0.01  # 仿真步长
        self.WINDOW_HEIGHT = 200 # 预定义的窗口大小 如果读图片则窗口大小为图片分辨率
        self.WINDOW_WIDTH = 800
        self.window_size = 65  # 每条边划分为多少个格子 这个值调大了之后影响邻域搜索
        self.pixel_number = self.window_size ** 2
        self.search_radius = 20 # 搜索半径 eg.search_radius=2代表在5*5的网格中进行搜索
        self.dynamic_search = 0 #邻域搜索时使用静态分配内存还是动态分配内存 1代表用dynamic node 0代表用静态分配内存方法

        # desired force
        self.relaxation_time = 0.5
        self.goal_threshold = 0.01 # 减速的threshold
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
        self.sigma = 0.2
        self.obstacle_threshold = 0.02 #障碍物/点比较密集时 可能需要调小以取得更好的效果
        self.obstacle_factor = 20

        # flocking
        self.flocking = 0 #为1时才在仿真中加入flocking force
        self.flocking_radius = 0.15
        self.alignment_factor = 10
        self.separation_factor = 10
        self.cohesion_factor = 0

        # steering force
        self.steering_force = 0 #为1时才在仿真中加入steering force
        self.flee_factor = 1
        self.seek_factor = 0
        self.arrival_factor = 1
        self.behind_distance = 0.01
        self.slowing_distance = 0.8
        self.leader_following_factor = 5 # 此实现中默认序号0的人为leader

        # 不读图片时用于设置障碍物的参数 目前只支持通过画线段来设置
        self.resolution = self.window_size+5 #一条障碍线段会被分成多少个障碍点 改邻域搜索后这个值比window_size大其实就可以
        self.obstacles = [[0, 0, 1, 0],[0, 1, 1, 1],[1,0,1,1],[0,0,0,1],[0.5,0,0.5,0.45],[0.5,1,0.5,0.55]] # 线段(startx, starty, endx, endy)
        self.obstacles_x = np.array(self.obstacles)[:,0:2]
        self.obstacles_y = np.array(self.obstacles)[:,2:4]

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

        # render
        self.people_radius = 6
        self.draw_triangle = 0  # 是否在代表人的圆点中画受力三角形
        self.triangle_size = self.people_radius / self.WINDOW_WIDTH 

        self.N = 5000  # people number
        self.max_speed_factor = 0.5
        self.group_target = np.array([[0.1, 0.5]]) #每批人的目的地,[0.9, 0.5]
        self.group_vel = np.array([[-0.01, 0.]]) #每批人的初始速度,[0.01, 0.]
        
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
        """根据N总人数和batch有几批人填充group 即每批人群有多少人"""
        sum = 0
        batch = math.floor(self.N/self.batch)
        for _ in range (self.batch-1):
            self.group.append(batch)
            sum+=batch
        self.group.append(self.N-sum)

    def fill_desiredvel(self):
        """设置每人的期望速度 期望速度*max_speed_factor = 此人的最大速度"""
        for i in range(self.N):
            self.desiredvel[i] = (60+random.randint(0, 15))/100

    def fill_pos_0(self):
        """调整pos_0以设置人群的初始位置"""
        for i in range(self.N):
            self.pos_0[i][1] = 0.5 + random.uniform(-0.4, 0.4)
            if i < self.group[0]:
                self.pos_0[i][0] = 0.9
            else:
                self.pos_0[i][0] = 0.1

    def fill_vel_target(self):
        """工具函数 按照self.group_target和self.group_vel设定每批人的目的地和初始速度"""
        _vels = []
        _desiredpos = []
        for i in range (self.batch):
            for _ in range (self.group[i]):
                _vels.append(self.group_vel[i])
                _desiredpos.append(self.group_target[i])
        self.vel_0 = np.vstack(np.array(_vels))
        self.desiredpos_0 = np.vstack(np.array(_desiredpos))