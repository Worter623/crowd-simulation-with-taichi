import taichi as ti
import numpy as np
import math
from PIL import Image

@ti.data_oriented
class Scene:
    def __init__(self, config):
        self.window_size = config.window_size #网格边长
        self.pixel_number = config.pixel_number #网格大小  pixel_number = window_size^2
        self.batch_size = config.batch
        self.group_target = config.group_target #每批人的目的地

        self.obstacle_exist = ti.field(ti.f32, shape=(self.window_size, self.window_size))  # 记录每个网格中存不存在障碍物（静态
        self.grid_pos = ti.Vector.field(2, ti.float32)  # 对于障碍物网格 记录每个网格的中心位置 (静态 sparse)
        ti.root.pointer(ti.ij,(self.window_size,self.window_size)).place(self.grid_pos)

        if config.dynamic_search == 0: #静态分配内存
            self.grid_count = ti.field(dtype=ti.i32,shape=(self.window_size, self.window_size))  # 记录每个网格中的人数
            self.list_head = ti.field(dtype=ti.i32, shape=self.pixel_number)
            self.list_cur = ti.field(dtype=ti.i32, shape=self.pixel_number)
            self.list_tail = ti.field(dtype=ti.i32, shape=self.pixel_number)
            self.column_sum = ti.field(dtype=ti.i32, shape=self.window_size, name="column_sum") 
            self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.window_size, self.window_size), name="prefix_sum")
            self.particle_id = ti.field(dtype=ti.i32, shape=config.N, name="particle_id")
        else:
            self.grid_count = ti.field(ti.i32, self.pixel_number)  # 记录每个网格中的人数
            self.grid_matrix = ti.field(ti.i32)  # 记录每个网格中的人的编号
            self.block = ti.root.dense(ti.i, self.pixel_number)
            pixel = self.block.dynamic(ti.j, config.N)
            pixel.place(self.grid_matrix)

        if config.im is None:
            self.init_obstacles(config)
        else:
            self.init_from_pic(config)
        self.make_obstacle_pos()

        #for pure python version
        self.obstacle = self.obstacle_exist.to_numpy()


    def init_obstacles(self, config):
        """Input an list of (startx, starty, endx, endy) as start and end of a line"""
        _obstacles = []
        for startx, starty, endx, endy in config.obstacles:
            samples = int(np.linalg.norm(
                (startx - endx, starty - endy)) * config.resolution)
            line = np.array(
                list(
                    zip(np.linspace(startx, endx, samples),
                        np.linspace(starty, endy, samples))
                )
            )
            _obstacles.append(line)
        _obstacles = np.vstack(_obstacles)
        for obstacle in _obstacles:
            x = math.floor(obstacle[0] * self.window_size)
            y = math.floor(obstacle[1] * self.window_size)
            if x >= self.window_size:
                x = self.window_size-1
            if y >= self.window_size:
                y = self.window_size-1
            if self.obstacle_exist[x,y] == 0:
                self.obstacle_exist[x,y] = 1
           
    def init_from_pic(self,config):
        """
        读图片 二值化 自动识别障碍物
        PIL读入的图片坐标系和taichi不同 需要逆时针旋转90°
        """
        im = np.array(Image.open(config.path).convert('1'))
        im = np.rot90(im, -1)
        ti_im = ti.field(ti.i8, shape=(config.WINDOW_WIDTH,config.WINDOW_HEIGHT))
        ti_im.from_numpy(im.astype(np.int8))
        self.make_pos_with_pic(ti_im,config.WINDOW_WIDTH,config.WINDOW_HEIGHT)       

    @ti.kernel
    def make_pos_with_pic(self,ti_im:ti.template(),WINDOW_WIDTH:ti.i32,WINDOW_HEIGHT:ti.i32):
        """
        对于传入图片的每个像素 判别是不是障碍物(0代表图片中的黑色网格) 
        若是 通过x坐标范围(x,x+1)y坐标范围(y,y+1) 算出它被拓展之后对应的grid范围(反采样) 
        将该范围的grid标为有障碍物(1)
        """
        for i,j in ti_im:
            if ti_im[i,j] == 0:
                start_i = ti.round(i / WINDOW_WIDTH * self.window_size)
                end_i = ti.round((i+1) / WINDOW_WIDTH * self.window_size)
                start_j = ti.round(j / WINDOW_HEIGHT* self.window_size)
                end_j = ti.round((j+1) / WINDOW_HEIGHT * self.window_size)
                for x in range (start_i,end_i):
                    for y in range(start_j,end_j):
                        self.obstacle_exist[x,y] = 1

    @ti.kernel
    def make_obstacle_pos(self):
        """
        对于障碍物网格 计算出所有grid的中心位置 以便后续调用
        对于非障碍物网格 A*出它的方向向量 理想情况下人不会撞墙 因此不会需要访问障碍物网格的A*结果
        """
        for grid_index in range(self.pixel_number):
            i = int(ti.floor(grid_index/ self.window_size))
            j = int(grid_index % self.window_size)
            if self.obstacle_exist[i,j] == 1 :
                self.grid_pos[i,j] = ti.Vector([i,j])/ self.window_size