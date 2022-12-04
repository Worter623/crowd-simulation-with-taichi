import taichi as ti
import numpy as np
from scene import Scene
from config import Config
import time
import utils

node = ti.types.struct(g=ti.float32, h=ti.float32,f=ti.float32, father=ti.int32)

@ti.data_oriented
class AStar:
    def __init__(self,enable,scene):
        self.scene = scene
        self.map = ti.Vector.field(2,dtype=ti.f32) #记录每个像素格子到达目标地点的A*单位方向向量
        ti.root.dense(ti.i, self.scene.batch_size).dense(ti.j,self.scene.pixel_number).place(self.map)
        self.map_done = ti.field(ti.i8)
        ti.root.dense(ti.ij,(self.scene.window_size,self.scene.window_size)).place(self.map_done) # 记录该网格的最短路径是否已经算出 1 = visited

        _list_offset = np.array([(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)])
        self.list_offset = ti.Vector.field(2, ti.i8,shape=8)
        self.list_offset.from_numpy(_list_offset)
        self.dist_list_offset = ti.field(ti.f32,shape = 8)
        self.init_list_offset()

        self.node_matrix = node.field(shape = scene.pixel_number)  # 创建存储节点的矩阵
        self.open_list = ti.field(ti.i32)  # 初始化打开列表 待遍历的节点编号
        self.close_list = ti.field(ti.i32)  # 初始化关闭列表 已经遍历过的节点编号
        ti.root.pointer(ti.i, scene.pixel_number).place(self.open_list)
        ti.root.pointer(ti.i, scene.pixel_number).place(self.close_list)

        self.target = ti.field(ti.i32,scene.batch_size)      #目标地点的计算
        self.group_target = ti.Vector.field(2,ti.i32,scene.batch_size)
        self.group_target.from_numpy(scene.group_target * scene.window_size)
        if enable == 1: #代表希望使用A*预渲染地图
            self.init_map()

    @ti.kernel
    def init_list_offset(self):
        """A*计算时对相邻网格的扩展"""
        for i in self.list_offset:
            self.dist_list_offset[i] = ((self.list_offset[i][0]**2+self.list_offset[i][1]**2)*100)**0.5

    @ti.kernel
    def init_target(self):
        """将scene中记录的标准化后的目标地点坐标[0.9,0.5]根据网格大小扩展成整数"""
        for i in range (self.scene.batch_size):
            self.target[i] = int(self.group_target[i][0]*self.scene.window_size + self.group_target[i][1])
            assert self.scene.obstacle_exist[int(self.group_target[i][0]),int(self.group_target[i][1])] == 0
        

    def init_map(self):
        self.init_target()
        print("targets are nodes:",self.target)
        start = time.time()
        for batch in range (self.scene.batch_size):
            
            self.map_done.fill(0)
            
            for index in range(self.scene.pixel_number):
                print(index)
                #对于每一个地图上的grid算一次A*
                i = ti.floor(index / self.scene.window_size)
                j = index % self.scene.window_size
                if self.scene.obstacle_exist[i,j] == 0 and self.map_done[i,j] == 0:  # 如果在障碍列表，则跳过
                    self.cal_next_loc(index,self.target[batch],batch)
        end = time.time()
        print(start-end)
    
    def cal_next_loc(self,start_pos,target_pos,batch):
        #起始地点=目标地点则直接返回
        if start_pos == target_pos:
            return (0,0)

        #对重复使用的数据结构做初始化（taichi kernel中不支持定义临时的field数据结构）
        self.node_matrix.fill(0)
        self.open_list.fill(0)
        self.close_list.fill(0)
        
        self.next_loc(start_pos,target_pos,batch)

    @ti.kernel
    def next_loc(self,start_pos:ti.i32,target_pos:ti.i32,batch:ti.i32): 
        """
        输入:起始地点的index,目标地点的index
        index计算方法为: index = x * window_size + y,(x,y为坐标,如[350,50],window_size为网格边长)
        使用A*计算起始地点到目标地点的最短路径 
        算出最短路之后,记录路径上所有节点到目标地点的方向向量到self.map中,作为社会力模型中人群的目标力方向向量        
        """
        self.open_list[0] = start_pos # 起始点添加至打开列表
        open_list_len = 1  # 初始化打开列表 待遍历的节点
        close_list_len = 0  # 初始化关闭列表 已经遍历过的节点
        target_x = ti.floor(target_pos / self.scene.window_size)
        target_y = target_pos % self.scene.window_size

        # 开始算法的循环
        while True:
            #  判断是否停止 若目标节点在关闭列表中则停止循环
            if utils.check_in_list(self.close_list,target_pos,close_list_len) == 1:
                break

            now_loc = self.open_list[0]
            place=0           
            #   （1）获取f值最小的点
            for i in range(0, open_list_len): 
                if self.node_matrix[self.open_list[i]].f < self.node_matrix[now_loc].f:
                    now_loc = self.open_list[i]
                    place = i   
            #   （2）切换到关闭列表
            # print("nowloc=",now_loc)
            open_list_len+=-1
            self.open_list[place]=self.open_list[open_list_len]
            self.close_list[close_list_len] = now_loc
            close_list_len+=1  

            grid_x = ti.floor(now_loc / self.scene.window_size)
            grid_y = now_loc % self.scene.window_size
            for i in range(8):#   （3）对3*3相邻格中的每一个
                index_i = grid_x+self.list_offset[i][0]
                index_j = grid_y+self.list_offset[i][1]
                if index_i < 0 or index_i >= self.scene.window_size or index_j < 0 or index_j >= self.scene.window_size: #如果越界则跳过
                    continue
                if self.scene.obstacle_exist[int(index_i),int(index_j)] == 1:  # 如果在障碍列表，则跳过
                    continue
                index = int(index_i * self.scene.window_size + index_j)
                if utils.check_in_list(self.close_list,index,close_list_len):  # 如果在关闭列表，则跳过
                    continue

                #  该节点不在open列表，添加，并计算出各种值
                if not utils.check_in_list(self.open_list,index,open_list_len):
                    self.open_list[open_list_len] = index
                    open_list_len+=1

                    self.node_matrix[index].g = self.node_matrix[now_loc].g +self.dist_list_offset[i]
                    self.node_matrix[index].h = (abs(target_x - index_i)+abs(target_y-index_j))*10 #采用曼哈顿距离
                    self.node_matrix[index].f = (self.node_matrix[index].g +self.node_matrix[index].h)
                    self.node_matrix[index].father = now_loc
                    continue
                #  如果在open列表中，比较，重新计算
                if self.node_matrix[index].g > self.node_matrix[index].g +self.dist_list_offset[i]:
                    self.node_matrix[index].g = self.node_matrix[index].g +self.dist_list_offset[i]
                    self.node_matrix[index].father = now_loc
                    self.node_matrix[index].f = (self.node_matrix[index].g +self.node_matrix[index].h)

        #  找到最短路 依次遍历父节点，找到下一个位置 close列表中的father都可以复用
        next_move = target_pos
        current = self.node_matrix[next_move].father
        while next_move != start_pos:
            next_move = current
            current = self.node_matrix[next_move].father
            index_i = ti.floor(next_move / self.scene.window_size)
            index_j = next_move % self.scene.window_size
            i = int(ti.floor(current / self.scene.window_size))
            j = int(current % self.scene.window_size)
            #  记录去往下一个位置的标准化方向向量
            re = ti.Vector([int(index_i - i), int(index_j-j)])
            self.map[batch,current],_ = utils.normalize(re)
            self.map_done[i,j] = 1
