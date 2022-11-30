import math
import time
import taichi as ti
import numpy as np
from scene import Scene
from config import Config

def normalize(vec):
    """向量标准化"""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec / norm
    return new_vec


class Node:
    def __init__(self):
        #  初始化各个坐标点的g值、h值、f值、父节点
        self.g = 0
        self.h = 0
        self.f = 0
        self.father = (0, 0)


class AStar:
    def __init__(self,enable,scene):
        self.scene = scene
        self.window_size = scene.window_size
        self.map = ti.Vector.field(2,dtype=ti.f32) #记录每个像素格子到达目标地点的A*单位方向向量
        ti.root.dense(ti.ij, (self.scene.window_size,self.scene.window_size)).place(self.map)
        self.map_done = ti.field(ti.i8)
        ti.root.dense(ti.ij,(self.scene.window_size,self.scene.window_size)).place(self.map_done) # 记录该网格的最短路径是否已经算出 1 = visited

        target = scene.group_target * scene.window_size
        self.aim_loc = (int(target[0][0]),int(target[0][1]))
        print(self.aim_loc,"is current target")

        if enable:
            start = time.time()
            for i in range(self.scene.window_size):
                for j in range(self.scene.window_size):
                    if scene.obstacle[i,j] == 0 and self.map_done[i,j] == 0:  # 如果在障碍列表，则跳过
                        start_loc = (i,j)
                        print(start_loc)
                        self.next_loc(start_loc,self.aim_loc)
            end = time.time()
            print(start-end)


    def next_loc(self,start_loc,aim_loc):
        if start_loc == aim_loc:
            return 
        # 初始化各种状态
        open_list = []  # 初始化打开列表 待遍历的节点
        close_list = []  # 初始化关闭列表 已经遍历过的节点   
        # 创建存储节点的矩阵
        node_matrix = [[0 for i in range(self.window_size)] for i in range(self.window_size)]
        for i in range(0, self.window_size):
            for j in range(0, self.window_size):
                node_matrix[i][j] = Node()

        open_list.append(start_loc)  # 起始点添加至打开列表
        # 开始算法的循环
        while True:
            now_loc = open_list[0]
            for i in range(1, len(open_list)):  # （1）获取f值最小的点
                if node_matrix[open_list[i][0]][open_list[i][1]].f < node_matrix[now_loc[0]][now_loc[1]].f:
                    now_loc = open_list[i]
            #   （2）切换到关闭列表
            open_list.remove(now_loc)
            close_list.append(now_loc)
            #  （3）对相邻格中的每一个
            list_offset = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)]
            for temp in list_offset:
                temp_loc = (now_loc[0] + temp[0], now_loc[1] + temp[1])

                if temp_loc[0] < 0 or temp_loc[0] >= self.window_size or temp_loc[1] < 0 or temp_loc[1] >= self.window_size:
                    continue
                if self.scene.obstacle[temp_loc[0],temp_loc[1]] == 1:  # 如果在障碍列表，则跳过
                    continue
                if temp_loc in close_list:  # 如果在关闭列表，则跳过
                    continue

                #  该节点不在open列表，添加，并计算出各种值
                if temp_loc not in open_list:
                    open_list.append(temp_loc)
                    node_matrix[temp_loc[0]][temp_loc[1]].g = (node_matrix[now_loc[0]][now_loc[1]].g +
                                                             int(((temp[0]**2+temp[1]**2)*100)**0.5))
                    node_matrix[temp_loc[0]][temp_loc[1]].h = (abs(aim_loc[0]-temp_loc[0])
                                                               + abs(aim_loc[0]-temp_loc[1]))*10
                    node_matrix[temp_loc[0]][temp_loc[1]].f = (node_matrix[temp_loc[0]][temp_loc[1]].g +
                                                               node_matrix[temp_loc[0]][temp_loc[1]].h)
                    node_matrix[temp_loc[0]][temp_loc[1]].father = now_loc
                    continue

                #  如果在open列表中，比较，重新计算
                if node_matrix[temp_loc[0]][temp_loc[1]].g > (node_matrix[now_loc[0]][now_loc[1]].g +
                                                             int(((temp[0]**2+temp[1]**2)*100)**0.5)):
                    node_matrix[temp_loc[0]][temp_loc[1]].g = (node_matrix[now_loc[0]][now_loc[1]].g +
                                                             int(((temp[0]**2+temp[1]**2)*100)**0.5))
                    node_matrix[temp_loc[0]][temp_loc[1]].father = now_loc
                    node_matrix[temp_loc[0]][temp_loc[1]].f = (node_matrix[temp_loc[0]][temp_loc[1]].g +
                                                               node_matrix[temp_loc[0]][temp_loc[1]].h)

            #  判断是否停止
            if aim_loc in close_list:
                break

        #  依次遍历父节点，找到下一个位置
        next_move = aim_loc
        current = node_matrix[next_move[0]][next_move[1]].father
        while current != start_loc:
            next_move = current
            current = node_matrix[next_move[0]][next_move[1]].father
            #  返回下一个位置的方向向量，例如：（-1,0），（-1,1）......
            re = ti.Vector([next_move[0] - current[0], next_move[1] - current[1]])
            self.map[current[0],current[1]] = normalize(re)
            self.map_done[current[0],current[1]] = 1
        return

ti.init(arch=ti.cpu, debug=True, kernel_profiler=True)#,cpu_max_num_threads=1
config = Config("./map/testpic.png") #输入图片路径 不输代表不读图片,用config中的默认配置
scene = Scene(config)
target = scene.group_target * scene.window_size
aim_loc = (int(target[0][0]),int(target[0][1]))
start = time.time()
a = AStar(0,scene)
# for i in range (150):
#     for j in range(150):
#         if scene.obstacle[i,j] == 0 :  # 如果在障碍列表，则跳过
#             print(i,j)

a.next_loc((96,51),aim_loc)


end = time.time()
print(start-end)
