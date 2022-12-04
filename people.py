import taichi as ti
from pre_astar import AStar
from scene import Scene
import utils
import numpy as np

@ti.data_oriented
class People:
    def __init__(self, config):
        self.config = config
        self.scene = Scene(config)
        print("map init begin")
        self.astar = AStar(config.Astar,self.scene)
        print("map init done")

        self.vel = ti.Vector.field(2, ti.f32)
        self.pos = ti.Vector.field(2, ti.f32)
        self.forces = ti.Vector.field(2, ti.f32)
        self.desiredpos = ti.Vector.field(2, ti.f32)
        self.max_vel = ti.field(ti.f32, self.config.N)
        self.belong_batch = ti.field(ti.i8,self.config.N) # 每个人属于哪一组
        ti.root.dense(ti.i, self.config.N).place(
            self.vel, self.pos, self.desiredpos, self.forces)
            
        self.fill_parm()

        self.color_list = ti.field(ti.i32,self.config.N)
        self.triangle_X = ti.Vector.field(2, ti.f32) # 三个顶点
        self.triangle_Y = ti.Vector.field(2, ti.f32)
        self.triangle_Z = ti.Vector.field(2, ti.f32)
        ti.root.dense(ti.i, self.config.N).place(self.triangle_X,self.triangle_Y,self.triangle_Z)


    def fill_parm(self):
        """init时调用的函数 实现numpy到taichi field的转换 初始化完成后的数据存访都使用taichi field"""
        self.vel.from_numpy(self.config.vel_0)
        self.pos.from_numpy(self.config.pos_0)
        self.desiredpos.from_numpy(self.config.desiredpos_0)
        self.max_vel.from_numpy(
            self.config.desiredvel * self.config.max_speed_factor)
        # 填充每个人属于哪一组
        group = []
        for batch in range (self.config.batch):
            group.append([batch] * self.config.group[batch])
        self.belong_batch.from_numpy(np.hstack(np.array(group)))
        print(self.belong_batch)

    def render(self, gui):
        """taichi gui渲染"""
        if self.config.im is None:
            gui.clear(0xffffff)
            gui.lines(begin=self.config.obstacles_x,end=self.config.obstacles_y, radius=2, color=0xff0000)
        else:
            gui.set_image(self.config.im)
        
        people_centers = self.pos.to_numpy()
        sum = 0
        for i in range (self.config.batch): #分批渲染人
            _color=0x0000ff
            if i % 2 :
                _color = 0x08ff08
            gui.circles(people_centers[sum:sum+self.config.group[i],:],color= _color,radius=self.config.people_radius)
            sum += self.config.group[i]

        if self.config.leader_following_factor != 0 and self.config.steering_force == 1:
            gui.circle(self.pos[i],color= 0x000000,radius=self.config.people_radius)

        # 画三角形
        self.make_triangle()
        if self.config.draw_triangle == 1:
            hex_color = self.color_list.to_numpy()
            triangle_X = self.triangle_X.to_numpy()
            triangle_Y = self.triangle_Y.to_numpy()
            triangle_Z = self.triangle_Z.to_numpy()
            gui.triangles(a=triangle_X, b=triangle_Y, c=triangle_Z,color=hex_color)


    @ti.kernel
    def make_triangle(self):
        """计算出三角形的顶点和颜色 三角形颜色深浅代表受力大小 方向为此刻合力方向"""
        for i in range(self.config.N):
            direction,value = utils.normalize(self.forces[i])
            hex_color = utils.convert_data(value)  
            self.color_list[i] = hex_color
            X = self.pos[i] + direction * self.config.triangle_size * 1.2
            self.triangle_X[i] = X
            per_direction = ti.Vector([direction[1],-direction[0]])
            X_back = self.pos[i] - direction * self.config.triangle_size * 0.8
            self.triangle_Y[i] = X_back + per_direction * self.config.triangle_size *0.6
            self.triangle_Z[i] = X_back - per_direction * self.config.triangle_size *0.6

    @ti.kernel
    def compute_desired_force(self):
        for i in range(self.config.N):
            if self.config.Astar == 1: #A*
                _, dist = utils.normalize(self.desiredpos[i]-self.pos[i])
                _pos2 = ti.floor(self.pos[i] * self.config.window_size , int)
                _pos = int(_pos2[0] * self.config.window_size + _pos2[1])
                direction = self.astar.map[self.belong_batch[i],_pos]
                desired_force = ti.Vector([0.0, 0.0])
                if dist > self.config.goal_threshold:
                    desired_force = direction * self.max_vel[i] - self.vel[i]
                else:
                    desired_force = -1.0 * self.vel[i]
                desired_force /= self.config.relaxation_time
                self.forces[i] += desired_force * self.config.desired_factor
            else: #非A*
                direction, dist = utils.normalize(self.desiredpos[i]-self.pos[i])
                desired_force = ti.Vector([0.0, 0.0])
                if dist > self.config.goal_threshold:
                    desired_force = direction * self.max_vel[i] - self.vel[i]
                else:
                    desired_force = -1.0 * self.vel[i]
                desired_force /= self.config.relaxation_time
                self.forces[i] += desired_force * self.config.desired_factor

    @ti.func
    def cut_val(self,i):
        if self.pos[i][0] >= 1:
            self.pos[i][0] = 0.999
        elif self.pos[i][0] <0:
            self.pos[i][0] = 0
        if self.pos[i][1] >= 1:
            self.pos[i][1] = 0.999
        elif self.pos[i][1] <0:
            self.pos[i][1] = 0

    @ti.kernel
    def update_grid_static(self):
        """
        邻域更新 静态分配内存版本
        """
        # count = 0
        for i in range(self.config.N):
            self.cut_val(i)
            grid_idx = ti.floor(self.pos[i] * self.config.window_size, int)
            self.scene.grid_count[grid_idx] += 1
        #     count +=1
        # assert(count == self.config.N)

        for i in range(self.config.window_size):
            sum = 0
            for j in range(self.config.window_size):
                sum += self.scene.grid_count[i, j]
            self.scene.column_sum[i] = sum

        self.scene.prefix_sum[0, 0] = 0

        ti.loop_config(serialize=True)
        for i in range(1, self.config.window_size):
            self.scene.prefix_sum[i, 0] = self.scene.prefix_sum[i - 1, 0] + self.scene.column_sum[i - 1]

        for i ,j in self.scene.prefix_sum:
            if j == 0:
                self.scene.prefix_sum[i, j] += self.scene.grid_count[i, j]
            else:
                self.scene.prefix_sum[i, j] = self.scene.prefix_sum[i, j - 1] + self.scene.grid_count[i, j]

            linear_idx = i * self.config.window_size + j

            self.scene.list_head[linear_idx] = self.scene.prefix_sum[i, j] - self.scene.grid_count[i, j]
            self.scene.list_cur[linear_idx] = self.scene.list_head[linear_idx]
            self.scene.list_tail[linear_idx] = self.scene.prefix_sum[i, j]

        for i in range(self.config.N):
            grid_idx = ti.floor(self.pos[i] * self.config.window_size, int)
            linear_idx = grid_idx[0] * self.config.window_size + grid_idx[1]
            grain_location = ti.atomic_add(self.scene.list_cur[linear_idx], 1)
            self.scene.particle_id[grain_location] = i

    @ti.kernel
    def update_grid_dynamic(self):
        """
        邻域更新 动态分配内存版本
        """
        for i in range(self.config.N):
            self.cut_val(i)
            grid_idx = ti.floor(self.pos[i] * self.config.window_size, int)
            index = grid_idx[0]*self.config.window_size + grid_idx[1]
            self.scene.grid_count[index] += 1
            ti.append(self.scene.grid_matrix.parent(),index,i)

    def update_grid(self):
        """
        邻域更新+邻域搜索 主要分了静态分配内存(https://zhuanlan.zhihu.com/p/563182093)
        或者动态分配内存(使用dynamic node)的版本
        其中静态分配内存在人群数量多的时候优势显著 (1w人 4FPS VS. 1.25FPS) 
        """
        self.scene.grid_count.fill(0)
        if self.config.dynamic_search == 0: # 静态分配内存
            self.update_grid_static()
            self.search_grid_static()
        else:
            self.scene.block.deactivate_all()
            self.update_grid_dynamic()
            self.search_grid_dynamic()

    @ti.kernel
    def search_grid_dynamic(self):
        """
        邻域搜索: 动态分配内存版本 
        地图网格化 对于每一个网格 更新fij和obstacle force
        时间复杂度O(N) N是地图中的格子数目
        人与人之间的斥力和障碍物斥力都在邻域搜素中完成计算
        """
        for grid in range(self.config.pixel_number):
            grid_x = ti.floor(grid / self.config.window_size)
            grid_y = grid % self.config.window_size
            x_begin = max(grid_x - self.config.search_radius, 0)
            x_end = min(grid_x + self.config.search_radius+1, self.config.window_size)
            y_begin = max(grid_y - self.config.search_radius, 0)
            y_end = min(grid_y + self.config.search_radius+1, self.config.window_size)

            for i in range(self.scene.grid_count[grid]):
                current_people_index = self.scene.grid_matrix[grid,i]
                flocking_count = 0
                alignment_force = ti.Vector([0.0, 0.0])
                separation_force = ti.Vector([0.0, 0.0])
                cohesion_force = ti.Vector([0.0, 0.0])
                for index_i in range(x_begin, x_end):
                    for index_j in range(y_begin, y_end):
                        index = int(index_i * self.config.window_size + index_j)
                        self.compute_obstacle_force(current_people_index,index_i,index_j)
                        for people_index in range(self.scene.grid_count[index]): 
                            other_people_index = self.scene.grid_matrix[index,people_index]
                            if current_people_index < other_people_index:
                                self.compute_fij_force(current_people_index,other_people_index)

                            if self.config.flocking == 1:
                                dist = (self.pos[current_people_index] - self.pos[other_people_index]).norm()
                                if current_people_index != other_people_index and dist < self.config.flocking_radius:
                                    alignment_force += self.vel[other_people_index]
                                    separation_force += (self.pos[current_people_index] - self.pos[other_people_index]) / dist
                                    cohesion_force += self.pos[other_people_index]
                                    flocking_count += 1
                if self.config.flocking == 1 and flocking_count > 0:
                    self.compute_flocking_force(current_people_index,alignment_force,separation_force,cohesion_force,flocking_count)

    @ti.kernel
    def search_grid_static(self):
        """
        邻域搜索: 静态分配内存版本 
        地图网格化 对于每一个网格 更新fij和obstacle force
        时间复杂度O(N) N是地图中的格子数目
        人与人之间的斥力和障碍物斥力都在邻域搜素中完成计算
        """
        for i in range (self.config.N):
            grid_index = ti.floor(self.pos[i]*self.config.window_size,int)
            x_begin = max(grid_index[0] - self.config.search_radius, 0)
            x_end = min(grid_index[0] + self.config.search_radius+1, self.config.window_size)
            y_begin = max(grid_index[1] - self.config.search_radius, 0)
            y_end = min(grid_index[1] + self.config.search_radius+1, self.config.window_size)
            flocking_count = 0
            alignment_force = ti.Vector([0.0, 0.0])
            separation_force = ti.Vector([0.0, 0.0])
            cohesion_force = ti.Vector([0.0, 0.0])
            for index_i in range(x_begin, x_end):
                for index_j in range(y_begin, y_end):
                    search_index = int(index_i * self.config.window_size + index_j)
                    self.compute_obstacle_force(i,index_i,index_j)
                    for p_idx in range(self.scene.list_head[search_index],self.scene.list_tail[search_index]):
                        j = self.scene.particle_id[p_idx]
                        if i < j:
                            self.compute_fij_force(i,j)
                        if self.config.flocking == 1:
                            dist = (self.pos[i] - self.pos[j]).norm()

                            if i != j and dist < self.config.flocking_radius:
                                alignment_force += self.vel[j]
                                separation_force += (self.pos[i] - self.pos[j]) / dist
                                cohesion_force += self.pos[j]
                                flocking_count += 1
            if self.config.flocking == 1 and flocking_count > 0:
                self.compute_flocking_force(i,alignment_force,separation_force,cohesion_force,flocking_count)

                        
    @ti.func
    def compute_flocking_force(self,i,alignment_force,separation_force,cohesion_force,flocking_count):
        """计算flocking force"""
        alignment = utils.limit(
            utils.set_mag((alignment_force / flocking_count), self.max_vel[i]) - self.vel[i],
            self.max_vel[i])
        separation = utils.limit(
            utils.set_mag((separation_force / flocking_count), self.max_vel[i]) - self.vel[i],
            self.max_vel[i]) 
        cohesion = utils.limit(
            utils.set_mag(((cohesion_force / flocking_count) - self.pos[i]), self.max_vel[i]) -
            self.vel[i], self.max_vel[i]) 

        self.forces[i] += alignment * self.config.alignment_factor
        self.forces[i] += separation * self.config.separation_factor
        self.forces[i] += cohesion * self.config.cohesion_factor

    @ti.func
    def compute_fij_force(self,current_people_index,other_people_index):
        """
        计算当前场景中该个体与其他所有人的社会力       
        把force添加到self.forces 整体做截断
        """
        pos_diff = self.pos[current_people_index]-self.pos[other_people_index]
        diff_direction, diff_length = utils.normalize(pos_diff)
        if diff_length < self.config.social_radius:
            vel_diff = self.vel[other_people_index]-self.vel[current_people_index]

            # compute interaction direction t_ij
            interaction_vec = self.config.lambda_importance * vel_diff + diff_direction
            interaction_direction, interaction_length = utils.normalize(
                interaction_vec)

            # compute angle theta (between interaction and position difference vector)
            theta = utils.vector_angles(
                interaction_direction) - utils.vector_angles(diff_direction)
            # compute model parameter B = gamma * ||D||
            B = self.config.gamma * interaction_length

            force_velocity_amount = ti.exp(
                -1.0 * diff_length / B - (self.config.n_prime * B * theta)**2)
            sign_theta = 0.0
            if theta > 0:
                sign_theta = 1.0
            elif theta < 0:
                sign_theta = -1.0
            force_angle_amount = -sign_theta * \
                ti.exp(-1.0 * diff_length / B -(self.config.n * B * theta)**2)

            force_velocity = force_velocity_amount * interaction_direction
            force_angle = ti.Vector([0.0, 0.0])
            force_angle[0] = -force_angle_amount * \
                interaction_direction[1]
            force_angle[1] = force_angle_amount * \
                interaction_direction[0]
            fij = force_velocity + force_angle

            self.forces[current_people_index] += fij*self.config.fij_factor
            self.forces[other_people_index] -= fij*self.config.fij_factor



    @ti.func
    def compute_obstacle_force(self,current_people_index,index_i,index_j):
        """
        计算一个行人(编号为current_people_index)和当前网格下的障碍物(编号为grid_index)之间的斥力
        障碍物位置取网格的中心
        """
        if self.scene.obstacle_exist[index_i,index_j] == 1 :
            diff = self.pos[current_people_index] - self.scene.grid_pos[index_i,index_j]
            directions, dist = utils.normalize(diff)
            dist += -self.config.shoulder_radius
            if dist < self.config.obstacle_threshold:               
                directions = directions * ti.exp(-dist / self.config.sigma)
                self.forces[current_people_index] += directions*self.config.obstacle_factor

    @ti.kernel
    def compute_steering_force(self):
        flee_target = self.pos[0]
        follower_target = self.calculate_follow_position()
        for i in range(self.config.N):
            # steering forces seek: pass through the target, and then turn back to approach again
            seek_acc = utils.limit(
                utils.set_mag((self.desiredpos[i]-self.pos[i]), self.max_vel[i]) - self.vel[i],
                self.max_vel[i]) 

            #flee forces:
            flee_acc = utils.limit(
                utils.set_mag((self.pos[i]-self.desiredpos[i]), self.max_vel[i]) - self.vel[i],
                self.max_vel[i]) 

            #arrival forces：
            target_offset = self.desiredpos[i]-self.pos[i]
            distance = target_offset.norm()
            norm_desired_speed = utils.set_mag(target_offset,self.max_vel[i])
            desired_speed = norm_desired_speed if distance >= self.config.slowing_distance else norm_desired_speed * distance / self.config.slowing_distance            
            arrival_acc = (desired_speed - self.vel[i])

            #leader following forces:
            follower_acc = ti.Vector([0,0])
            if (self.pos[i]-flee_target).norm() < self.config.behind_distance :
                #flee forces:
                follower_acc = utils.limit(
                    utils.set_mag((self.pos[i]-flee_target), self.max_vel[i]) - self.vel[i],
                    self.max_vel[i]) * self.config.flee_factor
            else: 
                target_offset = follower_target-self.pos[i]
                distance = target_offset.norm()
                norm_desired_speed = utils.set_mag(target_offset,self.max_vel[i])
                desired_speed = norm_desired_speed if distance >= self.config.slowing_distance else norm_desired_speed * distance / self.config.slowing_distance
                follower_acc = (desired_speed - self.vel[i])*15

            if self.config.leader_following_factor != 0:
                if i == 0:
                    self.forces[i] += arrival_acc * self.config.arrival_factor
                else:
                    #对于跟随者
                    self.forces[i] += follower_acc * self.config.leader_following_factor
            else:
                self.forces[i] += arrival_acc * self.config.arrival_factor
                self.forces[i] += flee_acc * self.config.flee_factor
                self.forces[i] += seek_acc * self.config.seek_factor  

    def make_force(self):
        self.forces.fill(0)
        self.update_grid() 
        self.compute_desired_force()
        if self.config.steering_force == 1:
            self.compute_steering_force()

    @ti.kernel
    def update(self):
        for i in range(self.config.N):
            new_vel = self.vel[i] + self.config.sim_step * self.forces[i]
            new_vel = utils.capped(new_vel, self.max_vel[i])
            # # if close enough to goal, stop
            # destination_vector = self.desiredpos[i] - self.pos[i]
            # _, dist = normalize(destination_vector)
            # if dist < self.config.stop_radius:
            #     new_vel = ti.Vector([0.0,0.0])

            self.pos[i] += new_vel * self.config.sim_step
            self.vel[i] = new_vel
            if self.pos[i][1] > 1 or self.pos[i][1] <0 :
                print("out!!!!!,i=",i,self.pos[i])

    @ti.kernel
    def print_id(self,mouse_x:ti.f32,mouse_y:ti.f32):
        mouse = ti.Vector([mouse_x,mouse_y])
        for i in range(self.config.N):
            diff = self.pos[i] - mouse
            _,dist = utils.normalize(diff)
            if (dist < 0.05):
                print("selecting people:",self.pos[i],i)

    @ti.func
    def calculate_follow_position(self):
        # assumes that only leader will trriger this function, so we take vel[0] and pos[0]
        tv = -self.vel[0]
        tv = tv / tv.norm() * self.config.behind_distance
        behind = self.pos[0] + tv
        return behind
