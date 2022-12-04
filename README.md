# crowd-simulation-with-taichi
crowd simulation algorithms implemented with taichi

a work for taichi hackathon



## 项目设计

### 选题背景

在许多方面上，基于agent的人群模拟算法与基于粒子的物理仿真算法是非常类似的，具体来说可以归纳为三点：

- 粒子性，人群中的每个agent都能够自主行动，并被赋予一组物理属性；
- 物理限制，每个agent都遵循一定的物理运动规律，如动力学规律；
- 空间情境，人群在一定的空间区域中移动，同样会出现许多spatial sparsity的场景。

当前人群仿真模拟项目的难点在于计算机算力对人群规模、仿真效率的限制。taichi 以高性能实现基于粒子的流体仿真而闻名，应当也能够大幅度提升人群行为仿真模拟的性能。

### 现有功能

实现一个人群行为仿真库，能够通过配置(config.py)指定相应的人群仿真算法、仿真参数、模拟场景、路径规划方式。

本项目目前涵盖的功能：

* 人群算法

  * steering behavior
  * social force
  * ~~基于人格的模型，如OCEAN~~（还没有实现！打脸啦

* 障碍场景模拟

  * 可通过参数指定：目前仅支持在配置中以(startx, starty, endx, endy)这样的四元组形式指定多条线段的起点、终点
  * 可传入图片指定：二值图片，如map文件夹中示例

  <img src="README.assets/map maze.gif" alt="map maze" style="zoom: 50%;" />
  
  <center><p>N=100, resolution = 65 ,social force + A*, in map maze </p></center>

* 路径规划

  * A*（在仿真开始前预渲染）：

    目前的算法主要是通过将顺序结构A*中这两个步骤并行化，去提升算法的性能：

    1）从open列表中提取出优先级最高的节点q

    2）扩展q的相邻节点

* 结果呈现

  * 通过taichi GUI展现仿真结果

      <center class="half">
        <img src="./demo/alignment=seperation=10.gif" width="200"/>
        <img src="./demo/alignment=cohension=10.gif" width="200"/>
      </center>
      <center><p>seperation+alignment ; cohension+alignment </p></center>
      <center>
      <img src="./demo/leader following.gif" width="400"/>
      </center>
      <center><p>leader following </p></center>
  
  * 将数据导出到其他渲染引擎中，完成基于数据驱动的人群行为仿真：
  
    在配置中指定导出人群位置的csv文件，通过data table导入UE
  <center class="half">
    <img src="./demo/export to UE.gif" width="300"/>
    <img src="./demo/export to UE-2.gif" width="300"/>
    <img src="./demo/src for UE.gif" width="300"/>
  </center>
  
  
  <center><p>N=100, resolution = 100, social force + A*,export to UE </p></center>

### 性能对比：

run on Intel(R) Core(TM) i5-8265U CPU

- 人群运动仿真算法使用taichi得到的性能提升：


| 人群规模 | with taichi | without taichi |
| ----------------- | ---------------- | ---------------- |
| N=100, 500 frame, run on RTX 3070GPU | 9.06s     | 31.04s    |
| N=1000,100 frame, run on RTX 3070GPU | 2.229s      | 304.15s |

- 分别计算10,000个pixel 到一个target pixel的A*，实现的时间分别为

| with taichi | without taichi |
| ----------- | -------------- |
| 29.5s       | 420s           |

- 人群算法中（flocking, social force-force between agents) 邻域搜索使用dynamic node或静态分配内存方式([参照胡老师的实现](https://zhuanlan.zhihu.com/p/563182093))，在紧凑场景中（所有人挤在一个小房间），不同人群规模下的帧率为

| 人群规模          | 使用dynamic node | 静态分配内存方式 |
| ----------------- | ---------------- | ---------------- |
| N=100, run on CPU | 29.6FPS          | 28.5FPS          |
| N=800, run on CPU | 1.28FPS          | 3.14FPS          |
| N=10,000, run on RTX 3070GPU | 1.25FPS | 4FPS |

### 其他

- 运行方式：

  在config.py中配置指定参数，运行crowd_simulation.py即可

- 使用第三方库：

  > numpy._\_version__=='1.23.5'
  >
  > PIL._\_version__ == '9.3.0'
  >
  > csv._\_version__=='1.0'

- 11月30日前已经使用taichi实现了部分人群算法，本项目的第一次commit是一份开赛前我的所有coding结果，方便hackathon评审进行打分。

  比赛期间实现的功能为：steering behavior，A* with taichi, 将数据导出到UE中, 传入图片fix bug

## Project design(English Version)

### Background

In many respects, agent-based crowd simulation models are very similar to particle-based physical simulation algorithms, which can be summarized as:

- each agent in the crowd is capable of acting autonomously and is assigned a set of physical properties.
- each agent follows certain physical laws of motion, often the laws of dynamics.
- the crowd moves in a certain spatial area, and likewise many scenarios of spatial sparsity occur.

The difficulty of current crowd simulation projects is the limitation of computing power on crowd size and simulation efficiency. Taichi is known for its high performance implementation of particle-based fluid simulation, thus it should also be able to significantly improve the performance of crowd simulation.

### Project Function

I hope to implement a crowd simulation library that can be configured to specify the corresponding crowd simulation algorithm, simulation parameters, simulation scenarios, and path planning methods.

Here are some flags ( Features I hope to cover at the end of hackathon):

- Crowd algorithm
  - steering behavior
  - social force
  - Personality-based models, such as OCEAN
- Simulation scenarios
  - Can be specified by parameters
  - Can be specified by passing in images
- Path planning
  - A*
- Results presentation
  - through taichi GUI
  - Export data to other rendering engines to complete data-driven crowd simulation

### Others

Now I have used taichi to implement some of the crowd algorithms, before November 30, I will upload a copy of all my coding results before the start of this project to facilitate scoring by hackathon judges.

Third party libraries: expected to use python numpy, PIL
