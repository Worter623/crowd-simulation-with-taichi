# crowd-simulation-with-taichi

crowd simulation algorithms implemented with taichi

a work for taichi hackathon



## 项目设计

### 选题背景

在许多方面上，基于agent的人群仿真算法与基于粒子的物理仿真算法是非常类似的，具体来说可以归纳为三点：

- 粒子性，人群中的每个agent都能够自主行动，并被赋予一组物理属性；
- 物理限制，每个agent都遵循一定的物理运动规律，如动力学规律；
- 空间情境，人群在一定的空间区域中移动，同样会出现许多spatial sparsity的场景。

当前人群仿真模拟项目的难点在于计算机算力对人群规模、仿真效率的限制。taichi 以高性能实现基于粒子的流体仿真而闻名，应当也能够大幅度提升人群运动仿真模拟的性能。

### 现有功能

实现一个人群行为仿真库，能够通过配置(config.py)指定相应的人群仿真算法、仿真参数、模拟场景、路径规划方式。

本项目目前涵盖的功能：

- 人群算法

  - steering behavior
  - social force
  - ~~基于人格的模型，如OCEAN~~（还没有实现！打脸啦

- 障碍场景模拟

  - 可通过参数指定：目前仅支持在配置中以(startx, starty, endx, endy)这样的四元组形式指定多条线段的起点、终点
  - 可传入图片指定：二值图片，如map文件夹中示例

- 路径规划

  - A*（在仿真开始前预渲染）：

    目前的算法主要是通过将顺序结构A*中这两个步骤并行化，去提升算法的性能：

    1）从open列表中提取出优先级最高的节点q

    2）扩展q的相邻节点

    一个简单的性能测试：

    分别计算10,000个pixel 到一个target pixel的A*，使用python和taichi实现的时间分别为（答辩的时候再给）

- 结果呈现

  - 通过taichi GUI展现仿真结果
  - 将数据导出到其他渲染引擎中，完成基于数据驱动的人群行为仿真：eg.写csv文件，通过data table导入UE

### 其他

11月30日前已经使用taichi实现了部分人群算法，本项目的第一次commit是一份开赛前我的所有coding结果，方便hackathon评审进行打分。

使用第三方库：python numpy, random, PIL




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
