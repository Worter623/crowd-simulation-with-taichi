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

### 功能设想

希望可以实现一个人群行为仿真库，能够通过配置指定相应的人群仿真算法、仿真参数、模拟场景、路径规划方式。

下面列举hackathon结题时本项目希望涵盖的功能：

- 人群算法
  - steering behavior
  - social force
  - 基于人格的模型，如OCEAN
- 模拟场景
  - 可通过参数指定
  - 可传入图片指定
- 路径规划
  - A*
- 结果呈现
  - 通过taichi GUI展现仿真结果
  - 将数据导出到其他渲染引擎中，完成基于数据驱动的人群行为仿真

### 其他

现在已经使用taichi实现了部分人群算法，在11月30日前，我会在本项目上传一份开赛前我的所有coding结果，以方便hackathon评审进行打分。

第三方库：预计会使用python numpy, PIL




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
