import taichi as ti
from config import Config
from people import People

#TODO: A* 现在只渲染一个target map 优化！
#TODO:add group_force
# desired relaxation_time 0.5
# desired goal_threshold 0.2
# desired factor 1.0
# social force factor 5.1
# obstacle sigma: 0.2
# obstacle threshold: 3.0
# obstacle factor 10.0


if __name__ == "__main__":
    ti.init(arch=ti.cpu, debug=True, kernel_profiler=True)#,cpu_max_num_threads=1
    # start = time.time()
    config = Config("./map/testpic.png")
    if config.dynamic_search == 0:
        print("static alloc memory")
    else:
        print("using dynamic node")
    gui = ti.GUI("social force model", res=(config.WINDOW_WIDTH, config.WINDOW_HEIGHT),background_color=0xffffff)

    people = People(config)
    print("------------")
    print(people.pos)
    print(people.max_vel)
    print("------------")

    # ti.profiler.print_kernel_profiler_info()
    # end = time.time()
    # print("initialize time = ", end-start)
    # ti.profiler.clear_kernel_profiler_info()  # clear all records

    while gui.running:
        #for i in range (100):
        people.make_force()
        people.update()
        # print(i)
        # print(people.pos)
        # print(people.vel)
        people.render(gui)
        gui.show()

    # endend = time.time()
    # print("simulation time = ", endend-end)
    # ti.profiler.print_kernel_profiler_info()
