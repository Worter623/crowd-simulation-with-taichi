import taichi as ti
from config import Config
from people import People
import utils

if __name__ == "__main__":
    ti.init(arch=ti.cpu, kernel_profiler=True)#,cpu_max_num_threads=1, debug=True
    # start = time.time()
    config = Config("./map/map1.png") #输入图片路径 不输代表不读图片,用config中的默认配置 
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
    csv = [[] for _ in range(config.N)] #记录每一帧每人位置的链表 TODO：taichi1.3改snode试一下性能
    map_size = [config.WINDOW_WIDTH*10, config.WINDOW_HEIGHT*10]
    
    while gui.running:
    #for i in range (1500):
        if gui.get_event(ti.GUI.RMB,(ti.GUI.PRESS,ti.GUI.SPACE)):
            if gui.event.key == ti.GUI.SPACE: #空格键退出
                gui.running = False
            else:
                mouse = gui.get_cursor_pos()  #打印鼠标右键点击位置的A* map 和 选中的人的编号
                people.print_id(mouse[0],mouse[1])
                _pos = int(ti.floor(mouse[0] * config.window_size) * config.window_size+ ti.floor(mouse[1] * config.window_size))
                print(mouse,_pos,people.astar.map[0,_pos],people.astar.map[1,_pos])

        if config.export_csv == 1:           
            for i in range(config.N):
                csv[i].append(list(map(lambda x,y: x*y ,people.pos[i],map_size)))

        people.make_force()
        people.update()
        people.render(gui)
        gui.show()

    # endend = time.time()
    # print("simulation time = ", endend-end)
    # ti.profiler.print_kernel_profiler_info()

    if config.export_csv == 1:
        utils.export_csv(csv,"data.csv") 

