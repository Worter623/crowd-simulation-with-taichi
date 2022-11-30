import taichi as ti

@ti.func
def check_in_list(field:ti.template(),element:ti.i32,len:ti.i32,people_index)->ti.i8:
    flag = 0
    for i in range (len):
        if flag == 1:
            continue
        if field[people_index,i] == element:
            flag = 1
    return flag
    
@ti.kernel
def print_field(in_A: ti.template()):
    for I in ti.grouped(in_A):
        print(I, ": ", in_A[I])

@ti.func
def normalize(vec):
    """向量标准化"""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec / norm
    return new_vec, norm


@ti.func
def capped(vec, limit):
    """Scale down a desired velocity to its capped speed."""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec * min(1, limit/norm)
    return new_vec


@ti.func
def vector_angles(vec):
    """Calculate angles for an array of vectors  = atan2(y, x)"""
    return ti.atan2(vec[1], vec[0])
