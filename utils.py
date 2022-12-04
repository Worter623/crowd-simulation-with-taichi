import taichi as ti
import csv
import numpy as np
from string import Template

_max = int(0xBBBBBB)
_min = int(0x111111)
_range = _max - _min + 1

@ti.func
def convert_data(_data):
    """将_data[0,1]映射到不同的hex色彩区间中"""
    hex_color = int(_data * _range + _min)
    return hex_color

def export_csv(data,path):
    """按指定格式写csv文件 路径为path指定"""
    headers = ['---', 'Loc2D']
    template = Template('(Locs=(X=${s1},Y=${s2}))')

    with open(path, "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        i = 0
        for row in data:
            column = "("
            for count, pos in enumerate(row):
                str_temp = template.safe_substitute(s1=pos[0], s2=pos[1])
                column += str_temp
                if count == len(row) - 1:
                    column += ')'
                else:
                    column += ','

            writer.writerow([i, column])
            i += 1

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

@ti.func
def set_mag(v: ti.template(), mag: ti.f32):
    return (v / v.norm()) * mag


@ti.func
def limit(a, mag):
    norm = a.norm()
    return (a / norm) * mag if norm > 0 and norm > mag else a

@ti.func
def check_in_list(field:ti.template(),element:ti.i32,len:ti.i32)->ti.i8:
    """
    检查某一个元素是否存在于field列表中 返回1存在
    len: 希望检查的长度
    """
    flag = 0
    for i in range (len):
        if flag == 1:
            continue
        if field[i] == element:
            flag = 1
    return flag
