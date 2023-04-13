import numpy as np
import sys

# robot -f -r "replayer/new.rep" -m ./maps/1.txt -c ./SDK/python "python main1.py"
# robot_gui.exe -r "replayer/new.rep" -m ./maps/1.txt -c ./SDK/python "python main1.py"
robots = []
workstations = []
map1_near8or9_station7 = [23, 17, 22, 12, 21, 11, 15, 10]  # 8,9都有
map1_near7_station6 = [31, 24, 32, 38, 25, 33, 39, 34, 40]
map1_near7_station5 = [7, 5, 13, 8, 3, 2, 1, 4, 9]
map1_near7_station4 = [29, 20, 28, 37, 27, 19, 36, 26, 35]
station_num = -1
map4_near8or9_station7 = [0]
map4_near7_station6 = [11,13,15]
map4_near7_station5 = [10,12,14]
map4_near7_station4 = [17]

map2_near8or9_station7 = [3,21]
map2_near7_station6 = [0,2,22,24]#[0,2,22,24]
map2_near7_station5 = [1,15,9,23]#[1,15,9,23]
map2_near7_station4 = [12,9]
map2_for_robot_3 = [22,23,24,15,2]#[23,22,9,24,15]


map3_sell_list, map3_sell_list_id = [], []
map3_needs_list, map3_needs_list_id = [], []
sell_list_id = [[], []]
consume_stations_idx = [[], []]
needs_list, needs_list_id = [], []
n_list = [[1, 2], [1, 3], [2, 3], [4, 5, 6], [7], list(range(1, 8))]

class Robot:
    def __init__(self,
                 target=-1,
                 location_id=None,
                 goods_id=None,
                 time_coefficient=None,
                 collide_coefficient=None,
                 angular_speed=None,
                 line_speed=None,
                 orientation=None,
                 location=None,
                 target_location=None,
                 target_id=None
                 ):
        self.location_id = location_id
        self.goods_id = goods_id
        self.time_coefficient = time_coefficient
        self.collide_coefficient = collide_coefficient
        self.angular_speed = angular_speed
        self.line_speed = line_speed
        self.orientation = orientation
        self.location = location
        self.target = target
        self.target_location = target_location
        self.target_id = target_id

class WorkStation:
    def __init__(self, id=None, location=None, res_time=None,
                 product_state=None, material_state=[], need_material=[], chosen=np.uint8(0)):
        self.id = id
        self.location = location
        self.res_time = res_time
        self.material_state = material_state
        self.product_state = product_state
        self.need_material = need_material
        self.chosen = chosen

# 初始化程序，使用文件读取时测试时间为0.0009975433349609375秒，满足要求
def initial():
    global robots,station_num,workstations, map1_near8or9_station7, map1_near7_station6, map1_near7_station5, map1_near7_station4,map4_near8or9_station7,map4_near7_station6,map4_near7_station5,map4_near7_station4
    j = 0
    while True:
        line = input()
        if line == 'OK':
            break
        assert len(line) == 100, 'number of the map columns error'
        for i, s in enumerate(list(line)):
            if s == 'A':
                xr = np.float32((i + 1) * 0.5)
                yr = np.float32(50 - (j + 1) * 0.5)
                robots.append(Robot(location=[xr, yr]))
            elif s.isdigit():
                id = np.int8(s)
                xw = np.float32((i + 1) * 0.5)
                yw = np.float32(50 - (j + 1) * 0.5)
                if id == 8 or id == 9:
                    consume_stations_idx[id - 8].append(len(workstations))
                workstations.append(WorkStation(id=id, location=[xw, yw],
                                                material_state=[], product_state=0))
        j += 1
    assert j == 100, 'number of rows error'
    assert len(robots) == 4, 'number of robots error'
    assert len(workstations) <= 50, 'number of workstations error'

    for worker in workstations:
        if worker.id in [4, 5, 6]:
            worker.need_material = [[1, 2], [1, 3], [2, 3]][worker.id - 4]
        if worker.id == 7:
            worker.need_material = [6, 5, 4]

    station_num = len(workstations)
    sys.stderr.write(f'statio_nnum:{len(workstations)}\n')
    finish()


def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()


def getInformation():
    global robots, workstations
    station_num = np.int8(sys.stdin.readline().strip())
    for i in range(station_num):
        newline = sys.stdin.readline()
        line = newline.strip().split(' ')
        workstations[i].id = np.int8(line[0])
        workstations[i].location = np.float32(line[1:3])
        workstations[i].res_time = np.int32(line[3])
        workstations[i].material_state = getMaterialState(np.int8(line[4]))
        workstations[i].product_state = np.int8(line[5])

    for j in range(4):
        line = sys.stdin.readline().strip().split(' ')
        robots[j].location_id = np.int8(line[0])
        robots[j].goods_id = np.int8(line[1])
        robots[j].time_coefficient = np.float32(line[2])
        robots[j].collide_coefficient = np.float32(line[3])
        robots[j].angular_speed = np.float32(line[4])
        robots[j].line_speed = np.float32(line[5:7])
        robots[j].orientation = np.float32(line[7])
        robots[j].location = np.float32(line[8:])
    line = sys.stdin.readline().strip()
    assert line == 'OK', 'Input frame does not end correctly!'

def getInformation_2():
    global robots, workstations 
    station_num = np.int8(sys.stdin.readline().strip())
    for i in range(station_num):
        line = sys.stdin.readline().strip().split(' ')
        workstations[i].id = np.int8(line[0])
        workstations[i].location = np.float32(line[1:3])
        workstations[i].res_time = np.int8(line[3])
        workstations[i].material_state = np.int8(line[4])
        workstations[i].product_state = np.int8(line[5])
    for j in range(4):
        line = sys.stdin.readline().strip().split(' ')
        robots[j].location_id = np.int8(line[0])
        robots[j].goods_id = np.int8(line[1])
        robots[j].time_coefficient = np.float32(line[2])
        robots[j].collide_coefficient = np.float32(line[3])
        robots[j].angular_speed = np.float32(line[4])
        robots[j].line_speed = np.float32(line[5:7])
        robots[j].orientation = np.float32(line[7])
        robots[j].location = np.float32(line[8:])
    line = sys.stdin.readline().strip()
    assert line == 'OK', 'Input frame does not end correctly!'

def computeAngle(source, target):
    angle = 0
    distance = np.sqrt(np.power((target[0] - source[0]), 2) + np.power((target[1] - source[1]), 2))
    if target[0] >= source[0]:
        angle = np.arcsin((target[1] - source[1]) / distance)
    elif target[0] < source[0] and target[1] >= source[1]:
        angle = np.pi - np.arcsin((target[1] - source[1]) / distance)
    elif target[0] < source[0] and target[1] < source[1]:
        angle = -(np.pi + np.arcsin((target[1] - source[1]) / distance))
    return angle


def getMaterialState(num):
    state_list = []
    for i in range(1, 7):
        temp = np.right_shift(num, i)
        if np.bitwise_and(temp, 1) == 1:
            state_list.append(i)
    state_list.sort(reverse=True)
    return state_list


def compute_dis(a, b):
    a, b = tuple(a), tuple(b)
    disc = ((a[0] - b[0]) ** 2 + (
            a[1] - b[1]) ** 2) ** 0.5
    return disc


def compute_near(station, id):  # 以station为定点，求出最近的id号工作台到station的距离
    min_disc = 10000
    near_worker = -1
    for i, woker in enumerate(workstations):
        if woker.id == id:
            disc = compute_dis(woker.location, station.location)
            if disc < min_disc:
                min_disc = disc
                near_worker = i
    return near_worker


def map1_robot_go_station(robot_id, i):
    if robot_id == -1:
        sys.stdout.write(f'forward {robot_id} 0\n')
        sys.stdout.write(f'rotate {robot_id} 0\n')
        return
    if i == -1:
        sys.stdout.write(f'forward {robot_id} 0\n')
        sys.stdout.write(f'rotate {robot_id} 0\n')
        return

    disc_robot2station = compute_dis(robots[robot_id].location, workstations[i].location)

    x, y = robots[robot_id].location
    xw, yw = workstations[i].location

    angle = computeAngle((x, y), (xw, yw))
    differ = angle - robots[robot_id].orientation
    if abs(differ) > np.pi:
        differ = -differ
    if abs(differ) > 0.3:
        angle_speed = (differ / abs(differ)) * np.pi
        line_speed = 2.3
    elif abs(differ) > 0.02:
        angle_speed = (5 / 180) * np.pi * (differ / abs(differ))
        line_speed = 6
    else:
        angle_speed = 0
        line_speed = 6

    if disc_robot2station < 0.4:
        line_speed = 0
        angle_speed = 0
        if robots[robot_id].goods_id == 0:
            if workstations[i].product_state == 1:
                # 如果没有放的地方，就不买了 待优化
                angle2 = computeAngle(robots[robot_id].location,
                                      workstations[compute_near(robots[robot_id], 8)].location)
                differ = angle2 - robots[robot_id].orientation
                if abs(differ) > np.pi:
                    differ = -differ
                if abs(differ) > 0.3:  # 这里值得调参
                    angle_speed = (differ / abs(differ)) * np.pi
                else:
                    angle_speed = 0
                    if frame_id < 8795:  # 最后一百帧不买了，但是可以卖
                        sys.stdout.write(f'buy {robot_id}\n')
                        robots[robot_id].target = -1
                line_speed = 0

        else:
            # 9工作台通吃，8工作台只吃7，添加材料
            if workstations[i].id == 9 or (robots[robot_id].goods_id == 7 and workstations[i].id == 8) or (
                    (robots[robot_id].goods_id in workstations[i].need_material) and (
                    robots[robot_id].goods_id not in workstations[i].material_state)):
                sys.stdout.write(f'sell {robot_id}\n')
                robots[robot_id].target = -1

    sys.stdout.write(f'forward {robot_id} %d\n' % (line_speed))
    sys.stdout.write(f'rotate {robot_id} %f\n' % (angle_speed))


def station_need(station_id):
    station = workstations[station_id]
    need = set(station.need_material) - set(station.material_state)
    need = sorted(need, reverse=True)
    return need


def map1_go_and_sell_7(i):
    station7 = map1_near8or9_station7.copy()
    while station7:
        sta7 = station7.pop()
        if workstations[sta7].product_state == 1:
            robots[i].target = sta7
        if robots[i].goods_id == 7:
            sta8, sta9 = compute_near(robots[i], 8), compute_near(robots[i], 9)
            robots[i].target = sta8 if compute_dis(workstations[sta8].location, robots[i].location) < compute_dis(
                workstations[sta9].location, robots[i].location) else sta9
    if robots[0].target == -1 and robots[0].goods_id == 0:
        robots[0].target = compute_near(robots[0], 8)


def map1_have_product(i):
    station = [map1_near7_station4.copy(), map1_near7_station5.copy(), map1_near7_station6.copy()][i - 4]
    while station:
        sta = station.pop()
        if workstations[sta].product_state == 1:
            return sta
    return -1


def map1_find_7_sell_456(i):
    station7 = map1_near8or9_station7.copy()
    sta6, sta5, sta4 = map1_have_product(6), map1_have_product(5), map1_have_product(4)
    while station7:
        sta7 = station7.pop()
        need7 = station_need(sta7)
        if not need7:
            continue
        if len(need7) == 3:
            robots[i].target = sta6
        elif 6 in need7:
            robots[i].target = sta6
        elif 5 in need7:
            robots[i].target = sta5
        else:
            robots[i].target = sta4
        if robots[i].goods_id in [4, 5, 6]:
            robots[i].target = sta7


def map1_find_456(i):
    station6 = map1_near7_station6.copy()
    station5 = map1_near7_station5.copy()
    station4 = map1_near7_station4.copy()
    station6, station5 = [[station5, station4], [station4, station6], [station6, station5]][i - 1]
    while station6:
        sta6 = station6.pop()
        need6 = station_need(sta6)
        if not need6:
            continue
        if len(need6) == 2:
            robots[i].target = sta6
        else:
            if i in need6:
                robots[i].target = sta6
                continue
            while station5:
                sta5 = station5.pop()
                need5 = station_need(sta5)
                if not need5:
                    continue
                if i in need5:
                    robots[i].target = sta5


def map1_buy_321():
    for i in range(1, 4):
        if robots[i].goods_id == 0:
            robots[i].target = compute_near(robots[i], i)
        else:
            map1_find_456(i)


def map1_robot_fill_456():
    # 光入4,5,6 ,有产品，有卖位，再卖去7
    # 有东西卖的时候，直接看离他最近（目标为他）（或没有目标）的机器人
    # 怎么保证4,5,6产出差不多呢
    map1_buy_321()
    map1_find_7_sell_456(0)  # 0号机器人
    map1_go_and_sell_7(0)

def map3_computeDistance(loc, computeList, idList):
    distance_list = []
    for index, good_id in enumerate(computeList):
        xw, yw = workstations[idList[index]].location
        square_distance = np.power((loc[0] - xw), 2) + np.power((loc[1] - yw), 2)
        distance_list.append(square_distance)
    sorted_dis = sorted(enumerate(distance_list), key=lambda x: x[1])
    idx = [idList[i[0]] for i in sorted_dis]
    distance = [i[1] for i in sorted_dis]
    # station = [computeList[i[0]] for i in sorted_dis]
    return idx, distance


def map3_updateListState():
    global map3_sell_list, map3_sell_list_id, map3_needs_list, map3_needs_list_id
    map3_sell_list, map3_sell_list_id, map3_needs_list, map3_needs_list_id = [], [], [], []
    n_list = [[1, 2], [1, 3], [2, 3], [4, 5, 6], [7], list(range(1, 8))]
    for i, station in enumerate(workstations):
        material_state = getMaterialState(station.material_state)
        if (station.id in range(1, 8)) and (station.product_state == 1):
            map3_sell_list.append(station.id)
            map3_sell_list_id.append(i)
        if station.id in range(4, 8):
            stand = n_list[station.id - 4]
            for m in stand:
                if m not in material_state:
                    map3_needs_list.append(m)
                    map3_needs_list_id.append(i)
        elif station.id == 8:
            map3_needs_list.append(7)
            map3_needs_list_id.append(i)
        elif station.id == 9:
            for j in range(1, 8):
                map3_needs_list.append(j)
                map3_needs_list_id.append(i)


def map3_findStations(id):
    station_list = []
    station_index_list = []
    for index, station in enumerate(workstations):
        if station.id in id:
            station_list.append(station.id)
            station_index_list.append(index)
    return station_list, station_index_list


def map3_computeAngle(source, target):
    angle = 0
    distance = np.sqrt(np.power((target[0] - source[0]), 2) + np.power((target[1] - source[1]), 2))
    if target[0] >= source[0]:
        angle = np.arcsin((target[1] - source[1]) / distance)
    elif target[0] < source[0] and target[1] >= source[1]:
        angle = np.pi - np.arcsin((target[1] - source[1]) / distance)
    elif target[0] < source[0] and target[1] < source[1]:
        angle = -(np.pi + np.arcsin((target[1] - source[1]) / distance))
    return angle


def map3_distinguish(good_id, station):
    n_list = [[2,1], [3,1], [3,2], [6,5,4], [7], list(range(1, 8))]
    material = getMaterialState(station.material_state)
    stand = n_list[station.id - 4]
    if good_id not in stand:
        return False
    if good_id in material:
        return False
    else:
        return True


def map3_robotAction(source, target):
    x, y = source
    xw, yw = target
    # 距离越近，速度越小
    # distance = np.sqrt(np.power((x - xw), 2) + np.power((y - yw), 2))
    angle = map3_computeAngle((x, y), (xw, yw))
    differ = angle - robots[i].orientation
    if abs(differ) > np.pi:
        differ = -differ
    if abs(differ) > 0.1:
        angular_speed = (differ / abs(differ)) * np.pi
        line_speed = 2
    elif abs(differ) > 0.01:
        angular_speed = (5 / 180) * np.pi * (differ / abs(differ))
        line_speed = 6
    else:
        angular_speed = 0
        line_speed = 6

    sys.stdout.write('rotate {} {}\n'.format(i, angular_speed))
    sys.stdout.write('forward {} {}\n'.format(i, line_speed))


def map3_findGood(station):
    need_count = map3_needs_list.count(station.id)
    got_count = sum([1 if robot.goods_id == station.id else 0 for robot in robots])
    if need_count > got_count:
        return True
    else:
        return False


def map3_anti_collision():
    for i in range(4):
        res_list = list(range(i, 4))
        res_list.remove(i)
        for j in res_list:
            x1, y1 = robots[i].location
            x2, y2 = robots[j].location
            dis = np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2))
            if dis < 1.15:
                i_conj = map3_computeAngle(robots[i].location, robots[j].location)
                j_conj = map3_computeAngle(robots[j].location, robots[i].location)
                i_differ = robots[i].orientation - i_conj
                j_differ = robots[j].orientation - j_conj
                if abs(i_differ) < 0.3:
                    angle_speed = np.pi * (i_differ / abs(i_differ))
                    sys.stdout.write('rotate {} {}\n'.format(i, angle_speed))
                    sys.stdout.write('forward {} {}\n'.format(i, 6))
                if abs(j_differ) < 0.3:
                    angle_speed = np.pi * (j_differ / abs(j_differ))
                    sys.stdout.write('rotate {} {}\n'.format(j, angle_speed))
                    sys.stdout.write('forward {} {}\n'.format(j, 6))

def map2_robot_go_station(robot_id,i):
    if robot_id == -1:
        sys.stdout.write(f'forward {robot_id} 0\n')
        sys.stdout.write(f'rotate {robot_id} 0\n')
        return
    if i == -1:
        sys.stdout.write(f'forward {robot_id} 0\n')
        sys.stdout.write(f'rotate {robot_id} 0\n')
        return

    disc_robot2station = compute_dis(robots[robot_id].location, workstations[i].location)

    x, y = robots[robot_id].location
    xw, yw = workstations[i].location

    angle = computeAngle((x, y), (xw, yw))
    differ = angle - robots[robot_id].orientation
    if abs(differ) > np.pi:
        differ = -differ
    if abs(differ) > 0.2:
        angle_speed = (differ / abs(differ)) * np.pi
        line_speed = 1.8
    elif abs(differ) > 0.02:
        angle_speed = (5 / 180) * np.pi * (differ / abs(differ))
        line_speed = 6
    else:
        angle_speed = 0
        line_speed = 6

    if disc_robot2station < 0.3 :
        line_speed = 0
        angle_speed = 0
        if robots[robot_id].goods_id == 0 :
            if workstations[i].product_state == 1:
                # 极端情况慢点买。
                if  workstations[i].id in[5,6,7]:
                    angle2 = computeAngle(robots[robot_id].location,workstations[10].location)
                    differ = angle2 - robots[robot_id].orientation
                    if abs(differ) > np.pi:
                        differ = -differ
                    if abs(differ) > 0.3:# 调参
                        angle_speed = (differ / abs(differ)) * np.pi
                    else:
                        angle_speed = 0
                        if frame_id < 8795:# 最后一百帧不买了，但是可以卖
                            sys.stdout.write(f'buy {robot_id}\n')
                            robots[robot_id].target = -1
                    line_speed = 0
                else:
                    if frame_id < 8795:  # 最后一百帧不买了，但是可以卖
                        sys.stdout.write(f'buy {robot_id}\n')
                        robots[robot_id].target = -1
                        line_speed = 6

        else:
            # 9工作台通吃，8工作台只吃7，添加材料
            if workstations[i].id == 9 or (robots[robot_id].goods_id == 7 and workstations[i].id == 8 )or((robots[robot_id].goods_id in workstations[i].need_material) and (robots[robot_id].goods_id not in workstations[i].material_state )):
                sys.stdout.write(f'sell {robot_id}\n')
                robots[robot_id].target = -1

    sys.stdout.write(f'forward {robot_id} %d\n' % (line_speed))
    sys.stdout.write(f'rotate {robot_id} %f\n' % (angle_speed))

def map4_robot_go_station(robot_id,i):
    if robot_id == -1:
        sys.stdout.write(f'forward {robot_id} 0\n')
        sys.stdout.write(f'rotate {robot_id} 0\n')
        return
    if i == -1:
        sys.stdout.write(f'forward {robot_id} 0\n')
        sys.stdout.write(f'rotate {robot_id} 0\n')
        return

    disc_robot2station = compute_dis(robots[robot_id].location, workstations[i].location)

    x, y = robots[robot_id].location
    xw, yw = workstations[i].location

    angle = computeAngle((x, y), (xw, yw))
    differ = angle - robots[robot_id].orientation
    if abs(differ) > np.pi:
        differ = -differ
    if abs(differ) > 0.3:
        angle_speed = (differ / abs(differ)) * np.pi
        line_speed = 3
    elif abs(differ) > 0.02:
        angle_speed = (5 / 180) * np.pi * (differ / abs(differ))
        line_speed = 6
    else:
        angle_speed = 0
        line_speed = 6

    if disc_robot2station < 0.3 :
        line_speed = 0
        angle_speed = 0
        if robots[robot_id].goods_id == 0 :
            if workstations[i].product_state == 1:
                # 极端情况慢点买。
                if  workstations[i].id == 4 or workstations[i].id == 7:
                    angle2 = computeAngle(robots[robot_id].location,workstations[compute_near(robots[robot_id], 8)].location)
                    differ = angle2 - robots[robot_id].orientation
                    if abs(differ) > np.pi:
                        differ = -differ
                    if abs(differ) > 0.3:# 调参
                        angle_speed = (differ / abs(differ)) * np.pi
                    else:
                        angle_speed = 0
                        if frame_id < 8795:# 最后一百帧不买了，但是可以卖
                            sys.stdout.write(f'buy {robot_id}\n')
                            robots[robot_id].target = -1
                    line_speed = 0
                else:
                    if frame_id < 8795:  # 最后一百帧不买了，但是可以卖
                        sys.stdout.write(f'buy {robot_id}\n')
                        robots[robot_id].target = -1
        else:
            # 9工作台通吃，8工作台只吃7，添加材料
            if workstations[i].id == 9 or (robots[robot_id].goods_id == 7 and workstations[i].id == 8 )or((robots[robot_id].goods_id in workstations[i].need_material) and (robots[robot_id].goods_id not in workstations[i].material_state )):
                sys.stdout.write(f'sell {robot_id}\n')
                robots[robot_id].target = -1

    sys.stdout.write(f'forward {robot_id} %d\n' % (line_speed))
    sys.stdout.write(f'rotate {robot_id} %f\n' % (angle_speed))

def map4_go_and_sell_7(i):
    station7 = map4_near8or9_station7.copy()
    while station7:
        sta7 = station7.pop()
        if workstations[sta7].product_state == 1:
            robots[i].target = sta7
        if robots[i].goods_id == 7:
            robots[i].target = compute_near(robots[i], 8)


def map4_have_product(i):
    station = [map4_near7_station4.copy(), map4_near7_station5.copy(), map4_near7_station6.copy()][i-4]
    while station:
        sta = station.pop()
        if workstations[sta].product_state == 1:
            return sta
    return -1


def map4_find_7_sell_456(i):
    station7 = map4_near8or9_station7.copy()
    sta6, sta5, sta4 = map4_have_product(6), map4_have_product(5), map4_have_product(4)
    while station7:
        sta7 = station7.pop()
        need7 = station_need(sta7)
        if not need7:
            continue
        if 6 in need7 and sta6 != -1:
            robots[i].target = sta6
        elif 5 in need7 and sta5 != -1:
            robots[i].target = sta5
        else:
            if sta4 != -1:
                robots[i].target = sta4

    for i in range(4):
        if robots[i].goods_id in [4, 5, 6]:
            robots[i].target = 0

def map4_find_456(i):
    station6 = map4_near7_station6.copy()
    station5 = map4_near7_station5.copy()
    station4 = map4_near7_station4.copy()
    station6 , station5  = [  [station5,station4],[station4,station6],[station6,station5] ][robots[i].goods_id-1]
    while station6:
        sta6 = station6.pop()
        need6 = station_need(sta6)
        if not need6:
            while station5:
                sta5 = station5.pop()
                need5 = station_need(sta5)
                if not need5:
                    continue
                if i in need5:
                    robots[i].target = sta5
                    return
            continue
        if len(need6) == 2:
            robots[i].target = sta6
        else:
            if i in need6:
                robots[i].target = sta6
                continue
            while station5:
                sta5 = station5.pop()
                need5 = station_need(sta5)
                if not need5:
                    continue
                if i in need5:
                    robots[i].target = sta5


def map4_buy_321():
    for i in range(1,4):
        if robots[i].goods_id == 0:
            robots[i].target = compute_near(robots[i],i)
        else:
            map4_find_456(i)


def map4_robot_fill_456():
    map4_buy_321()
    map4_find_7_sell_456(0)# 0号机器人
    map4_go_and_sell_7(0)# 0号机器人
    if robots[0].goods_id == 0 :
        if robots[0].target == -1:
            robots[0].target = 1
    else:
        if robots[0].goods_id in [1,2,3]:
            map4_find_456(0)
    # 给 7 让道，防碰撞


def map2_have_product(i):
    station = [map2_near7_station4.copy(), map2_near7_station5.copy(), map2_near7_station6.copy()][i-4]
    while station:
        sta = station.pop()
        if workstations[sta].product_state == 1:
            return sta
    return -1

def map2_buy_321():
    station5 = map2_near7_station5.copy()
    if robots[0].goods_id == 0:
        while station5:
            sta5 = station5.pop()
            need5 = station_need(sta5)
            if not need5:
                continue
            robots[0].target = compute_near(workstations[sta5],need5[0])
    station5 = map2_near7_station6.copy()
    if robots[1].goods_id == 0:
        while station5:
            sta5 = station5.pop()
            need5 = station_need(sta5)
            if not need5:
                continue
            robots[1].target = compute_near(workstations[sta5],need5[0])
    station5 = map2_near7_station4.copy()
    if robots[2].goods_id == 0:
        while station5:
            sta5 = station5.pop()
            need5 = station_need(sta5)
            if not need5:
                continue
            robots[2].target = compute_near(workstations[8],need5[0])
    station4 = map2_for_robot_3.copy()
    if robots[3].goods_id == 0:
        while station4:
            sta5 = station4.pop()
            if workstations[sta5].product_state == 1:
                if workstations[sta5].id in station_need(21):
                        robots[3].target = sta5
                break
            need5 = station_need(sta5)
            if not need5:
                continue
            robots[3].target = compute_near(workstations[sta5], need5[0])


def map2_sell_123(i):
    if i in [1,2,0]:
        station5 = [map2_near7_station5.copy(),map2_near7_station6.copy(),map2_near7_station4.copy()][i]
        while station5:
            sta5 = station5.pop()
            need5 = station_need(sta5)
            if not need5:
                continue
            if robots[i].goods_id in need5:
                robots[i].target = sta5
    if i==3:
        station5 = map2_for_robot_3.copy()
        while station5:
            sta5 = station5.pop()
            need5 = station_need(sta5)
            if not need5:
                continue
            if robots[i].goods_id in need5:
                robots[i].target = sta5


def map2_buy_456():
    station7 = map2_near8or9_station7.copy()
    while station7:
        sta7 = station7.pop()
        need7 = station_need(sta7)
        if not need7:
            continue
        else:
            for i in [4,5,6]:
                if i in need7:
                    id = [2,0,1][i-4]
                    sta456 = map2_have_product(i)  # 0号机器人
                    if sta456 != -1:
                        robots[id].target = sta456

def map2_near_free(sta456):
    mindisc = 1000
    id = -1
    for i in range(4):
        if robots[i].goods_id ==0:
            disc = compute_dis(robots[i].location,workstations[sta456].location)
            if disc<mindisc:
                mindisc = disc
                id = i
    return id


def map2_sell_7():
    station7 = map2_near8or9_station7.copy()
    while station7:
        sta7 = station7.pop()
        if workstations[sta7].product_state == 1:
            id = map2_near_free(sta7)
            if id != -1:
                robots[id].target = sta7
    for i in range(4):
        if robots[i].goods_id == 7:
            robots[i].target = compute_near(robots[i], 8)

def map2_sell_456():
    station7 = map2_near8or9_station7.copy()
    for i in range(4):
        if robots[i].goods_id in[4,5,6]:
            # robots[i].target = compute_near(robots[i],7)
            # if robots[i].goods_id ==4:
            while station7:
                sta7 = station7.pop()
                need7=station_need(sta7)
                if robots[i].goods_id in need7:
                    robots[i].target = sta7

    for i in range(4):
        if robots[i].goods_id == 7:
            robots[i].target = 10

def map2_robot_fill_456():
    map2_buy_321()
    for i in range(4):
        map2_sell_123(i)
    map2_buy_456()
    map2_sell_456()
    map2_sell_7()


def getMaterialState_2(num, mode, station_id=4):
    global n_list
    state_list = []
    if mode == 'm':
        for i in range(1, 7):
            temp = np.right_shift(num, i)
            if np.bitwise_and(temp, 1) == 1:
                state_list.append(i)
    elif mode == 'f':
        if station_id >= 4:
            for i in n_list[station_id - 4]:
                temp = np.right_shift(num, i)
                if np.bitwise_and(temp, 1) == 0:
                    state_list.append(i)
        state_list.append(np.bitwise_and(num, 1))
    return state_list


def setFreeList(station_idx, index, value):
    chosen = workstations[station_idx].chosen
    if value == 1:
        temp = np.left_shift(1, index)
        workstations[station_idx].chosen = np.bitwise_or(chosen, temp)
    elif value == 0:
        temp = np.bitwise_not(np.left_shift(1, index))
        workstations[station_idx].chosen = np.bitwise_and(chosen, temp)


def computeDistance(loc, idxList):
    distance_list = []
    for idx in idxList:
        xw, yw = workstations[idx].location
        square_distance = np.power((loc[0] - xw), 2) + np.power((loc[1] - yw), 2)
        distance_list.append(square_distance)
    sorted_dis = sorted(enumerate(distance_list), key=lambda x: x[1])
    index = [i[0] for i in sorted_dis]
    sorted_idx = [idxList[j] for j in index]
    return index, sorted_idx


def updateListState():
    global sell_list_id, needs_list, needs_list_id
    sell_list_id, needs_list, needs_list_id = [[], []], [], []
    global n_list
    for i, station in enumerate(workstations):
        material_state = getMaterialState_2(station.material_state, 'm')
        chosen = getMaterialState_2(station.chosen, 'f', station.id)
        if (station.id in [2, 3, 6]) and (station.product_state == 1) and (chosen[-1] == 0):
            if station.id == 6:
                sell_list_id[1].append(i)
            else:
                sell_list_id[0].append(i)
        if station.id == 6:
            for m in [2, 3]:
                if (m not in material_state) and (m in chosen[:-1]):
                    needs_list.append(m)
                    needs_list_id.append(i)


def findStations(id):
    station_list = []
    station_index_list = []
    for index, station in enumerate(workstations):
        if station.id in id:
            station_list.append(station.id)
            station_index_list.append(index)
    return station_list, station_index_list
def distinguish(good_id, station):
    global n_list
    material = getMaterialState_2(station.material_state, 'm')
    stand = n_list[station.id - 4]
    if good_id not in stand:
        return False
    if good_id in material:
        return False
    else:
        return True


def robotAction(source, target):
    x, y = source
    xw, yw = target
    # 距离越近，速度越小
    # distance = np.sqrt(np.power((x - xw), 2) + np.power((y - yw), 2))
    angle = computeAngle((x, y), (xw, yw))
    differ = angle - robots[i].orientation
    if abs(differ) > np.pi:
        differ = -differ
    if abs(differ) > 0.1:
        angular_speed = (differ / abs(differ)) * np.pi
        line_speed = 2
    elif abs(differ) > 0.01:
        angular_speed = (5 / 180) * np.pi * (differ / abs(differ))
        line_speed = 6
    else:
        angular_speed = 0
        line_speed = 6

    sys.stdout.write('rotate {} {}\n'.format(i, angular_speed))
    sys.stdout.write('forward {} {}\n'.format(i, line_speed))


def findGood(idx):
    idx = workstations[idx].id
    need_count = needs_list.count(idx)
    got_count = sum([1 if robot.goods_id == idx else 0 for robot in robots])
    if need_count > got_count:
        return True
    else:
        return False


def anti_collision():
    for i in range(4):
        res_list = list(range(i, 4))
        res_list.remove(i)
        for j in res_list:
            x1, y1 = robots[i].location
            x2, y2 = robots[j].location
            dis = np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2))
            if dis < 1.15:
                i_conj = computeAngle(robots[i].location, robots[j].location)
                j_conj = computeAngle(robots[j].location, robots[i].location)
                i_differ = robots[i].orientation - i_conj
                j_differ = robots[j].orientation - j_conj
                if abs(i_differ) < 0.5:
                    angle_speed = np.pi * (i_differ / abs(i_differ))
                    sys.stdout.write('rotate {} {}\n'.format(i, angle_speed))
                    sys.stdout.write('forward {} {}\n'.format(i, 6))
                if abs(j_differ) < 0.5:
                    angle_speed = np.pi * (j_differ / abs(j_differ))
                    sys.stdout.write('rotate {} {}\n'.format(j, angle_speed))
                    sys.stdout.write('forward {} {}\n'.format(j, 6))


def wait(i):
    sys.stdout.write('forward {} {}\n'.format(i, 0))
    sys.stdout.write('rotate {} {}\n'.format(i, 0))


if __name__ == '__main__':
    initial()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.split(' ')
        frame_id = int(parts[0])
        score = int(parts[1])

        if station_num == 43:
            getInformation()
            if frame_id == 1:
                sys.stderr.write(f'station_num:{station_num}\n')
            sys.stdout.write('{}\n'.format(frame_id))
            map1_robot_fill_456()
            for i in range(4):
                # sys.stderr.write(f'robots[{i}].goods_id:{robots[i].goods_id},robots[{i}].target:{robots[i].target},id:{workstations[robots[i].target].id}\n')
                map1_robot_go_station(i, robots[i].target)
        elif station_num == 18:
            getInformation()
            if frame_id == 1:
                sys.stderr.write(f'station_num:{station_num}\n')
            sys.stdout.write('{}\n'.format(frame_id))
            map4_robot_fill_456()
            for i in range(4):
                # sys.stderr.write(f'robots[{i}].goods_id:{robots[i].goods_id},robots[{i}].target:{robots[i].target},id:{workstations[robots[i].target].id}\n')
                map4_robot_go_station(i, robots[i].target)
        # sys.stderr.write(f'\nframe_id:{frame_id}\n')
        elif station_num == 25:
            getInformation()
            if frame_id == 1:
                sys.stderr.write(f'station_num:{station_num}\n')
            sys.stdout.write('{}\n'.format(frame_id))
            map2_robot_fill_456()
            for i in range(4):
                # sys.stderr.write(f'robots[{i}].goods_id:{robots[i].goods_id},robots[{i}].target:{robots[i].target},id:{workstations[robots[i].target].id}\n')
                map2_robot_go_station(i, robots[i].target)

        elif station_num==50:
            getInformation_2()
            sys.stdout.write('{}\n'.format(frame_id))

            if frame_id == 1:
                for i in range(4):
                    x, y = robots[i].location
                    station, index = map3_findStations([1, 2, 3])
                    idx, dis = map3_computeDistance((x, y), station, index)
                    for j in idx:
                        if workstations[j].chosen == 1:
                            continue
                        elif workstations[j].chosen == 0:
                            robots[i].target_location = workstations[j].location
                            robots[i].target_id = j
                            workstations[j].chosen = 1
                            break
            else:
                map3_updateListState()
                for i in range(4):
                    x, y = robots[i].location
                    if robots[i].target_location is None:
                        if robots[i].goods_id == 0:
                            idx, _ = map3_computeDistance((x, y), map3_sell_list, map3_sell_list_id)
                            for j in idx:
                                if workstations[j].chosen == 1:
                                    continue
                                elif workstations[j].chosen == 0 and map3_findGood(workstations[j]):
                                    robots[i].target_location = workstations[j].location
                                    robots[i].target_id = j
                                    workstations[j].chosen = 1
                                    break
                        else:
                            idx, _ = map3_computeDistance((x, y), map3_needs_list, map3_needs_list_id)
                            for j in idx:
                                if workstations[j].chosen == 1:
                                    continue
                                elif (workstations[j].chosen == 0) and map3_distinguish(robots[i].goods_id, workstations[j]):
                                    robots[i].target_location = workstations[j].location
                                    robots[i].target_id = j
                                    workstations[j].chosen = 1
                                    break
                    else:
                        if robots[i].location_id == robots[i].target_id:
                            if robots[i].goods_id == 0 :
                                if frame_id < 8831:
                                    sys.stdout.write('buy {}\n'.format(i))
                            else:
                                sys.stdout.write('sell {}\n'.format(i))
                            workstations[robots[i].target_id].chosen = 0
                            robots[i].target_location = None
                            robots[i].target_id = None
                        else:
                            map3_robotAction((x, y), robots[i].target_location)
            map3_anti_collision()
        else:
            getInformation()
            sys.stdout.write('{}\n'.format(frame_id))
            pass


        finish()
