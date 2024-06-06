import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy.interpolate import interp1d
import math
# 设置中文字体路径
font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
# font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
font_prop = FontProperties(fname=font_path)

# 全局设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取 Excel 文件中的特定工作表
excel_file = '线路条件数据.xlsx'  # 确保文件路径正确
df_station = pd.read_excel(excel_file, sheet_name='station')
df_curve = pd.read_excel(excel_file, sheet_name='curve')
# 打印表头
print(df_curve.columns.tolist())

###------------------EBI------------------------------
df_grad = pd.read_excel(excel_file, sheet_name='grad')
df_baseInfor = pd.read_excel(excel_file, sheet_name='BasicInfo')

# 提取字段 常用制动率（m/s2） 的值
ATO_value = df_baseInfor.loc[0, 'ATO余量（km/h）']
print(f"ATO余量（km/h）：{ATO_value}")

traction_cut_off_delay = df_baseInfor.loc[0, '牵引切断延时/s']
traction_acceleration =  df_baseInfor.loc[0, '牵引加速度（m/s2）']
brake_establish_delay = df_baseInfor.loc[0, '制动建立时延/s']
 
# 加速度单位换算 
traction_acceleration = traction_acceleration*3.6
###--------------------End-----------------------------------

# 初始化数据点列表
x_values = []
y_values = []

# 合并 station 和 curve 数据
for index, row in df_station.iterrows():
    x_values.extend([row['限速起点（m）'], row['限速终点（m）']])
    y_values.extend([row['限速值（km/h）'], row['限速值（km/h）']])

for index, row in df_curve.iterrows():
    x_values.extend([row['限速起点（m）'], row['限速终点（m）']])
    y_values.extend([row['限速值（km/h）'], row['限速值（km/h）']])

# 按照 x 值排序
sorted_data = sorted(zip(x_values, y_values))
x_values_sorted, y_values_sorted = zip(*sorted_data)

# 插入缺失的 y 值，默认设置为 87
final_x_values = []
final_y_values = []
previous_y = None
# previous_y = 87

for x, y in sorted_data:
    if x == 18975.905:
        final_x_values.append(previous_x)
        final_y_values.append(87)  # 设置缺失的 y 值为 87
        final_x_values.append(x)
        final_y_values.append(87)  # 设置缺失的 y 值为 87

    else:
        if previous_y is not None and y != previous_y:
            if x > previous_x:
                # 插入从 previous_x 到 x 之间的缺失值
                final_x_values.append(previous_x)
                final_y_values.append(87)  # 设置缺失的 y 值为 87
                final_x_values.append(x)
                final_y_values.append(87)  # 设置缺失的 y 值为 87
    final_x_values.append(x)
    final_y_values.append(y)

    previous_x = x
    previous_y = y

###------------------EBI------------------------------
# 找出 y 值下跌的地方[]
falling_points = []
rising_points = []
rising_points_temp=[]
# 记录y值下跌的前一个速度
pre_falling_points = []
for i in range(1, len(final_y_values)):
    if final_y_values[i] < final_y_values[i - 1]:
        # EBI的速度值要减去ATO余量（km/h）
        falling_points.append((final_x_values[i], final_y_values[i])-ATO_value)
        pre_falling_points.append((final_x_values[i], final_y_values[i-1])-ATO_value)
    elif final_y_values[i] > final_y_values[i - 1]:
         # EBI的速度值要减去ATO余量（km/h）
        rising_points_temp.append((final_x_values[i], final_y_values[i])-ATO_value)
        rising_points.append((final_x_values[i], final_y_values[i])-ATO_value)

# 在开头添加新的点
pre_falling_points.insert(0, (357, 60))
rising_points_temp.insert(0, (357, 60))
falling_points.insert(0, (357, 60))

# ------------------------------------------------------------------------

# 绘制限速区间图
fig, ax = plt.subplots(figsize=(82, 15))


# 绘制连续的折线，这个是绘制静态限速
for i in range(1, len(final_x_values)):
    if final_x_values[i] == final_x_values[i - 1] or final_y_values[i] == final_y_values[i - 1]:
        ax.plot([final_x_values[i - 1], final_x_values[i]], [final_y_values[i - 1], final_y_values[i]], color='darkblue')

        # 在两个端点之间生成一系列等间隔的点，并将这些点的坐标存储起来

#下面开始画sbi
# 创建一个空列表，用于存储线上所有的点的坐标
line_points = []
# 创建一个空字典，用于存储每个 x 值对应的最小 y 值
x_to_y_min = {}
a=0.5
for i in range(1, len(final_x_values)):
    if final_x_values[i] == final_x_values[i - 1] or final_y_values[i] == final_y_values[i - 1]:
        #这里是在画sbi的顶棚区
        ax.plot([final_x_values[i - 1], final_x_values[i]], [final_y_values[i - 1]-8, final_y_values[i]-8], color='green')
        #我要记录水平顶棚区的y值，用于求最底下的线（垂直的不用记录了）
        if final_y_values[i] == final_y_values[i - 1]:
            # 找到 x_values 的起始点和结束点，使它们都是 0.5 的倍数
            start_multiple_of_half = np.ceil(final_x_values[i - 1] / 0.5) * 0.5
            end_multiple_of_half = np.floor(final_x_values[i] / 0.5) * 0.5

            # 生成 x_values，使其从起始点到结束点，以 0.5 为步长
            x_values = np.arange(start_multiple_of_half, end_multiple_of_half + 0.5, step=0.5)

            # 生成与 x_values 相同长度的 y_values，使其保持与 final_y_values 相同
            y_values = np.full_like(x_values, final_y_values[i - 1] - 8)
            # 更新 x_to_y_min 字典中的值，保留每个 x 对应的最小 y 值
            for x, y in zip(x_values, y_values):
                if x not in x_to_y_min or y < x_to_y_min[x]:
                    x_to_y_min[x] = y

    #如果下降的话，涉及到曲线了
    if final_y_values[i] < final_y_values[i - 1]:
        # x_end = final_x_values[i]
        # x_start = x_end - 400
        # x_curve = np.linspace(x_start, x_end, 100)
        # y_curve = np.sqrt(2 * a * (x_end - x_curve) + (final_y_values[i]-8) * (final_y_values[i]-8) / (3.6 * 3.6)) * 3.6
        # ax.plot(x_curve, y_curve, color='green')
        # 计算曲线起点和终点，使其都是 0.5 的倍数
        x_end = final_x_values[i]
        x_start = np.floor((x_end - 400) / 0.5) * 0.5  # 调整起点为最接近的 0.5 的倍数

        # 生成曲线上的点
        x_curve = np.arange(x_start, x_end + 0.1, 0.5)  # 步长为 0.5
        y_curve = np.sqrt(2 * a * (x_end - x_curve) + (final_y_values[i] - 8) ** 2 / (3.6 ** 2)) * 3.6

        # 绘制曲线
        ax.plot(x_curve, y_curve, color='green')

        # # 更新 x_to_y_min 字典中的值，保留每个 x 对应的最小 y 值
        # for x, y in zip(x_curve, y_curve):
        #     if x not in x_to_y_min or y < x_to_y_min[x]:

        # 更新 x_to_y_min 字典中的值，保留每个 x 对应的最小 y 值
        for x, y in zip(x_curve, y_curve):
            if x not in x_to_y_min or y < x_to_y_min[x]:
                x_to_y_min[x] = y

# 在每个车站终点前800米范围内绘制根号下2ax的曲线，a=1.2
a = 0.5
for index, row in df_station.iterrows():
    # x_end = row['限速终点（m）']
    # x_start = x_end - 500
    # x_curve = np.linspace(x_start, x_end, 100)
    # y_curve = np.sqrt(2 * a * (x_end - x_curve))*3.6
    x_end = row['限速终点（m）']
    x_start = np.floor((x_end - 500) / 0.5) * 0.5  # 调整起点为最接近的 0.5 的倍数

    # 生成曲线上的点
    x_curve = np.arange(x_start, x_end + 0.1, 0.5)  # 步长为 0.5
    y_curve = np.sqrt(2 * a * (x_end - x_curve)) * 3.6

    ax.plot(x_curve, y_curve, color='green')

    for x, y in zip(x_curve, y_curve):
        if x not in x_to_y_min or y < x_to_y_min[x]:
            x_to_y_min[x] = y

# 按照 x 值从小到大排序键值对
sorted_line_points = sorted(x_to_y_min.items())
# print("这是sorted_line_point")
# print(sorted_line_points)
# 解压缩排序后的点
sorted_x_values, sorted_y_values = zip(*sorted_line_points)
# plt.plot(sorted_x_values, sorted_y_values, color='yellow')
for i in range(1, len(sorted_x_values)):
    ax.plot([sorted_x_values[i - 1], sorted_x_values[i]], [sorted_y_values[i - 1], sorted_y_values[i]], color='yellow')
        # 在两个端点之间生成一系列等间隔的点，并将这些点的坐标存储起来



# 为每个限速区间绘制一条水平线
# 这个是用红色表示它有的数据，暂时这样方便检查有误的地方
for index, row in df_station.iterrows():
    line_color = 'blue'  # 设置线条颜色
    ax.plot([row['限速起点（m）'], row['限速终点（m）']], [row['限速值（km/h）'], row['限速值（km/h）']], label=row['站台名'],color='red')
for index, row in df_curve.iterrows():
     ax.plot([row['限速起点（m）'], row['限速终点（m）']], [row['限速值（km/h）'], row['限速值（km/h）']], color='red')

# 间隔时间
Interval_times = [87.51, 71.27, 124.99, 99.48, 78.67, 94.40, 78.25, 84.61, 90.24, 99.35, 94.62, 98.90, 74.06,
                  77.12, 82.53, 105.38, 79.65, 122.91, 96.34, 85.98,0.0]

# 添加正常的横坐标刻度
ax.set_xticks(range(0, int(df_station['限速终点（m）'].max()) + 1000, 1000))

# 在站台位置添加矩形和站台名称
for index, row in df_station.iterrows():
    rect_start = row['限速起点（m）']
    rect_end = row['限速终点（m）']
    rect_width = rect_end - rect_start
    rect = Rectangle((rect_start, -10), rect_width, 10, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.5)
    ax.add_patch(rect)
    midpoint = (rect_start + rect_end) / 2
    ax.text(midpoint, -15, row['站台名'], ha='center', fontproperties=font_prop)
    time_str = str(row['站停时间（s）']) + 's'
    ax.text(midpoint, -18, time_str, ha='center', fontproperties=font_prop)
    interval_time_str = str(Interval_times[index]) + 's'
    ax.text(midpoint+500, -10, interval_time_str, ha='center', fontproperties=font_prop, color='red')

ax.text(90, -18, '站停时间（s）:', ha='center', fontproperties=font_prop)
ax.text(90, -10, 'Run Time（s）:', ha='center', fontproperties=font_prop)

# ----------------------EBI------------------------------
# 记录求得的交点
EBI_intersection_points_up = []
EBI_intersection_points_constant = []
EBI_intersection_points_down = []
# 创建大列表用于存储所有的 L 值和 v 值
all_L_values = []
all_v_values = []

# 使用 zip 并行遍历 pre_falling_points 和 falling_points  
for (rise_L0, rise_v0),(rise_t_L0, rise_t_v0), (pre_L0, pre_v0), (L0, v0) in zip(rising_points,rising_points_temp,pre_falling_points, falling_points):  
    # 牵引的峰值速度
    reclosing_velocity = pre_v0 + 0.5 * traction_acceleration * np.power(traction_cut_off_delay, 2)  
  
    # 从 0 到 L0 的 L 值  
    L_values = np.linspace(L0 - 500, L0, 500)  
  
    # 计算对应的 v 值  
    v_values = v0 + np.sqrt(2 * 8 * (L0 - L_values))  
  
    # 找到最接近 reclosing_velocity 的 v_value 的索引  
    closest_index = np.argmin(np.abs(v_values - reclosing_velocity))  
  
    # 使用索引获取最接近的 L_value 和 v_value  
    closest_L_value = L_values[closest_index]  
    closest_v_value = v_values[closest_index]  

    #-----开始绘制减速曲线-----
    # 使用索引获取从 closest_index 到末尾的子集
    subset_L_values = L_values[closest_index:]
    subset_v_values = v_values[closest_index:]    

    # 计算匀速部分
    constant_speed_y = closest_v_value
    constant_speed_L = closest_L_value - closest_v_value * brake_establish_delay

    # v^2 - v0^2 = 2as 牵引加速阶段的行驶距离
    delet_up_distance = (math.pow(closest_v_value, 2) - math.pow(pre_v0, 2)) / (2 * traction_acceleration)
    up_speed_start_L = constant_speed_L - delet_up_distance

    # 将这个点（L_value, v_value）添加到 EBI_intersection_points 列表中  
    # 开始加速的交点
    EBI_intersection_points_up.append((up_speed_start_L, pre_v0)) 
    # 开始匀速的交点
    EBI_intersection_points_constant.append((constant_speed_L, constant_speed_y))
    # 开始减速的交点
    EBI_intersection_points_down.append((closest_L_value, closest_v_value))  

    #-----开始绘制加速曲线-----    
    # 从 up_speed_start_L 到 constant_speed_L 的 L 值  
    L_values_up = np.linspace(up_speed_start_L, constant_speed_L - 5, 300)  # 加入曲线平滑处理
  
    # 计算对应的 v 值  
    v_values_up = pre_v0 + np.sqrt(2 * 0.005 * (L_values_up - up_speed_start_L))
    
    # 在末尾添加新值
    L_values_up = np.append(L_values_up, constant_speed_L)
    v_values_up = np.append(v_values_up, constant_speed_y)   

    
    #-----开始绘制匀速曲线-----
    L_values_constant = np.linspace(constant_speed_L, closest_L_value, 300)  # 加入曲线平滑处理
      # 计算对应的 v 值  
    v_values_constant = np.full_like(L_values_constant, constant_speed_y)

    #------补充空白部分--------
    # 低速阶段
    L_values_low = np.linspace(L0, rise_L0, 400)  # 加入曲线平滑处理
    v_values_low = np.linspace(v0, v0, 400)  # 加入曲线平滑处理
    # 高速阶段
    L_values_high = np.linspace(rise_t_L0, up_speed_start_L, 100)  # 加入曲线平滑处理
    v_values_high = np.full_like(L_values_high, rise_v0)  # 加入曲线平滑处理

    print(rise_t_L0, up_speed_start_L)
    
    all_L_values.extend(L_values_high)
    all_v_values.extend(v_values_high)

    # 拼接加速曲线数据
    all_L_values.extend(L_values_up)
    all_v_values.extend(v_values_up)

    # 拼接匀速曲线数据
    all_L_values.extend(L_values_constant)
    all_v_values.extend(v_values_constant)

    # 拼接减速曲线数据
    all_L_values.extend(subset_L_values)
    all_v_values.extend(subset_v_values)

    # 拼接顶棚数据
    all_L_values.extend(L_values_low)
    all_v_values.extend(v_values_low)    


# 绘制拼接后的完整曲线 EBI
ax.plot(all_L_values, all_v_values, color='green')

plt.xlabel('距离（m）', fontproperties=font_prop)
plt.ylabel('限速值（km/h）', fontproperties=font_prop)
plt.title('限速区间图', fontproperties=font_prop)
plt.grid(True)
plt.ylim(bottom=-20)  # 确保底部有足够的空间显示矩形和站台名称

# 模拟列车运动
def simulate_train_movement(start_position, start_speed, x_values, y_values, max_speed=87, accel=0.8, traction_accel=0.5):
    positions = [start_position]
    speeds = [start_speed]

    current_speed = start_speed * 1000 / 3600  # 转换为 m/s
    current_position = start_position
    dt = 1  # 时间步长，秒

    max_speed = max_speed * 1000 / 3600  # 转换为 m/s

    # iteration = 0
    while current_speed < max_speed:
        if current_speed > max_speed:  # 限制最大速度
            break
        else:
            if current_speed < 40 * 1000 / 3600:  # 40 km/h in m/s
                current_speed += accel * dt
            else:
                current_speed += traction_accel * dt

        current_position += current_speed * dt
        positions.append(current_position)
        speeds.append(current_speed * 3600 / 1000)  # 转换为 km/h

    return positions, speeds

final_x = set(df_station['限速终点（m）'])
# 处理特殊值
additional_values = {1137.5: 55, 6974.5: 76, 18822: 72, 20180.5: 62, 22784: 72}

# 创建合并的x_values和y_values
train_x = list(final_x) + list(additional_values.keys())
train_y = [0] * len(final_x) + list(additional_values.values())

# 将x和y值结合起来去重
unique_coords = list(set(zip(train_x, train_y)))

# 按x值排序
unique_coords.sort()

# 分离x和y值
train_x = [round(coord[0] * 2) / 2 for coord in unique_coords]  # 调整起点为最接近的 0.5 的倍数
train_y = [coord[1] for coord in unique_coords]

coordinates = []

# 从每个限速终点开始模拟
for i in range(len(train_x)):
    start_position = train_x[i]
    start_speed = train_y[i]

    positions, speeds = simulate_train_movement(start_position, start_speed, train_x, train_y)
    
    # 使用线性插值
    f_speed = interp1d(positions, speeds)
    # 生成每隔0.5米的位置数据
    interpolated_positions = np.arange(min(positions), max(positions), 0.5)
    # 计算对应的速度
    interpolated_speeds = f_speed(interpolated_positions)
    # 输出每隔0.5米的纵坐标
    for pos, speed in zip(interpolated_positions, interpolated_speeds):
        coordinate = (pos, speed)
        coordinates.append(coordinate)
        # print(f"位置 {pos:.2f}, 速度 {speed:.2f}")
    
    # ax.plot(positions, speeds, color='pink')

 
merged_coordinates = []

# 找到从横坐标为515开始的索引
start_index = 0
for i, (x, _) in enumerate(sorted_line_points):
    if x >= 515:
        start_index = i
        break

# 合并两个数组
merged_coordinates.extend(sorted_line_points[start_index:])
for x, y in coordinates:
    if x not in [coord[0] for coord in merged_coordinates]:
        merged_coordinates.append((x, y))
    else:
        for i, (cx, cy) in enumerate(merged_coordinates):
            if cx == x:
                merged_coordinates[i] = (x, min(y, cy))

# # 按照横坐标从小到大排序
merged_coordinates.sort(key=lambda coord: coord[0])

x_merged = [point[0] for point in merged_coordinates]
y_merged = [point[1] for point in merged_coordinates]
ax.plot(x_merged, y_merged, color='pink')

# with open('merged_coordinates.txt', 'w') as file:
#     for x, y in merged_coordinates:
#         file.write(f"{x},{y}\n")
# 计算时间

# time_x = list(final_x)
# time_x = [round(coord * 2) / 2 for coord in time_x]

# 找到所有 y_merged 为 0 的索引
# zero_indices = [i for i, y in enumerate(y_merged) if y == 0]
# 输出索引
# print(zero_indices)

# plt.legend()
plt.show()




# (1137.5,55)/(6974.5,76)/(18822,72)/(20180.5,62)/(22784,72)