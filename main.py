import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
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

# 找出 y 值下跌的地方
falling_points = []
for i in range(1, len(final_y_values)):
    if final_y_values[i] < final_y_values[i - 1]:
        falling_points.append((final_x_values[i], final_y_values[i]))

# 输出下跌点
print("下跌点位置（横坐标, 纵坐标）：")
for point in falling_points:
    print(point)

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

plt.xlabel('距离（m）', fontproperties=font_prop)
plt.ylabel('限速值（km/h）', fontproperties=font_prop)
plt.title('限速区间图', fontproperties=font_prop)
plt.grid(True)
plt.ylim(bottom=-20)  # 确保底部有足够的空间显示矩形和站台名称

# 模拟列车运动
def get_speed_limit(position, x_values, y_values):
    # 获取当前位置的限速值
    for i in range(len(x_values) - 1):
        if x_values[i] <= position < x_values[i + 1]:
            return y_values[i] * 1000 / 3600  # 转换为 m/s
    return y_values[-1] * 1000 / 3600  # 转换为 m/s


def simulate_train_movement(start_position, start_speed, accel, traction_accel, x_values, y_values):
    positions = [start_position]
    speeds = [start_speed]

    current_speed = start_speed
    current_position = start_position
    dt = 0.1  # 时间步长，秒

    while current_position < max(x_values):
        if current_speed < 40 * 1000 / 3600:  # 40 km/h in m/s
            current_speed += accel * dt
        else:
            current_speed += traction_accel * dt

        current_limit = get_speed_limit(current_position, x_values, y_values)
        if current_speed > current_limit:  # 如果超过限速，调整速度
            current_speed = current_limit

        current_position += current_speed * dt
        positions.append(current_position)
        speeds.append(current_speed * 3600 / 1000)  # 转换为 km/h

        if current_position in x_values:  # 到达限速终点，停止计算
            break

    return positions, speeds

train_x = final_x_values
train_y = final_y_values

# 删除指定的 train_x 元素
value_to_remove = set(df_station['限速终点（m）'])  # 使用集合以避免重复值
indices_to_keep = [i for i, x in enumerate(train_x) if x in value_to_remove]

# 使用列表推导式创建新的train_x和train_y列表
train_x_new = [train_x[i] for i in indices_to_keep]
train_y_new = [train_y[i] for i in indices_to_keep]

# 将新列表赋值给原始变量
train_x = train_x_new
train_y = train_y_new

# 从每个限速终点开始模拟
for i in range(1, len(train_x), 2):  # 只在限速终点开始模拟
    start_position = train_x[i]
    positions, speeds = simulate_train_movement(start_position, 0, 0.8, 0.5, train_x, train_y)
    ax.plot(positions, speeds, color='pink', label=f'列车运动轨迹从{start_position}m开始')

# plt.legend()
plt.show()
