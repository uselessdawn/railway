import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
import math

# 设置中文字体路径
# font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
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
###-------------------------------------------------------

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
fig, ax = plt.subplots(figsize=(22, 6))

# 绘制连续的折线
for i in range(1, len(final_x_values)):
    if final_x_values[i] == final_x_values[i - 1] or final_y_values[i] == final_y_values[i - 1]:
        ax.plot([final_x_values[i - 1], final_x_values[i]], [final_y_values[i - 1], final_y_values[i]], color='darkblue')

# 为每个限速区间绘制一条水平线
# 这个是用红色表示它有的数据，暂时这样方便检查有误的地方
for index, row in df_station.iterrows():
    line_color = 'blue'  # 设置线条颜色
    ax.plot([row['限速起点（m）'], row['限速终点（m）']], [row['限速值（km/h）'], row['限速值（km/h）']], label=row['站台名'], color='red')
for index, row in df_curve.iterrows():
    ax.plot([row['限速起点（m）'], row['限速终点（m）']], [row['限速值（km/h）'], row['限速值（km/h）']], color='red')

# 添加正常的横坐标刻度
ax.set_xticks(range(0, int(df_station['限速终点（m）'].max()) + 1000, 1000))

# 在站台位置添加矩形和站台名称
for index, row in df_station.iterrows():
    midpoint = (row['限速起点（m）'] + row['限速终点（m）']) / 2
    rect = Rectangle((midpoint - 50, -10), 100, 10, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.5)
    ax.add_patch(rect)
    ax.text(midpoint, -15, row['站台名'], ha='center', fontproperties=font_prop)

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
plt.show()
