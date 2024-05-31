import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

# 设置中文字体路径
font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
font_prop = FontProperties(fname=font_path)

# 全局设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取 Excel 文件中的特定工作表
excel_file = '/Users/dingyuhan/Documents/课程学习/大三下/railway/线路条件数据.xlsx'  # 确保文件路径正确
df_station = pd.read_excel(excel_file, sheet_name='station')
df_curve = pd.read_excel(excel_file, sheet_name='curve')
# 打印表头
print(df_curve.columns.tolist())
# 绘制限速区间图
# fig, ax = plt.subplots(figsize=(22, 6))

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
previous_y = 87

for x, y in sorted_data:
    if final_x_values and x != final_x_values[-1]:
        final_x_values.append(x)
        final_y_values.append(previous_y)
    final_x_values.append(x)
    final_y_values.append(y)
    previous_y = y

# 绘制限速区间图
fig, ax = plt.subplots(figsize=(22, 10))

# 绘制连续的折线
ax.plot(final_x_values, final_y_values, color='darkblue')

# # 为每个限速区间绘制一条水平线
# line_color = 'blue'  # 设置线条颜色
# for index, row in df_station.iterrows():
#     line_color = 'blue'  # 设置线条颜色
#     ax.plot([row['限速起点（m）'], row['限速终点（m）']], [row['限速值（km/h）'], row['限速值（km/h）']], label=row['站台名'],color=line_color)
# for index, row in df_curve.iterrows():
#      ax.plot([row['限速起点（m）'], row['限速终点（m）']], [row['限速值（km/h）'], row['限速值（km/h）']], color='red')

# 添加正常的横坐标刻度
ax.set_xticks(range(0, int(df_station['限速终点（m）'].max()) + 1000, 1000))

# 在站台位置添加矩形和站台名称
for index, row in df_station.iterrows():
    midpoint = (row['限速起点（m）'] + row['限速终点（m）']) / 2
    rect = Rectangle((midpoint - 50, -10), 100, 10, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.5)
    ax.add_patch(rect)
    ax.text(midpoint, -15, row['站台名'], ha='center', fontproperties=font_prop)

plt.xlabel('距离（m）', fontproperties=font_prop)
plt.ylabel('限速值（km/h）', fontproperties=font_prop)
plt.title('限速区间图', fontproperties=font_prop)
plt.grid(True)
plt.ylim(bottom=-20)  # 确保底部有足够的空间显示矩形和站台名称
plt.show()
