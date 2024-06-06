import pandas as pd
def calculate_interval_time(distances, velocities):
    interval_times = []
    start_index = 0
    end_index = 0

    for i in range(len(velocities)):
        if velocities[i] == 0:
            end_index = i
            # 计算当前区段的用时
            interval_time = calculate_time(distances[start_index:end_index + 1], velocities[start_index:end_index + 1])
            interval_times.append(interval_time)
            print("start_index 是",start_index,"   end_index 是",(end_index + 1))
            print()
            # 更新起始索引为下一个区段的起始位置
            start_index = i + 1

    return interval_times


def calculate_time(distances, velocities):
    total_time = 0.0
    # print("distances是",distances)
    # print("velocities是", velocities)
    for i in range(1, len(distances)):
        delta_distance = distances[i] - distances[i - 1]
        avg_velocity = (velocities[i] + velocities[i - 1]) / 2 * 1000 / 3600  # 转换为 m/s
        if avg_velocity != 0:
            time = delta_distance / avg_velocity
            total_time += time
    return total_time


# 从文件中读取数据
loaded_coordinates = []
with open('merged_coordinates.txt', 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        loaded_coordinates.append((x, y))

# 恢复 distances 和 velocities
loaded_coordinates.sort(key=lambda coord: coord[0])
distances = [point[0] for point in loaded_coordinates]
velocities = [point[1] for point in loaded_coordinates]

# 计算区间内的用时
interval_times = calculate_interval_time(distances, velocities)
# print("Interval times:", interval_times)
print("Interval times:", ', '.join([f'{time:.2f}' for time in interval_times]))


excel_file = '线路条件数据.xlsx'  # 确保文件路径正确
df_station = pd.read_excel(excel_file, sheet_name='station')

wait_time = df_station['站停时间（s）'].values
# print("wait_time times:", wait_time)

total_wait_time = wait_time.sum()
total_interval_times = sum(interval_times)

v = (distances[len(loaded_coordinates)-1]-distances[0])/(total_wait_time+total_interval_times-25) # 减了最后一站的停站时间
print("全程平均速度为:", 3.6*v,'km/h')


