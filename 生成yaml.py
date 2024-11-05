import os

# 定义路径及参数
train_path = r'./'
val_path = r'./'
nc = 0
names = ['',]

# 生成 YAML 配置内容
config_content = f"""train: {train_path}
val: {val_path}
nc: {nc}
names: {names}
"""

# 输出文件路径
output_file = 'config.yaml'

# 写入文件
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(config_content)

print(f'配置文件已生成: {output_file}')
