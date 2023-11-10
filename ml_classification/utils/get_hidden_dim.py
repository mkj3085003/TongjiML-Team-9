''' 以下是为了完成 report 所添加的代码 '''

# # 提前输出模型参数数量，以便调整网络架构
# total_params = (
#         (input_dim + 1) * hidden_dim +
#         (hidden_dim + 1) * hidden_dim * (hidden_layers - 1) +
#         (hidden_dim + 1) * 41
# )
# print(f'Total params: {total_params}')


def get_dest_dim(input_dim, output_dim, hidden_layers, dest_hidden_layers, hidden_dim):
    '''获取目标网络隐藏层的维度（总参数量接近于原网络）'''
    # 计算一元二次方程的系数 a,b,c
    a = dest_hidden_layers - 1  # a = l_d - 1
    b = input_dim + output_dim + dest_hidden_layers  # b = i + o + l_d
    c = - (hidden_layers - 1) * (hidden_dim ** 2) - (
                input_dim + output_dim + hidden_layers) * hidden_dim  # c = - (l - 1) * (d ** 2) - (i + o + l) * d

    # 计算分子中的平方根部分，即 b^2-4ac
    sqrt_part = (b ** 2) - 4 * a * c

    # 计算两个解，一个是加号，一个是减号，即(-b±√(b^2-4ac))/(2a)
    d_d_plus = (-b + sqrt_part ** (0.5)) / (2 * a)
    d_d_minus = (-b - sqrt_part ** (0.5)) / (2 * a)

    # 返回两个解的元组
    return (d_d_plus, d_d_minus)


# # 设置你想要的目标网络隐藏层数量
# dest_hidden_layers = 9

# # 获取对应的维数
# dest_hidden_dim, _ = get_dest_dim(input_dim, 41, hidden_layers, dest_hidden_layers, hidden_dim)
# print(f"若将隐藏层网络层数改为: {dest_hidden_layers}，则维数应当改为: {round(dest_hidden_dim)}", )