from matplotlib import pyplot as plt

def draw_agglomerative_clustering_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Flowchart nodes
    nodes = [
        "开始",
        "将每个数据点视为一个簇",
        "计算所有簇之间的距离",
        "合并最接近的两个簇",
        "达到指定的簇数量或条件？",
        "结束",
        "是",
        "否"
    ]

    # Node positions
    positions = [
        (0.5, 0.9), (0.5, 0.7), (0.5, 0.5), (0.5, 0.3), (0.5, 0.1),
        (0.5, -0.1), (0.7, 0.1), (0.3, 0.1)
    ]

    # Draw nodes
    for node, pos in zip(nodes, positions):
        bbox_props = dict(boxstyle="round,pad=0.3", ec="black", lw=2, fc="white")
        ax.text(pos[0], pos[1], node, ha="center", va="center", size=12, bbox=bbox_props)

    # Draw arrows and lines
    arrows = [
        (positions[0], positions[1]), (positions[1], positions[2]),
        (positions[2], positions[3]), (positions[3], positions[4]),
        (positions[4], positions[6]), (positions[6], positions[1]),
        (positions[4], positions[7]), (positions[7], positions[5])
    ]

    for start, end in arrows:
        ax.annotate("", xy=end, xycoords='data', xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    return fig, ax

fig, ax = draw_agglomerative_clustering_flowchart()
plt.show()
