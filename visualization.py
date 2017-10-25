from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines

def render_scatter_gng(X, weights, connections, title="GNG", show=True):
    fig, ax = plt.subplots(figsize=[12.0, 8.0])
    ax.annotate(title, (0.02, 0.95), textcoords='axes fraction', color='midnightblue')
    ax.scatter(X[:, 0], X[:, 1], color='midnightblue', s=8)
    ax.patch.set_facecolor('aliceblue')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if weights is not None:
        if connections:
            for pair in connections:
                w1, w2 = weights[pair[0]], weights[pair[1]]
                line = mlines.Line2D(
                    (w1[0], w2[0]),
                    (w1[1], w2[1]),
                    color='burlywood', alpha=0.75)
                ax.add_line(line)
        ax.scatter(weights[:, 0], weights[:, 1], color='chocolate', s=90, marker=(5, 0), edgecolor="none")
    if show:
        plt.show()
    else:
        plt.savefig(title + ".png", bbox_inches='tight', pad_inches=0)
    plt.close()

def render_gng_animation(weights_history, X=None, connections_history=None, samples_history=None,
                         title="SOM Fitting", show=True, frame_divisor=1):

    fig, ax = plt.subplots(figsize=[12.0, 8.0])
    ax.annotate(title, (0.02, 0.95), textcoords='axes fraction', color='midnightblue')
    if X:
        ax.scatter(X[:, 0], X[:, 1], color='midnightblue', s=8)
    ax.patch.set_facecolor('aliceblue')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    clearlist = []

    def update(frame_number):

        print("Frame {}".format(frame_divisor*frame_number))
        history_index = (frame_divisor*frame_number)%len(weights_history)

        W = weights_history[history_index]

        while clearlist:
            c = clearlist.pop()
            c.remove()
            del c

        clearlist.append(ax.scatter(W[:, 0], W[:, 1], color='chocolate', s=90))

        if samples_history:
            x = samples_history[history_index]
            if len(x.shape) == 2:
                clearlist.append(ax.scatter(x[:, 0], x[:, 1], color='mediumblue', s=90))
            else:
                clearlist.append(ax.scatter(x[0], x[1], color='mediumblue', s=90))

        if connections_history:
            connections = connections_history[history_index]
            for pair in connections:
                w1, w2 = W[pair[0]], W[pair[1]]
                line = mlines.Line2D(
                    (w1[0], w2[0]),
                    (w1[1], w2[1]),
                    color='burlywood', alpha=0.75)
                clearlist.append(ax.add_line(line))

        clearlist.append(ax.annotate(str((frame_divisor*frame_number)%len(weights_history)), (0.02, 0.02), textcoords='axes fraction' ))

    if show:
        ani = animation.FuncAnimation(fig, update, interval=200, blit=False, save_count=(len(weights_history)//frame_divisor-1))
        plt.show()
    else:
        ani = animation.FuncAnimation(fig, update, interval=200, blit=True, save_count=(len(weights_history)//frame_divisor-1))
        ani.save(title + '.mp4', writer='ffmpeg', fps=2, bitrate=2048)

