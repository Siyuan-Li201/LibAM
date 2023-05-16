
import matplotlib.pyplot as plt

def draw_dot_graph(file_name, save_path="", node_num=200):
    with open(file_name, "r") as f:
        d = eval(f.readline())
    truths = d[1]
    scores = d[0]
    true_sample = []
    false_sample = []
    for i in range(len(truths)):
        truth = truths[i]
        score = scores[i]
        if truth == 1 and len(true_sample) < node_num:
            true_sample.append(score)
        if truth == -1 and len(false_sample) < node_num:
            false_sample.append(score)
    index = [[i for i in range(len(true_sample))], [i + len(true_sample) for i in range(len(false_sample))]]
    # true_sample.extend(false_sample)

    # samples = np.array([true_sample, false_sample])
    # index = np.array([[0 for i in range(len(true_sample))], [1 for i in range(len(false_sample))]])
    plt.figure(figsize=(8, 8))
    plt.scatter(false_sample, index[0], c='red', label='Negative sample')
    plt.scatter(true_sample, index[1], c='green', label='Positive sample')
    # plt.axvline(0.72, c="yellow", ls="-.")
    plt.legend()
    plt.savefig(save_path)

if __name__ == '__main__':
    draw_dot_graph("train_dot_200.txt", "train_dot.jpg")
    draw_dot_graph("test_dot_200.txt", "test_dot.jpg")
    draw_dot_graph("valid_dot_200.txt", "valid_dot.jpg")
