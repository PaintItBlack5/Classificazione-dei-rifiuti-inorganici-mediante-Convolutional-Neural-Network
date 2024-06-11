import matplotlib.pyplot as plt
import argparse

labels = ["t_loss", "v_loss", "t_acc", "v_acc", "t_ce_loss", "v_ce_loss", "t_ce_acc", "v_ce_acc"]

DETAILS = "Dataset_3_model_2"

def main(args):
    data = importFile(args.input)
    epochs = str(len(data["t_loss"]))
    title = "Pre-Training"

    if data["t_ce_loss"][0] != 0:
        title = "Scouter Training"

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(title + " - " + epochs + " epochs - " + DETAILS, fontsize=16)

    ax[0, 0].plot(data["t_loss"], 'r')  # row=0, col=0
    ax[0, 0].set_title("Train Loss")
    ax[0, 0].set_ylim([0, 2.0])

    ax[1, 0].plot(data["t_acc"], 'b')  # row=1, col=0
    ax[1, 0].set_title("Train Accuracy")
    ax[1, 0].set_ylim([0, 1.0])

    ax[0, 1].plot(data["v_loss"], 'r')  # row=0, col=1
    ax[0, 1].set_title("Validation Loss")
    ax[0, 1].set_ylim([0, 2.0])

    ax[1, 1].plot(data["v_acc"], 'b')  # row=1, col=1
    ax[1, 1].set_title("Validation Accuracy")
    ax[1, 1].set_ylim([0, 1.0])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig('Pre-training_1.png')
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title + " - " + epochs + " epochs - " + DETAILS, fontsize=16)

    ax[0].plot(data["t_loss"], 'r', label='Train')
    ax[0].plot(data["v_loss"], 'b', label='Val')
    ax[0].set_title("Loss Function")
    ax[0].legend(loc="upper right")
    ax[0].set_ylim([0, 2.0])

    ax[1].plot(data["t_acc"], 'r', label='Train')
    ax[1].plot(data["v_acc"], 'b', label='Val')  # row=1, col=1
    ax[1].set_title("Accuracy")
    ax[1].legend(loc="lower right")
    ax[1].set_ylim([0, 1.0])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig('Pre-training_2.png')
    if data["t_ce_loss"][0] != 0:
        fig, ax = plt.subplots(2, 2)
        fig.suptitle(title + " - " + epochs + " epochs", fontsize=16)

        ax[0, 0].plot(data["t_ce_loss"], 'r')  # row=0, col=0
        ax[0, 0].set_title("Cross Entropy - Train Loss")

        ax[1, 0].plot(data["t_ce_acc"], 'b')  # row=1, col=0
        ax[1, 0].set_title("Train Attention Loss")

        ax[0, 1].plot(data["v_ce_loss"], 'r')  # row=0, col=1
        ax[0, 1].set_title("Cross Entropy - Validation Loss")

        ax[1, 1].plot(data["v_ce_acc"], 'b')  # row=1, col=1
        ax[1, 1].set_title("Validation Attention Loss")

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig('training.png')


def convertFloatArray(line):
    l = line.split(": ")
    n_line = l[1].replace("[", "")
    n_line = n_line.replace("]", "")
    n_line = n_line.split(', ')
    return [float(numeric_string) for numeric_string in n_line]


def importFile(path):
    my_file = {}
    with open(path) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            my_file[labels[idx]] = convertFloatArray(line)
    return my_file

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph SCOUTER', add_help=False)
    parser.add_argument('--input', default="data/test.txt", type=str)
    args = parser.parse_args()
    main(args)

