import numpy as np,matplotlib.pyplot as plt

def model_visualise(EPOCHS, BATCHES, batch_cost,batch_train_ac,batch_test_ac):
    # model visualisation
    colors = ["#8A2BE2", "#F8A028", "#3EEE2E", "#00A5A9", "#FF0000"]
    plt.style.use("seaborn")
    # plt.style.use('dark_background')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # batch cost
    ax0 = ax[0][0]
    for i in range(EPOCHS):
        ax0.plot(range(0, BATCHES), batch_cost[i * BATCHES:(i + 1) * BATCHES], color=colors[i % len(colors)],
                 label=str(i + 1))

    ax0.set_title("Batch cost for all epochs")
    ax0.set_xlabel("no. of batch")
    ax0.set_ylabel("cost of that batch")
    ax0.legend()
    # train accuracy
    ax0 = ax[0][1]
    for i in range(EPOCHS):
        ax0.plot(range(0, BATCHES), batch_train_ac[i * BATCHES:(i + 1) * BATCHES], color=colors[i % len(colors)],
                 label=str(i + 1))

    ax0.set_title("Train accuracies")
    ax0.set_xlabel("no. of batch")
    ax0.set_ylabel("accuracy that batch(%)")
    ax0.legend()

    # test accuracy
    ax0 = ax[1][0]
    for i in range(EPOCHS):
        ax0.plot(range(0, BATCHES), batch_test_ac[i * BATCHES:(i + 1) * BATCHES], '--', color=colors[i % len(colors)],
                 label=str(i + 1))

    ax0.set_title("Test accuracies")
    ax0.set_xlabel("no. of batch")
    ax0.set_ylabel("accuracy that batch(%)")
    ax0.legend()

    # train/test value
    ax0 = ax[1][1]
    for i in range(EPOCHS):
        ax0.plot(range(0, BATCHES), np.array(batch_train_ac[i * BATCHES:(i + 1) * BATCHES])/np.array(batch_test_ac[i * BATCHES:(i + 1) * BATCHES]), '-', color=colors[i % len(colors)],
                 label=str(i + 1))

    ax0.set_title("Variance(Train/Test)")
    ax0.set_xlabel("no. of batch")
    ax0.set_ylabel("value")
    ax0.legend()

    plt.tight_layout()
    plt.show()


def cost(y,yp):
    c = 0
    for i in range(len(y)):
        c += -1*y[i]*np.log(yp[i])
    return c

def std(a):
    return (a-np.mean(a))/np.std(a)