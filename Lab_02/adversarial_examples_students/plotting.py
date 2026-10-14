import matplotlib.pyplot as plt


def plot_images(X, y, yp, M, N, grayscale=False):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    for i in range(M):
        for j in range(N):
            sample = 1 - X[i * N + j].cpu().numpy() if grayscale else X[i * N + j].cpu().numpy()
            ax[i][j].imshow(sample, cmap='gray' if grayscale else None)
            prediction = yp[i*N+j].item()
            title = ax[i][j].set_title(f'Pred: {prediction}')
            plt.setp(title, color=('g' if prediction == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    plt.show()