import matplotlib.pyplot as plt

def imshow(img):
    npimg = img.numpy().reshape(243, 243, 3)
    plt.imshow(npimg)
    plt.show()


def show_grid(images, size):
    fig = plt.figure()
    for i in range(size * size):
        fig.add_subplot(size, size, i + 1)
        plt.imshow(images[i].numpy().reshape(243, 243, 3))
    plt.show()