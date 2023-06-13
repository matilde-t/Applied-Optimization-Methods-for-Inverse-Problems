import numpy as np


class noise:
    def __init__(self, img):
        self.img = img
        self.size = img.shape
        self.min = np.min(img)
        self.max = np.max(img)

    def gaussian(self):
        noise = np.random.normal(0, 25, size=self.size)
        return np.clip(self.img + noise, self.min, self.max)

    def poisson(self):
        noise = np.random.poisson(25, size=self.size)
        return np.clip(self.img + noise, self.min, self.max)

    def salt_pepper(self):
        max = np.prod(self.size)
        pixels = np.random.randint(max / 20, max / 10)
        img = self.img.copy()
        for i in range(pixels):
            x = np.random.randint(self.size[0])
            y = np.random.randint(self.size[1])
            img[x][y] = np.random.choice([self.min, self.max])
        return img
