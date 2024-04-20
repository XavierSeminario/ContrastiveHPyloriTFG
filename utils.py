import os
import torchvision
import random



def list_imgs(direc='./HPyloriData/annotated_windows'):
        # os.chdir(direc)
        carpetes=os.listdir(direc)
        # imatges_senceres=[]
        imatges_cropped=[]
        for carp in carpetes:
            if carp.endswith('.png'):
                # imatges_senceres.append(skimage.io.imread(carp, as_gray=False))
                pass
            elif carp.endswith('.csv') or carp.endswith('.db'):
                pass
            else:
                # os.chdir('./'+carp)
                for file in os.listdir(direc+'/'+carp):
                    if file.endswith('.png'):
                        imatges_cropped.append((carp+"."+file).rstrip('.png'))
        return imatges_cropped


def rotate_some(imgs):
    angles = 90,180,270
    size, _, __, ___ = imgs.shape
    for i in range(size//2):
        num = random.randint(0,size-1)
        imgs[num] = torchvision.transforms.functional.rotate(imgs[num], angle=random.choice(angles))
    return imgs