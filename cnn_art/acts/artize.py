import numpy as np
import caffe2.python.model_helper as model_helper
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

from zoo_helper import make

def prep(x):
    x = x.astype(np.float32)/255. - [.403,.456,.478]
    x = x[np.newaxis,...].astype(np.float32).transpose(0,3,1,2)
    return x


def spect(rep, cmod=0, sig_noise=None,C=64):
    _,H,W = rep.shape
    #out = np.zeros([H,W,3], dtype=np.uint8)
    out = np.zeros([H,W,3], dtype=np.float32)

    mus = np.mean(rep.transpose(1,2,0).reshape([-1,C]),axis=0)
    sigs = np.std(rep.transpose(1,2,0).reshape([-1,C]),axis=0)
    print(sigs)

    c2rgb = [mpl_colors.hsv_to_rgb((((c+cmod)%C)/(C),1.,1.)) for c in range(C)]
    c2rgb = [z/np.linalg.norm(z) for z in c2rgb]
    #c2rgb = [z*z*z for z in c2rgb] # Raise bandwidth-falloff
    #c2rgb = [z/np.linalg.norm(z) for z in c2rgb]
    #for c in c2rgb: print(c)
    #print('\n\n\n')
    #for c in c2rgb: print(c)
    #c2rgb = [(rgb*255).astype(np.uint8) for rgb in c2rgb]

    '''
    for y in range(H):
        for x in range(W):
            for c in range(C):
                val = (rep[c,y,x]-mus[c]) / sigs[c]
                #val = min(val, 1) / 64.
                val = val / 64
                #out[y,x] += (c2rgb[c] * val).astype(np.uint8)
                out[y,x] += (c2rgb[c] * val)
    '''

    sigs = sigs * sig_noise # Tamer
    #sig_noise *= abs(np.random.randn(C)*.0001+1.)

    #sigs = sigs * (sig_noise+.3*(np.random.randn(C))) # Prety cool+chaotic
    #sigs = sigs * sig_noise * (.0000001+abs(np.random.randn(C))) # Prety cool+chaotic
    #sigs = sigs * sig_noise * (.005+abs(np.random.randn(C))) # Prety cool+chaotic
    #sigs = sigs * np.clip(.0001,999,abs(np.random.randn(C))) # Prety cool+chaotic


    for c in range(C):
        vals = (rep[c,:,:]-mus[c]) / sigs[c]
        #val = min(val, 1) / 64.
        vals = vals / 64
        #print(vals.shape)
        #out[y,x] += (c2rgb[c] * val).astype(np.uint8)
        out += (np.tile(c2rgb[c],(H,W,1)) * np.tile(vals[...,np.newaxis],(1,1,3)))


    #return out
    #out = (out/255.).astype(np.uint8)
    print('min',np.min(out))
    print('max',np.max(out))
    # Order of clip/min/max and the clip values are VERY important
    return out


def main():
    sz = 1200
    #sz = 224
    mh = make(img_size=sz)

    #img_name = 'shanghai'
    img_name = 'gorge'
    #img_name = 'lijiang'
    img = cv2.imread('/home/slee/Downloads/'+img_name+'.jpg')
    img = cv2.resize(img,(sz,sz))
    x = prep(img)
    print(x.shape)

    workspace.FeedBlob('data', x)
    workspace.RunNet(mh.net)

    y0 = workspace.FetchBlob('conv1')[0]
    #y1 = workspace.FetchBlob('res3_3_branch2c')[0]
    y1 = workspace.FetchBlob('res2_2_branch2c_bn')[0]
    #y1 = workspace.FetchBlob('res5_2_branch2c_bn')[0]

    C0 = y0.shape[0]//2
    C1 = y1.shape[0]//4

    sig_noise0 = abs(np.random.randn(C0)+.01) * 10
    sig_noise1 = np.random.randn(C1) * .3

    for cmod in range(C0):
        z0 = spect(y0, cmod=cmod, sig_noise=sig_noise0,C=C0)
        z1 = spect(y1, cmod=cmod%C1, sig_noise=sig_noise1*10,C=C1) * .1
        z1 = cv2.resize(z1,(z0.shape[1],z0.shape[0]),0,0, cv2.INTER_LANCZOS4)
        print(z1.shape)

        z1 /= np.max(z1) * 8

        yy = (z0 + z1)
        #yy = z1

        #yy = yy.clip(np.mean(yy)-np.random.randn()*5,8000)
        yy = yy.clip(0,8000)
        yy -= np.min(yy)
        yy /= np.max(yy)
        yy = (yy * 255).astype(np.uint8)

        cv2.imwrite('out/{}{:03d}.jpg'.format(img_name,cmod),yy)


    #plt.imshow(y),plt.show()







main()
