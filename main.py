import cv2 as cv
import numpy as np
import time

np.set_printoptions(suppress=True)


# 生成图像的基本参数，如R块的索引，D块的索引等
def image_parameter(img, rx, delta):
    ri_index = np.array([0, 0, 0])    # Ri y, x, index
    ri_index[0] = int(img.shape[0] / rx)
    ri_index[1] = int(img.shape[1] / rx)
    ri_index[2] = int(ri_index[0] * ri_index[1])
    di_index = np.array([0, 0, 0])
    di_index[0] = int((img.shape[0] - 2 * rx) / delta + 1)
    di_index[1] = int((img.shape[1] - 2 * rx) / delta + 1)
    di_index[2] = int(di_index[0] * di_index[1])
    return ri_index, di_index


def codebook(img, rx, dx, delta, di_index):     # 生成码本
    domain = np.zeros((di_index[2], dx, dx))
    index = 0
    range_x = np.arange(0, img.shape[0] - dx + 1, delta)
    range_y = np.arange(0, img.shape[1] - dx + 1, delta)
    for cols in range_y:    # 滑动获取未压缩码本D
        for rows in range_x:
            domain[index] = img[cols: cols+dx, rows:rows+dx]
            index += 1
    idx = np.arange(di_index[2])
    idx.shape = (di_index[0], di_index[1])
    omega = np.zeros((di_index[2], rx, rx))
    for y in range(di_index[0]):    # 4邻域平均
        for x in range(di_index[1]):
            for l_rx in range(rx):
                for k_rx in range(rx):
                    omega[idx[y, x], l_rx, k_rx] = domain[idx[y, x], 2*l_rx:2*l_rx+2, 2*k_rx:2*k_rx+2].sum()/4
    # 下面是八种变换的代码
    omega_8 = np.zeros((di_index[2], 8, rx, rx))
    for d_idx in range(di_index[2]):
        omega_8[d_idx, 0] = omega[d_idx]   # 恒等变换
        omega_8[d_idx, 1] = np.fliplr(omega[d_idx])    # Y轴对称
        omega_8[d_idx, 2] = np.flipud(omega[d_idx])    # X轴对称
        omega_8[d_idx, 3] = np.rot90(np.fliplr(omega[d_idx]))  # 主对角线翻转
        omega_8[d_idx, 4] = np.rot90(np.flipud(omega[d_idx]))  # 副对角线翻转
        omega_8[d_idx, 5] = np.rot90(omega[d_idx], k=1)    # 逆时针90°旋转
        omega_8[d_idx, 6] = np.rot90(omega[d_idx], k=2)    # 逆时针180°旋转
        omega_8[d_idx, 7] = np.rot90(omega[d_idx], k=3)    # 逆时针270°旋转
    # print(omega)
    return omega_8


# 编码过程，每个R块逐个匹配码本块，输出每个R块的分形编码[码本的索引，等距变换的参数，参数s，参数o]
def code_ing(img, code_book, rx, ri_index, di_index):
    r_idx = np.arange(ri_index[2])
    r_idx.shape = (ri_index[0], ri_index[1])
    code = np.zeros((ri_index[2], 4))
    d_idx = np.arange(di_index[2])
    d_idx.shape = (di_index[0], di_index[1])
    for r_y_idx in range(ri_index[0]):
        for r_x_idx in range(ri_index[1]):
            ri = img[r_y_idx * rx:r_y_idx * rx + rx, r_x_idx * rx:r_x_idx * rx + rx]
            ri_m = ri.sum()
            rr2 = np.square(np.linalg.norm(ri))
            par = np.array([0, 0, 0, 0])
            h_min = np.inf
            for d_idx in range(di_index[2]):
                di = code_book[d_idx, 0]
                di_m = di.sum()
                di_d2 = np.square(np.linalg.norm(di))
                ri_di = np.vdot(ri, di)
                rx2 = np.square(rx)
                st = (rx2*ri_di - ri_m*di_m) / (rx2*di_d2 - np.square(di_m))
                if -1 < st < 1:
                    ot = (ri_m - st*di_m) / rx2
                    h = (rr2 + st*(st*di_d2 - 2*ri_di + 2*ot*di_m) + ot*(rx2*ot - 2*ri_m)) / rx2
                    if h < h_min:
                        # h_min = h
                        for conversion in range(8):
                            di = code_book[d_idx, conversion]
                            di_m = di.sum()
                            di_d2 = np.square(np.linalg.norm(di))
                            ri_di = np.vdot(ri, di)
                            st = (rx2*ri_di - ri_m*di_m) / (rx2*di_d2 - np.square(di_m))
                            if -1 < st < 1:
                                ot = (ri_m - st * di_m) / rx2
                                h = (rr2 + st*(st*di_d2 - 2*ri_di + 2*ot*di_m) + ot*(rx2*ot - 2*ri_m)) / rx2
                                if h < h_min:
                                    par = [d_idx, conversion, st, ot]
                                    h_min = h
            code[r_idx[r_y_idx, r_x_idx]] = par
            print(r_idx[r_y_idx, r_x_idx], par)   # ri - (par[2]*code_book[par[0], par[1]] + par[3])
    return code


# 解码过程，使用全黑图像和分形码，进行迭代输出
def decode(code, iterate, shape, rx, dx, delta, ri_index, di_index):
    decode_ing = np.zeros((shape[0], shape[1]))
    r_idx = np.arange(ri_index[2])
    r_idx.shape = (ri_index[0], ri_index[1])
    di_t2 = np.zeros((rx, rx))
    for it in range(iterate):
        for r_y_idx in range(ri_index[0]):
            for r_x_idx in range(ri_index[1]):
                temp = code[r_idx[r_y_idx, r_x_idx]]
                y = int(temp[0] / di_index[0])*delta
                y1 = y+dx
                x = int(temp[0] % di_index[1])*delta
                x1 = x+dx
                di = decode_ing[y:y1, x:x1]
                for l_rx in range(rx):
                    for k_rx in range(rx):
                        di_t2[l_rx, k_rx] = di[2 * l_rx:2 * l_rx + 2, 2 * k_rx:2 * k_rx + 2].sum() / 4
                if temp[1] == 0:
                    di_t2 = di_t2
                elif temp[1] == 1:
                    di_t2 = np.fliplr(di_t2)
                elif temp[1] == 2:
                    di_t2 = np.flipud(di_t2)
                elif temp[1] == 3:
                    di_t2 = np.rot90(np.fliplr(di_t2))
                elif temp[1] == 4:
                    di_t2 = np.rot90(np.flipud(di_t2))
                elif temp[1] == 5:
                    di_t2 = np.rot90(di_t2, k=1)
                elif temp[1] == 6:
                    di_t2 = np.rot90(di_t2, k=2)
                elif temp[1] == 7:
                    di_t2 = np.rot90(di_t2, k=3)
                decode_ing[r_y_idx*rx:r_y_idx*rx+rx, r_x_idx*rx:r_x_idx*rx+rx] = temp[2]*di_t2 + temp[3]
    return decode_ing


# 计算图像的PSNR，即峰值信噪比，这个参数越大，图像失真越小
def ps_nr(de_img, img):
    mse = np.square(img - de_img) / (256 * 256)
    ps_ = 10 * np.log10((255*255) / np.sum(mse))
    return ps_


# main函数，可调参数有R块的大小RX（一般是4或者8，满足2的次方就好），D块的大小DX（这个是RX的两倍）
# D块的位移步长Delta（这个无所谓，不过步长越大，码本越小，一般是RX的一倍或者两倍）
# 原图像的路径str_image，解码图像路径str_image1，保存的文件code_read（其内的路径和文件名可修改）
if __name__ == '__main__':
    # 参数定义
    time_star = time.time()
    RX = 4
    DX = 8
    Delta = 8
    str_image = './image/BABOO.bmp'
    str_image1 = './decode/BABOO-fractal.bmp'
    image = cv.imread(str_image, 0)
    R_index, D_index = image_parameter(image, RX, Delta)
    Omega = codebook(image, RX, DX, Delta, D_index)   # 构成码本
    Code = code_ing(image, Omega, RX, R_index, D_index)
    # 打印时间
    time_end = time.time()
    run_time = time_end - time_star
    print(f'运行时间：{run_time} s')
    np.save('./code/code484.npy', Code)
    code_read = np.load('./code/code484.npy')
    Decode = decode(code_read, 5, image.shape, RX, DX, Delta, R_index, D_index)
    ps = ps_nr(image, Decode)
    print('峰值信噪比： ')
    print(ps)
    cv.imwrite(str_image1, Decode)
    cv.imshow(str_image, image)
    cv.imshow(str_image1, cv.imread(str_image1))
    # 关闭窗口
    cv.waitKey(0)
    cv.destroyAllWindows()
