import cv2 as cv
import numpy as np
import time
from tqdm import tqdm

np.set_printoptions(suppress=True)


# 生成图像的基本参数，如R块的索引，D块的索引等
def image_parameter(img, rx, delta):
    ri_index = np.array([0, 0, 0])  # Ri y, x, index
    ri_index[0] = int(img.shape[0] / rx)
    ri_index[1] = int(img.shape[1] / rx)
    ri_index[2] = int(ri_index[0] * ri_index[1])
    di_index = np.array([0, 0, 0])
    di_index[0] = int((img.shape[0] - 2 * rx) / delta + 1)
    di_index[1] = int((img.shape[1] - 2 * rx) / delta + 1)
    di_index[2] = int(di_index[0] * di_index[1])
    return ri_index, di_index


def codebook(img, rx, dx, delta, di_index):  # 生成码本
    domain = np.zeros((di_index[2], dx, dx))
    index = 0
    range_x = np.arange(0, img.shape[0] - dx + 1, delta)
    range_y = np.arange(0, img.shape[1] - dx + 1, delta)
    for cols in range_y:  # 滑动获取未压缩码本D
        for rows in range_x:
            domain[index] = img[cols: cols + dx, rows:rows + dx]
            index += 1
    idx = np.arange(di_index[2])
    idx.shape = (di_index[0], di_index[1])
    omega = np.zeros((di_index[2], rx, rx))
    for y in range(di_index[0]):  # 4邻域平均
        for x in range(di_index[1]):
            for l_rx in range(rx):
                for k_rx in range(rx):
                    omega[idx[y, x], l_rx, k_rx] = domain[idx[y, x], 2*l_rx:2*l_rx+2, 2*k_rx:2*k_rx+2].sum()/4
    # 下面是八种变换的代码
    omega_8 = np.zeros((di_index[2], 8, rx, rx))
    for d_idx in range(di_index[2]):
        omega_8[d_idx, 0] = omega[d_idx]  # 恒等变换
        omega_8[d_idx, 1] = np.fliplr(omega[d_idx])  # Y轴对称
        omega_8[d_idx, 2] = np.flipud(omega[d_idx])  # X轴对称
        omega_8[d_idx, 3] = np.rot90(np.fliplr(omega[d_idx]))  # 主对角线翻转
        omega_8[d_idx, 4] = np.rot90(np.flipud(omega[d_idx]))  # 副对角线翻转
        omega_8[d_idx, 5] = np.rot90(omega[d_idx], k=1)  # 逆时针90°旋转
        omega_8[d_idx, 6] = np.rot90(omega[d_idx], k=2)  # 逆时针180°旋转
        omega_8[d_idx, 7] = np.rot90(omega[d_idx], k=3)  # 逆时针270°旋转
    # print(omega)
    return omega_8


# 编码过程，每个R块逐个匹配码本块，输出每个R块的分形编码[码本的索引，等距变换的参数，参数s，参数o]
def encoder(img, code_book, rx, ri_index, di_index):
    r_idx = np.arange(ri_index[2])
    r_idx.shape = (ri_index[0], ri_index[1])
    code = np.zeros((ri_index[2], 4))
    d_idx = np.arange(di_index[2])
    d_idx.shape = (di_index[0], di_index[1])
    pbar = tqdm(total=ri_index[2])
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
                st = (rx2 * ri_di - ri_m * di_m) / (rx2 * di_d2 - np.square(di_m))
                if -1 < st < 1:
                    ot = (ri_m - st * di_m) / rx2
                    h = (rr2 + st * (st * di_d2 - 2 * ri_di + 2 * ot * di_m) + ot * (rx2 * ot - 2 * ri_m)) / rx2
                    if h < h_min:
                        # h_min = h
                        for conversion in range(8):
                            di = code_book[d_idx, conversion]
                            di_m = di.sum()
                            di_d2 = np.square(np.linalg.norm(di))
                            ri_di = np.vdot(ri, di)
                            st = (rx2 * ri_di - ri_m * di_m) / (rx2 * di_d2 - np.square(di_m))
                            if -1 < st < 1:
                                ot = (ri_m - st * di_m) / rx2
                                h = (rr2 + st * (st * di_d2 - 2 * ri_di + 2 * ot * di_m) + ot * (
                                        rx2 * ot - 2 * ri_m)) / rx2
                                if h < h_min:
                                    par = [d_idx, conversion, st, ot]
                                    h_min = h
            code[r_idx[r_y_idx, r_x_idx]] = par
            pbar.update(1)
            # print(r_idx[r_y_idx, r_x_idx], par, h_min)  # ri - (par[2]*code_book[par[0], par[1]] + par[3])
    pbar.close()
    return code


# 解码过程，使用全黑图像和分形码，进行迭代输出
def decoder(code, iterate, shape, rx, dx, delta, ri_index, di_index):
    decode_ing = np.zeros((shape[0], shape[1]))
    r_idx = np.arange(ri_index[2])
    r_idx.shape = (ri_index[0], ri_index[1])
    di_t2 = np.zeros((rx, rx))
    for it in range(iterate):
        for r_y_idx in range(ri_index[0]):
            for r_x_idx in range(ri_index[1]):
                temp = code[r_idx[r_y_idx, r_x_idx]]
                y = int(temp[0] / di_index[0]) * delta
                y1 = y + dx
                x = int(temp[0] % di_index[1]) * delta
                x1 = x + dx
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
                decode_ing[r_y_idx * rx:r_y_idx * rx + rx, r_x_idx * rx:r_x_idx * rx + rx] = temp[2] * di_t2 + temp[3]
    return decode_ing


# 计算图像的PSNR，即峰值信噪比，这个参数越大，图像失真越小
def ps_nr(de_img, img):
    mse = np.square(img - de_img) / (256 * 256)
    ps_ = 10 * np.log10((255 * 255) / np.sum(mse))
    return ps_


# main函数，可调参数有R块的大小RX（一般是4或者8，满足2的次方就好），D块的大小DX（这个是RX的两倍）
# D块的位移步长Delta（这个无所谓，步长越大，码本越小，相应的解码效果就差，一般是RX的一倍或者两倍）
# picture_format为图像的格式，我们一般用.bmp, 也可以是jpg什么的
# iteration为解码的迭代次数，5-7次就收敛了
# debug_mode设置为True则不保存图像，比较省空间。False则保存图像参数，方便写实验参数
def main(image_name, r_block=4, d_block=8, delta_size=8, iteration=6, debug_mode=True, picture_format='.bmp'):
    # 参数定义
    time_star = time.time()
    rx = r_block
    dx = d_block
    delta = delta_size
    image_dir = '{dir}{name}{pic_format}'.format(name=image_name, dir='./image/', pic_format=picture_format)
    image = cv.imread(image_dir, 0)
    r_index, d_index = image_parameter(image, rx, delta)
    omega = codebook(image, rx, dx, delta, d_index)  # 构成码本
    code = encoder(image, omega, rx, r_index, d_index)
    # 打印时间
    run_time = time.time() - time_star
    print(f'运行时间：{run_time} s')
    # 后面还可以针对这个文件内的分形码进行量化，比如s采用7bit量化， 等距变换的编号3bit，o 8bit
    if debug_mode:
        decode_image = decoder(code, iteration, image.shape, rx, dx, delta, r_index, d_index)
        ps = ps_nr(image, decode_image)
        print(f'峰值信噪比：{ps}')
        cv.imshow(image_name, image)
        decode_image = decode_image.astype(np.uint8)  # 必须换成uint8才可以显示
        cv.imshow('decode', decode_image[:])
    else:
        save_npy_dir = '{dir}{name}-R{rx}-D{dx}-Iteration{iteration}{format}'.format(
            name=image_name, dir='./code/', format='.npy',
            rx=rx, dx=dx, iteration=iteration)
        np.save(save_npy_dir, code)
        code_read = np.load(save_npy_dir)
        decode_image = decoder(code_read, iteration, image.shape, rx, dx, delta, r_index, d_index)
        ps = ps_nr(image, decode_image)
        decode_image_dir = '{dir}{name}-R{rx}-D{dx}-Time{time:.2f}-PSNR{psnr:.2f}-Iteration{iteration}{pic_format}' \
            .format(name=image_name, dir='./decode/', pic_format=picture_format,
                    rx=rx, dx=dx, time=run_time, psnr=ps, iteration=iteration)
        print(f'峰值信噪比：{ps}')
        cv.imwrite(decode_image_dir, decode_image)
        cv.imshow(image_dir, image)
        cv.imshow(decode_image_dir, cv.imread(decode_image_dir, cv.IMREAD_GRAYSCALE))
    # 关闭窗口
    cv.waitKey(0)
    cv.destroyAllWindows()


# 原图像的路径./image/*.bmp，解码图像路径./decode/*.bmp，保存的文件./code/*.npy（其内的路径和文件名可修改）
if __name__ == '__main__':
    main('Lena', 4, 8, 8)  # 传入图片的名字，更多参数介绍请看main()
