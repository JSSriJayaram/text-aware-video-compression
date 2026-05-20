import os
import time
import copy

import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from torch import from_numpy, load, no_grad
from torch.autograd import Variable
from torch.backends.cudnn import benchmark as cudnn_benchmark
from torch.cuda import empty_cache as empty_cuda_cache
from torch.nn import DataParallel

def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def get_prediction(
    image,
    craft_net,
    refine_net=None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    cuda: bool = False,
    long_size: int = 1280,
    poly: bool = True,
):
    """
    """
    t0 = time.time()

    # read/convert image
    image = read_image(image)

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, long_size, interpolation=cv2.INTER_LINEAR
    )
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        import torch
        if torch.backends.mps.is_available():
            x = x.to('mps')
        elif torch.cuda.is_available():
            x = x.cuda()
        # else: stays on CPU
    preprocessing_time = time.time() - t0
    t0 = time.time()

    # forward pass
    with no_grad():
        y, feature = craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    refinenet_time = time.time() - t0
    t0 = time.time()

    # Post-processing
    boxes, polys = getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # get image size
    img_height = image.shape[0]
    img_width = image.shape[1]

    # calculate box coords as ratios to image size
    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])
    boxes_as_ratio = np.array(boxes_as_ratio)

    # calculate poly coords as ratios to image size
    polys_as_ratio = []
    for poly in polys:
        polys_as_ratio.append(poly / [img_width, img_height])
    polys_as_ratio = np.array(polys_as_ratio)

    text_score_heatmap = cvt2HeatmapImg(score_text)
    link_score_heatmap = cvt2HeatmapImg(score_link)

    postprocess_time = time.time() - t0

    times = {
        "resize_time": resize_time,
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    return {
        "boxes": boxes,
        "boxes_as_ratios": boxes_as_ratio,
        "polys": polys,
        "polys_as_ratios": polys_as_ratio,
        "heatmaps": {
            "text_score_heatmap": text_score_heatmap,
            "link_score_heatmap": link_score_heatmap,
        },
        "times": times,
    }

def read_image(image):
    if type(image) == str:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img = image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]

    return img


def normalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img

def resize_aspect_ratio(img, long_size, interpolation):
    height, width, channel = img.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def export_detected_region(image, poly, file_path, rectify=False):
    """
    Arguments:
        image: full image
        points: bbox or poly points
        file_path: path to be exported
        rectify: rectify detected polygon by affine transform
    """
    #if rectify:
        # rectify poly region
    #    result_rgb = rectify_poly(image, poly)
    #else:
    result_rgb = crop_poly(image, poly)

    # export corpped region
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, result_bgr)


def export_detected_regions(
    image,
    regions,
    file_name: str = "image",
    output_dir: str = "output/",
    rectify: bool = False,
):
    """
    Arguments:
        image: path to the image to be processed or numpy array or PIL image
        regions: list of bboxes or polys
        file_name (str): export image file name
        output_dir: folder to be exported
        rectify: rectify detected polygon by affine transform
    """

    # read/convert image
    image = read_image(image)

    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # create crops dir
    crops_dir = os.path.join(output_dir, file_name + "_crops")
    create_dir(crops_dir)

    # init exported file paths
    exported_file_paths = []

    # export regions
    for ind, region in enumerate(regions):
        # get export path
        file_path = os.path.join(crops_dir, "crop_" + str(ind) + ".png")
        # export region
        export_detected_region(image, poly=region, file_path=file_path, rectify=rectify)
        # note exported file path
        exported_file_paths.append(file_path)

    return exported_file_paths


def export_extra_results(
    image,
    regions,
    heatmaps,
    file_name: str = "image",
    output_dir="output/",
    verticals=None,
    texts=None,
):
    """save text detection result one by one
    Args:
        image: path to the image to be processed or numpy array or PIL image
        file_name (str): export image file name
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4]
            for QUAD output
    Return:
        None
    """
    # read/convert image
    image = read_image(image)

    # result directory
    res_file = os.path.join(output_dir, file_name + "_text_detection.txt")
    res_img_file = os.path.join(output_dir, file_name + "_text_detection.png")
    text_heatmap_file = os.path.join(output_dir, file_name + "_text_score_heatmap.png")
    link_heatmap_file = os.path.join(output_dir, file_name + "_link_score_heatmap.png")

    # create output dir
    create_dir(output_dir)

    # export heatmaps
    cv2.imwrite(text_heatmap_file, heatmaps["text_score_heatmap"])
    cv2.imwrite(link_heatmap_file, heatmaps["link_score_heatmap"])

    with open(res_file, "w") as f:
        for i, region in enumerate(regions):
            region = np.array(region).astype(np.int32).reshape((-1))
            strResult = ",".join([str(r) for r in region]) + "\r\n"
            f.write(strResult)

            region = region.reshape(-1, 2)
            cv2.polylines(
                image,
                [region.reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    (region[0][0] + 1, region[0][1] + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                )
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    tuple(region[0]),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness=1,
                )

    # Save result image
    cv2.imwrite(res_img_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def export_results(image,regions,heatmaps,out,out1,out2):
    ### Params

    # image: The input image.
    # regions: A list of text detections.
    # heatmaps: A dictionary containing the text score heatmap
    # and the link score heatmap.
    # out: The output file for the original image with the text detections overlaid.
    # out1: The output file for the text score heatmap.
    # out2: The output file for the link score heatmap.

    image = read_image(image)

    for i, region in enumerate(regions):

        # Converts the text detection region to a NumPy array
        # Casts the NumPy array to the int32 data type.
        # Reshapes the NumPy array to a 1D array.
        region = np.array(region).astype(np.int32).reshape((-1))

        # Reshapes the 1D array to a 2D array
        # where each row represents a point in the polygon around the text detection
        region = region.reshape(-1, 2)

        # Draw a polygon around the text detection using the cv2.polylines() function.
        cv2.polylines(
            image,
            [region.reshape((-1, 1, 2))],
            True,
            color=(0, 0, 255),
            thickness=2,
        )

    # Writes the text score heatmap to the output file out1
    out1.write(heatmaps["text_score_heatmap"])
    # Write link/affinity score heatmap to output file out2
    out2.write(cv2.cvtColor(heatmaps["link_score_heatmap"],cv2.COLOR_RGB2BGR))
    # Write image to output file out
    out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def crop_poly(image, poly):
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    # create mask with shape of image
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)

    # method 1 smooth region
    cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))

    # crop around poly
    res = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(poly)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    return cropped


# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_craftnet_model(
        cuda: bool = False,
        weight_path: Optional[Union[str, Path]] = None
):
    import torch
    from pathlib import Path
    
    # get craft net path
    if weight_path is None:
        home_path = str(Path.home())
        weight_path = Path(
            "pretrained",
            "craft_mlt_25k.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    # load craft net
    from models.craftnet import CraftNet

    craft_net = CraftNet()  # initialize

    # Determine device and load
    if cuda:
        if torch.backends.mps.is_available():
            # Apple Silicon GPU
            device = torch.device("mps")
            craft_net.load_state_dict(
                copyStateDict(torch.load(weight_path, map_location=device))
            )
            craft_net = craft_net.to(device)
        elif torch.cuda.is_available():
            # NVIDIA GPU
            craft_net.load_state_dict(copyStateDict(torch.load(weight_path)))
            craft_net = craft_net.cuda()
            craft_net = DataParallel(craft_net)
            cudnn_benchmark = False
        else:
            # Fallback to CPU
            craft_net.load_state_dict(
                copyStateDict(torch.load(weight_path, map_location="cpu"))
            )
    else:
        # CPU
        craft_net.load_state_dict(
            copyStateDict(torch.load(weight_path, map_location="cpu"))
        )
    
    craft_net.eval()
    return craft_net


def load_refinenet_model(
        cuda: bool = False,
        weight_path: Optional[Union[str, Path]] = None
):
    import torch
    from pathlib import Path
    
    # get refine net path
    if weight_path is None:
        home_path = Path.home()
        weight_path = Path(
            "pretrained",
            "craft_refiner_CTW1500.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    # load refine net
    from models.refinenet import RefineNet

    refine_net = RefineNet()  # initialize

    # Determine device and load
    if cuda:
        if torch.backends.mps.is_available():
            # Apple Silicon GPU
            device = torch.device("mps")
            refine_net.load_state_dict(
                copyStateDict(torch.load(weight_path, map_location=device))
            )
            refine_net = refine_net.to(device)
        elif torch.cuda.is_available():
            # NVIDIA GPU
            refine_net.load_state_dict(copyStateDict(torch.load(weight_path)))
            refine_net = refine_net.cuda()
            refine_net = DataParallel(refine_net)
            cudnn_benchmark = False
        else:
            # Fallback to CPU
            refine_net.load_state_dict(
                copyStateDict(torch.load(weight_path, map_location="cpu"))
            )
    else:
        # CPU
        refine_net.load_state_dict(
            copyStateDict(torch.load(weight_path, map_location="cpu"))
        )
    
    refine_net.eval()
    return refine_net


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        # remove link area
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = (x - niter, x + w + niter + 1, y - niter, y + h + niter + 1)
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_temp = np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
        np_contours = np_temp.transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # boundary check due to minAreaRect may have out of range values 
        # 
        for p in box:
            if p[0] < 0:
                p[0] = 0
            if p[1] < 0:
                p[1] = 0
            if p[0] >= img_w:
                p[0] = img_w
            if p[1] >= img_h:
                p[1] = img_h

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = (
            int(np.linalg.norm(box[0] - box[1]) + 1),
            int(np.linalg.norm(box[1] - box[2]) + 1),
        )
        if w < 10 or h < 10:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2:
                continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len:
                max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0:
                    break
                cp_section[seg_num] = [
                    cp_section[seg_num][0] / num_sec,
                    cp_section[seg_num][1] / num_sec,
                ]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [
                cp_section[seg_num][0] + x,
                cp_section[seg_num][1] + cy,
            ]
            num_sec += 1

            if seg_num % 2 == 0:
                continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh
        # is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = -math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (
            pp[2][1] - pp[1][1]
        ) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (
            pp[-3][1] - pp[-2][1]
        ) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                    np.sum(np.logical_and(word_label, line_img)) == 0
                    or r + 2 * step_r >= max_r
                ):
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                    np.sum(np.logical_and(word_label, line_img)) == 0
                    or r + 2 * step_r >= max_r
                ):
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None)
            continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text
    )

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    adjusted_polys = []
    if len(polys) > 0:
        for poly in polys:
            if poly is not None:
                adjusted_poly = np.array(poly)
                # Perform any necessary adjustments to the coordinates here
                adjusted_poly *= (ratio_w * ratio_net, ratio_h * ratio_net)
                adjusted_polys.append(adjusted_poly)
    if adjusted_polys:
        adjusted_polys = np.array(adjusted_polys)
    return adjusted_polys