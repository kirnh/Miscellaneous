import cv2
import argparse
import numpy as np
import time

# Defining the function that takes an image as input and returns a zoomed image 
# according to the specified scale and pivot point
def zoom(image, pivot_point, scale):
    # Storing dimensions of the input image
    image = image.astype(np.float32)
    img_shape = image.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    # Calculating the width and height of the portion of the image retained after scaling
    retained_height, retained_width = img_height/scale, img_width/scale
    # Calculating the position of the pixels of the retained portion after selecting the pivot point
    pivot_x, pivot_y = pivot_point[0], pivot_point[1]
    x_high = int(pivot_x + (retained_width / 2))
    x_low = int(pivot_x - (retained_width / 2))
    y_high = int(pivot_y + (retained_height / 2))
    y_low = int(pivot_y - (retained_height / 2))
    # Edge cases
    if x_high > img_width:
        x_high = img_width
        x_low = img_width - retained_width
    if y_high > img_height:
        y_high = img_height
        y_low = img_height - retained_height
    if x_low < 0:
        x_low = 0
        x_high = retained_width
    if y_low < 0:
        y_low = 0
        y_high = retained_height 
    # Slicing only the portion of the image to be retained after zooming
    retained = image[y_low:y_high, x_low:x_high, :]

    # Using the pixels of the retained image portion to construct the new image of the original size   
    # using an implementation of K-times zoom
    
    new_rows = []
    # row wise filling
    # We first copy the first row
    new_rows.append(retained[0])
    for row in range(retained_height-1):
        # print(row)
        row1 = retained[row]
        row2 = retained[row+1]
        # get boolean values for where we need to use ascending order
        ascending = row1<=row2 #just need to include equal elements cases somewhere (diff=0) 
        # term that we use to create gray values between rows
        diff = abs(row1-row2)
        op = diff/float(scale)
        # We need to add (scale-1) number of rows between these two rows
        for i in range(scale-1):
            next_row = new_rows[-1].copy()
            # Ascending
            next_row[ascending] = (next_row+op)[ascending]
            # Descending
            next_row[~ascending] = (next_row-op)[~ascending]
            new_rows.append(next_row)
        # We copy the second row at last
        new_rows.append(row2)
    retained = np.array(new_rows)

    # column wise filling (need to transpose at the end of column wise filling since we're appending to a list)
    new_columns = []
    # we first copy the first column
    new_columns.append(retained[:,0])
    for col in range(retained_width-1):
        col1 = retained[:, col]
        col2 = retained[:, col+1]
        # get boolean values for where we need to use ascending order
        ascending = col1<=col2
        # term that we use to create gray values between columns
        diff = abs(col1-col2)
        op = (diff/scale).astype(np.float32)
        # We need to add (scale-1) number of columns between these two cols
        for i in range(scale-1):
            next_col = new_columns[-1].copy()
            # Ascending
            next_col[ascending] = (next_col+op)[ascending]
            # Descending
            next_col[~ascending] = (next_col-op)[~ascending]
            new_columns.append(next_col)
        # We copy the second column at last
        new_columns.append(col2)
    final_image = np.array(new_columns) 
    final_image = np.transpose(final_image, (1, 0, 2)).astype(np.uint8)
    
    return final_image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to input image", required=True)
ap.add_argument("-p", "--pivot-point", help="Pivot point coordinates x, y separated by comma (,)", required=True)
ap.add_argument("-s", "--scale", help="Scale to zoom", type=int, required=True)
args = vars(ap.parse_args())

image_path = args["image"]
x, y = map(int, args["pivot_point"].split(","))
scale = args["scale"]
image = cv2.imread(image_path)

t1 = time.time()
zoomed_image = zoom(image, [x, y], scale)
t2 = time.time()
print('Time taken to zoom: {}s'.format(t2-t1))
cv2.imwrite("zoomed_image.png", zoomed_image)
