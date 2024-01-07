import streamlit as st
import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt

# im1 = cv2.imread('/content/images/965564035.jpg',cv2.COLOR_BGR2GRAY)
# im2 = cv2.imread('/content/images/965225293.jpg',cv2.COLOR_BGR2GRAY)
# im_2 = cv2.imread('/content/images/892325437.jpg',cv2.COLOR_BGR2GRAY)


def similarity(img1,img2):
  # Initiate SURF detector
  # surf=cv2.xfeatures2d.SURF_create()
  sift=cv2.xfeatures2d.SIFT_create()

  # find the keypoints and descriptors with SURF
  # kp1, des1 = surf.detectAndCompute(img1,None)
  # kp2, des2 = surf.detectAndCompute(img2,None)
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)

  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)

  # Apply ratio test
  good = []
  for m,n in matches:
      if m.distance < 0.75*n.distance:
          good.append([m])
          a=len(good)
          percent=(a*100)/len(kp2)
          print("{} % similarity".format(percent))
          if percent >= 75.00:
              print('Match Found')
          if percent < 75.00:
              print('Match not Found')
        
  st.markdown("## {} % similarity".format(percent))
  if percent >= 75.00:
    st.markdown('# Match Found')
  if percent < 75.00:
    st.markdown('# Match Not Found')
  img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
  plt.imshow(img3),plt.show()
  return img3

def main():
    st.title("Image Processing with Streamlit")

    # Upload the first image
    image1 = st.file_uploader("Upload Image 1", type=['jpg', 'jpeg', 'png'])

    # Upload the second image
    image2 = st.file_uploader("Upload Image 2", type=['jpg', 'jpeg', 'png'])

    if image1 is not None and image2 is not None:
        # Load and display the input images
        image1 = np.array(Image.open(image1))
        image2 = np.array(Image.open(image2))

        st.image(image1, caption="Image 1", use_column_width=True)
        st.image(image2, caption="Image 2", use_column_width=True)

        # Convert images to grayscale using OpenCV
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Process the images
        output_image = similarity(image1, image2)
        # Display the processed output image
        st.image(output_image, caption="Image Comparison", use_column_width=True)


if __name__ == "__main__":
    main()
