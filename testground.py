import zernike
import util
import cv2
# import contour

images = util.load_images("icon_sample")
img = images[0]

# img = util.gray(img)


# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('imzero.png',img)




# images = util.load_images("LLD-icon.hdf5")
# images = images[:10]

# img = util.gray(images[0])




# zernike_database = zernike.generate_database(images)
# util.save_obj(zernike_database, 'zernike_database_icon_10')

# zernike_database = util.load_obj('zernike_database_icon_10')

# z = zernike.create_query(images[0])
# print(zernike.test_query(z, zernike_database))


# c = contour.create_query(images[0])
# print(c)

# contour_database = contour.generate_database(images)
# util.save_obj(contour_database, 'contour_database_icon_10')

# print(contour.test_query(c, images, contour_database))
# # contour_database = contour.generate_database(images)
# # util.save_obj(contour_database, 'contour_database_icon_10')

print(len(images))