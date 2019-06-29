import zernike
import util
import cv2
import plotter
import combined
import contour
import orb
import sift

zernike_database = util.load_obj('zernike_database_icon_50')
orb_database = util.load_obj('orb_database_icon_50')
sift_database = util.load_obj('sift_database_icon_50')
images = util.load_images("shoes")
images = images[:50]
img = images[0]

# orb_database = orb.generate_database(images)
# util.save_obj(orb_database, 'orb_database_shoes_50')
# sift_database = sift.generate_database(images)
# util.save_obj(sift_database, 'sift_database_shoes_50')
# zernike_database = zernike.generate_database(images)
# util.save_obj(zernike_database, 'zernike_database_shoes_50')


# s = sift.create_query(img)
# s = sift.test_query(s, sift_database)
# plotter.plot_results(img, s, images)

print("starting")
o = orb.create_query(img)
o = orb.test_query(o, orb_database)
print("orb")
s = sift.create_query(img)
s = sift.test_query(s, sift_database)
print("sift")
z = zernike.create_query(img)
z = zernike.test_query(z, zernike_database)
print("zernike")
res = combined.test_combined([z,o,s], [1,5,5])
plotter.plot_results(img, res, images)



# img = images[0]
# z = zernike.create_query(images[0])
# z = zernike.test_query(z, zernike_database)
# res = combined.test_combined([z])
# plotter.plot_results(img, res, images)

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

# contour_database = util.load_obj('contour_database_icon_10')
# c = contour.create_query(images[0])
# print(c)

#contour_database = contour.generate_database(images)
#util.save_obj(contour_database, 'contour_database_icon_10')

# print(contour.test_query(c, contour_database))
# # contour_database = contour.generate_database(images)
# # util.save_obj(contour_database, 'contour_database_icon_10')

print(len(images))