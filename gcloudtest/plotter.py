from matplotlib import pyplot as plt
from util import gray
import io

def plot_results (img, results, images, web = None):
# results is a list of (index, score) tuples sorted by descending score, eg [(1, 100), (0, 0.7), (2, 15.6)]	

	# create plot of original image and best matches 
	fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(32, 32),sharex=False, sharey=False)
	( (ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12) ) = ax
	result_cells = [ax3, ax4, ax5, ax6, ax9, ax10, ax11, ax12]

	# result_cells = []
	# fig, ax = plt.subplots(ncols=6, nrows=2)

	gs = ax[1,2].get_gridspec()
	ax1.remove()
	ax2.remove()
	ax7.remove()
	ax8.remove()
	ax1 = fig.add_subplot(gs[0:2, 0:2])

	# fig7, f7_axs = plt.subplots(ncols=3, nrows=3)
 #    gs = f7_axs[1, 2].get_gridspec()
	# # remove the underlying axes
	# for ax in f7_axs[1:, -1]:
 #    	ax.remove()
	
	ax1.imshow(gray(img), cmap=plt.cm.gray)
	ax1.set_title('Query Image', fontsize=20, y = 1.0)

	# im_inx = 0
	# for i in ax[]:
	# 	for j in i:
	# 		if im_inx >= len(images):
	# 			break
	# 		j.imshow(gray(images[results[im_inx][0]]), cmap=plt.cm.gray)
	# 		j.set_xlim([0,32])
	# 		j.set_ylim([32,0])
	# 		j.set_title('match score: ' + '%.1f'%(results[im_inx][1]), fontsize=20, y = 1.0)
	# 		im_inx += 1

	for c_inx in range(len(result_cells)):
		if c_inx >= len(images):
			break
		result_cells[c_inx].imshow(gray(images[results[c_inx][0]]), cmap=plt.cm.gray)
		result_cells[c_inx].set_xlim([0,32])
		result_cells[c_inx].set_ylim([32,0])
		result_cells[c_inx].set_title('match score: ' + '%.1f'%(results[c_inx][1]), fontsize=20, y = 1.0)
	
	# maximize the window and display plots 
	fig.tight_layout()
	#mng = plt.get_current_fig_manager()
	#mng.window.state('zoomed')	
	if not web:
		plt.show()
	else:
		rimg = io.BytesIO()
		plt.savefig(rimg)
		rimg.seek(0)
		return rimg

def percentify(a,n):
	return [[j/n for j in i] for i in a]
    
def log_results ( index, mutation, combined_method, method1, method2, method3=[], method4=[], avg_rankings=[], top5=[], top10=[], n=1):
	avg_rankings = percentify(avg_rankings, n)
	top5 = percentify(top5, n)
	top10 = percentify(top10, n)

	logger = logging.getLogger()
	fhandler = logging.FileHandler(filename='./Logs/query_' + str(image_index) + '.log', mode='a')
	if (logger.hasHandlers()):
		logger.handlers.clear()
	logger.addHandler(fhandler)
	logger.setLevel(logging.DEBUG)
	aberrs = ["ab_identity","ab_translate", "ab_rotate","ab_affine","ab_scale","ab_flip","ab_line","ab_circle", "ab_line_circle","ab_two_line_circle"]
	for ab_index in range(10):
		logging.info("Aberration: %s", aberrs[ab_index])
	
		logging.info("Avg Rankings: %s", avg_rankings[ab_index])
		logging.info("Top5 accuracy: %s", top5[ab_index])
		logging.info("Top10 accuracy: %s", top10[ab_index])
	print("logging results")
	print(avg_rankings)
	print(top5)
	print(top10)
    
	logging.shutdown()