def plot_results (img, combined_method):	
	
	# create plot of original image and best matches 
	fig, ( (ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12) ) = plt.subplots(nrows=2, ncols=6, figsize=(32, 32),sharex=False, sharey=False)
	
	ax1.imshow(img, cmap=plt.cm.gray)
	ax1.set_title('Query Image', fontsize=20, y = 1.0)
	
	ax2.imshow(gray(images[combined_method[0][0]]), cmap=plt.cm.gray)
	ax2.set_xlim([0,32])
	ax2.set_ylim([32,0])
	ax2.set_title('% Match: ' + str(combined_method[0][3]), fontsize=20, y = 1.0)
	
	ax3.imshow(gray(images[combined_method[1][0]]), cmap=plt.cm.gray)
	ax3.set_xlim([0,32])
	ax3.set_ylim([32,0])
	ax3.set_title('% Match: ' + str(combined_method[1][3]), fontsize=20, y = 1.0)
	
	ax4.imshow(gray(images[combined_method[2][0]]), cmap=plt.cm.gray)
	ax4.set_xlim([0,32])
	ax4.set_ylim([32,0])
	ax4.set_title('% Match: ' + str(combined_method[2][3]), fontsize=20, y = 1.0)
	
	ax5.imshow(gray(images[combined_method[3][0]]), cmap=plt.cm.gray)
	ax5.set_xlim([0,32])
	ax5.set_ylim([32,0])
	ax5.set_title('% Match: ' + str(combined_method[3][3]), fontsize=20, y = 1.0)
	
	ax6.imshow(gray(images[combined_method[4][0]]), cmap=plt.cm.gray)
	ax6.set_xlim([0,32])
	ax6.set_ylim([32,0])
	ax6.set_title('% Match: ' + str(combined_method[4][3]), fontsize=20, y = 1.0)
	
	ax7.imshow(gray(images[combined_method[5][0]]), cmap=plt.cm.gray)
	ax7.set_xlim([0,32])
	ax7.set_ylim([32,0])
	ax7.set_title('% Match: ' + str(combined_method[5][3]), fontsize=20, y = 1.0)
	
	ax8.imshow(gray(images[combined_method[6][0]]), cmap=plt.cm.gray)
	ax8.set_xlim([0,32])
	ax8.set_ylim([32,0])
	ax8.set_title('% Match: ' + str(combined_method[6][3]), fontsize=20, y = 1.0)
	
	ax9.imshow(gray(images[combined_method[7][0]]), cmap=plt.cm.gray)
	ax9.set_xlim([0,32])
	ax9.set_ylim([32,0])
	ax9.set_title('% Match: ' + str(combined_method[7][3]), fontsize=20, y = 1.0)
	
	ax10.imshow(gray(images[combined_method[8][0]]), cmap=plt.cm.gray)
	ax10.set_xlim([0,32])
	ax10.set_ylim([32,0])
	ax10.set_title('% Match: ' + str(combined_method[8][3]), fontsize=20, y = 1.0)
	
	ax11.imshow(gray(images[combined_method[9][0]]), cmap=plt.cm.gray)
	ax11.set_xlim([0,32])
	ax11.set_ylim([32,0])
	ax11.set_title('% Match: ' + str(combined_method[9][3]), fontsize=20, y = 1.0)
	
	ax12.imshow(gray(images[combined_method[10][0]]), cmap=plt.cm.gray)
	ax12.set_xlim([0,32])
	ax12.set_ylim([32,0])
	ax12.set_title('% Match: ' + str(combined_method[10][3]), fontsize=20, y = 1.0)
	
	# maximize the window and display plots 
	fig.tight_layout()
	#mng = plt.get_current_fig_manager()
	#mng.window.state('zoomed')	
	plt.show()

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