def test_combined( weights, method1, method2, method3 = [], method4 = [] ):

	# check all 3 methods and average the scores 
	matched_combined = []
	N = 2
	if len(method3) > 0:
		N = N + 1
	if len(method4) > 0:
		N = N + 1
	
	for w in  range(len(images)):
			
		if N == 3:
			matched_combined.append( (w, method1[w][1] , method2[w][1], method3[w][1], round(weights[0]* method1[w][1] + weights[1]*method2[w][1] + weights[2]*method3[w][1],1)) )
		elif N == 2:
			matched_combined.append( (w, method1[w][1] , method2[w][1], 0, round(weights[0]*method1[w][1] + weights[1]*method2[w][1],1)) )
		elif N == 4:            
			matched_combined.append( (w, method1[w][1] , method2[w][1], method3[w][1], method4[w][1], round(weights[0]* method1[w][1] + weights[1]*method2[w][1] + weights[2]*method3[w][1] + weights[3]*method4[w][1],1)) )
	matched_combined = sorted(matched_combined, key = lambda tup: tup[N+1], reverse = True )	
	return matched_combined