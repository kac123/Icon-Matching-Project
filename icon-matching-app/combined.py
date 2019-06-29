def test_combined(results_list, weights=[]):
#reults list is a list of result lists [r1, r2, r3...]
#where each result list is a list of (index, score) tuples
#assume that each result list is sorted by index ascending
	if not results_list: #just in case we somehow get an empty resuts list
		return []

	matched_combined = []
	for i_inx in range(len(results_list[0])):
		score = 0
		total_weighting = 0
		for r_inx in range(len(results_list)):
			weight = weights[r_inx] if r_inx < len(weights) else 1
			score += weight * results_list[r_inx][i_inx][1]
			total_weighting += weight
		score /= total_weighting
		matched_combined.append((i_inx, score))
			
	matched_combined = sorted(matched_combined, key = lambda tup: tup[1], reverse = True )	
	return matched_combined