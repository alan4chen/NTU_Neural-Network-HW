
import random
import math

num_data_points = 100
num_epoches = 5000
learning_rate_att = 0.01
learning_rate_rep = 0.1
num_layers = 5
max_dimension = 5

similar_pairs = []
distinct_pairs = []

def cal_dis(p, dim, layer):
	distance = 0
	for i in range(dim):
		distance += (layer[p[0] * max_dimension + i] - layer[p[1] * max_dimension + i]) ** 2
	return distance


if __name__ == "__main__":
	dimension_per_layer = [2, 5, 5, 5, 5]

	cur_layer = [0] * num_data_points * max_dimension
	pre_layer = [0] * num_data_points * max_dimension

	# Read Data
	with open('hw2pt.dat', 'r') as f:
		for idx, line in enumerate(f.readlines()):
			lst = line.split("\t")
			cur_layer[idx * max_dimension] = float(lst[0])
			cur_layer[idx * max_dimension + 1] = float(lst[1])

	with open('hw2class.dat') as f:
		for idx, line in enumerate(f.readlines()):
			for idy, val in enumerate(line.split('\t')):
				if int(val) == 1:
					similar_pairs.append((idx, idy))
				else:
					distinct_pairs.append((idx, idy))

	weights = [0] * max_dimension * (max_dimension + 1)
	weight_scales_left = [0] * max_dimension
	weight_scales_right = [0] * max_dimension

	# SOM Algorithm
	for	layer_index in range(0, num_layers-1):

		### Initialize_layer(layer_index)
		print("Initialize layer ", layer_index)
		input_dimension = dimension_per_layer[layer_index]
		output_dimension = dimension_per_layer[layer_index + 1]

		# Swap
		tmp = pre_layer
		pre_layer = cur_layer
		cur_layer = tmp

		for i in range(output_dimension):
			for j in range(input_dimension):
				weights[i * (max_dimension + 1) + j] = random.uniform(-1, 1)


		for epoch_index in range(num_epoches):
			selected_similar_index = [-1, 1000000]
			selected_distinct_index = [-1, 1000000]

			### find_shortest_longest_distance(layer_index, epoch_index)
			rand_select_index = random.randint(0, num_data_points-1)

			## update outputs of neurons
			for data_index in range(num_data_points):
				for curr_neuron in range(output_dimension):
					cur_layer[data_index * max_dimension + curr_neuron] = weights[(max_dimension + 1) * curr_neuron + input_dimension]
					for prev_neuron in range(input_dimension):
						cur_layer[data_index * max_dimension + curr_neuron] += \
							pre_layer[data_index * max_dimension + prev_neuron] * weights[(max_dimension + 1) * curr_neuron + prev_neuron]

					cur_layer[data_index * max_dimension + curr_neuron] = \
						math.tanh(cur_layer[data_index * max_dimension + curr_neuron] * 0.5)


			## find the pairs of longest/shortest distance
			for i in range(len(similar_pairs)):
				p = similar_pairs[i]
				distance = cal_dis(p, output_dimension, cur_layer)
				if i == 0 or selected_similar_index[1] < distance:
					selected_similar_index[0] = i
					selected_similar_index[1] = distance

			for i in range(len(distinct_pairs)):
				p = distinct_pairs[i]
				distance = cal_dis(p, output_dimension, cur_layer)
				if i == 0 or selected_distinct_index[1] > distance:
					selected_distinct_index[0] = i
					selected_distinct_index[1] = distance


			print("layer:", layer_index, "~", layer_index+1, " epoch:", epoch_index,
				" longest similar:", selected_similar_index[1],
				" shortest distinct:", selected_distinct_index[1])

			### Adjust_weights(layer_index, epoch_index);
			# Similar class pair
			selected_pair = similar_pairs[selected_similar_index[0]]

			for i in range(output_dimension):
				left_curr_val = cur_layer[selected_pair[0] * max_dimension + i]
				right_curr_val = cur_layer[selected_pair[1] * max_dimension + i]
				weight_scales_left[i] = (left_curr_val - right_curr_val) * (1.0 - left_curr_val ** 2)
				weight_scales_right[i] = (left_curr_val - right_curr_val) * (1.0 - right_curr_val ** 2)

			# Update Bias Weights
			for curr_neuron in range(output_dimension):
				weights[curr_neuron * (max_dimension + 1) + input_dimension] -= \
					learning_rate_att * (-weight_scales_left[curr_neuron] + weight_scales_right[curr_neuron])

			# Update Weights
			for curr_neuron in range(output_dimension):
				for prev_neuron in range(input_dimension):
					left_prev_val = pre_layer[selected_pair[0] * max_dimension + prev_neuron]
					right_prev_val = pre_layer[selected_pair[1] * max_dimension + prev_neuron]
					weights[curr_neuron * (max_dimension + 1) + prev_neuron] -= \
						learning_rate_att * (weight_scales_left[curr_neuron] * left_prev_val - weight_scales_right[curr_neuron] * right_prev_val)

			# Distinct class pair
			selected_pair = distinct_pairs[selected_similar_index[0]]
			for i in range(output_dimension):
				left_curr_val = cur_layer[selected_pair[0] * max_dimension + i]
				right_curr_val = cur_layer[selected_pair[1] * max_dimension + i]
				weight_scales_left[i] = (left_curr_val - right_curr_val) * (1.0 - left_curr_val ** 2)
				weight_scales_right[i] = (left_curr_val - right_curr_val) * (1.0 - right_curr_val ** 2)

			for curr_neuron in range(output_dimension):
				weights[curr_neuron * (max_dimension + 1) + input_dimension] -= \
					learning_rate_rep * (weight_scales_left[curr_neuron] - weight_scales_right[curr_neuron])

			for curr_neuron in range(output_dimension):
				for prev_neuron in range(input_dimension):
					left_prev_val = pre_layer[selected_pair[0] * max_dimension + prev_neuron]
					right_prev_val = pre_layer[selected_pair[1] * max_dimension + prev_neuron]
					weights[curr_neuron * (max_dimension + 1) + prev_neuron] -= \
						learning_rate_rep * (-weight_scales_left[curr_neuron] * left_prev_val + weight_scales_right[curr_neuron] * right_prev_val)
