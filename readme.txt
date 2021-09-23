readme.txt



#define training parameters
epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
# tức là có 10% là chọn random còn lại 90% là chọn từ bảng best action

discount_factor = 0.9 #discount factor for future rewards
# giảm giá trị của reward

learning_rate = 0.9 #the rate at which the AI agent should learn
# thành tích chỉ nên học tập 90% từ kết quả mới

#run through 1000 training episodes
for episode in range(1000):
  #get the starting location for this episode
  row_index, column_index = get_starting_location()

  #continue taking actions (i.e., moving) until we reach a terminal state
  #(i.e., until we reach the item packaging area or crash into an item storage location)
  while not is_terminal_state(row_index, column_index):
    #choose which action to take (i.e., where to move next)
    action_index = get_next_action(row_index, column_index, epsilon)

    #perform the chosen action, and transition to the next state (i.e., move to the next location)
    old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
    row_index, column_index = get_next_location(row_index, column_index, action_index)
    
    #receive the reward for moving to the new state, and calculate the temporal difference
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
    ##########################################################################################################

    #update the Q-value for the previous state and action pair
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    #***************************************************************#
    q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')
