######## RUN GAMES ############

# RUN A SET OF GAMES FOR FACL AGENT
#Step 1: load the fuzzy info to recreate the membership functions and rules
# Note: Turn into a function to do this dynamically
fuzzy_info = np.loadtxt("fuzzy_info.txt")
max_state_vals = fuzzy_info[0:2]
min_state_vals = fuzzy_info[2:4]
number_of_membership_functions = [0,0]
number_of_membership_functions[0] = int(fuzzy_info[4])
number_of_membership_functions[1] = int(fuzzy_info[5])
print('max ',max_state_vals) # Print out for troubleshooting purposes
print('min ',min_state_vals)
print('num mf',number_of_membership_functions)
#Step 2: create the controller using the fuzzy info
run_controller = TestController(state, max_state_vals, min_state_vals, number_of_membership_functions)
#Step 3: create agent
eli = Agent(run_controller)
#Step 4: call the load function for actor critic weights
eli.controller.load()
#Step 5: run the game 100 times
for i in range(100):
    eli.run_one_game()
print("num of sucessful games : ",eli.success)
eli.print_path()