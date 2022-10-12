import sys; args = sys.argv[1:]
import math

# t_funct is symbol of transfer functions: 'T1', 'T2', 'T3', or 'T4'
# input is a list of input (summation) values of the current layer
# returns a list of output values of the current layer
def transfer(t_funct, input):
   if t_funct == 'T3': return [1 / (1 + math.e** - x) for x in input]
   elif t_funct == 'T4': return [-1 + 2 / (1 + math.e** - x) for x in input]
   elif t_funct == 'T2': return [x if x > 0 else 0 for x in input]
   else: return [x for x in input]

# example: 4 inputs, 12 weights, and 3 stages(the number of next layer nodes)
# weights are listed like Example Set 1 #4 or on the NN_Lab1_description note
# returns a list of dot_product result. the len of the list == stage
# Challenge? one line solution is possible
def dot_product(input, weights, stage): # help
   return [sum([input[x] * weights [x + s * len(input)] for x in range(len(input))]) for s in range(stage)]

# file has weights information. Read file and store weights in a list or a nested list
# input_vals is a list which includes input values from terminal
# t_funct is a string, e.g. 'T1'
# evaluate the whole network (complete the whole forward feeding)
# and return a list of output(s)
def evaluate(file, input_vals, t_funct):
   stages = [len(input_vals)]
   weights = [[float(x) for x in line.split()] for line in open(file, 'r').read().splitlines()]
   # print(stages, weights)
   for w_index in range(len(weights)-1):
      stages.append(len(weights[w_index])//stages[w_index])
   # print(stages)
   for stage in range(1, len(stages)):
      #print(t_funct, dot_product(input_vals, weights[stage - 1], stages[stage]))
      input_vals = transfer(t_funct, dot_product(input_vals, weights[stage - 1], stages[stage]))
   return [input_vals[i] * weights[len(weights)-1][i] for i in range (len(input_vals))]
     
def main():
   file, inputs, t_funct, transfer_found = '', [], 'T1', False

   file = args[0]
   t_funct = args[1]
   for num in args[2:]:
      inputs.append(float(num))

   if len(file)==0: exit("Error: Weights file is not given")
   li =(evaluate(file, inputs, t_funct)) # forward feeding
   for x in li:
      print (x, end=' ')
if __name__ == '__main__': main()