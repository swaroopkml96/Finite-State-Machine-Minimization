# Finite-State-Machine-Minimization
### Python program to visualize, run and minimize finite state machines

**Finite state machines lie at the heart of computation.**
A real computer can be modelled as a finite state machine. This has led many to speculate that everything that can be simulated in a computer, including the weather, the human brain and the whole universe are finite state machines.

But the coolest thing about finite state machines is the fact that **they can be minimized.**
Let's say you build a number of machines M1, M2, ... to recognize numbers divisible by 4 (each one solving the problem in a different way). One of the machine takes a binary number as input and does the following:
  1. Convert the binary number to decimal by weighted multiplication with successive powers of 2
  2. Repeatedly subtract 4 from it until you get a number < 4
  3. If the number is exactly 0, conclude that it is divisible by 4

If you model the machines as finite state machines F1, F2, F3,..., each of F1, F2, F3,... will be different.
But, F1, F2, F3,... when minimized, all become the same, minimal machine, which does the following:
  1. Conclude that it is divisible by 4 if the last two bits are 0

In other words, **every problem that can be modelled as a finite state machine will have a unique minimal solution, and you can find it!**

Enough with the rambling. Let's come to the program at hand.

You need to describe your finite state machines in the following format in a seperate file
```
  # comments. Ignored.
  states: name_of_state1 name_of_state2 ...
  init: name_of_initial_state
  final: name_of_final_state1 name_of_final_state_2 ...
  name_of_from_state name_of_to_state 0/1
  name_of_from_state name_of_to_state 0/1
  name_of_from_state name_of_to_state 0/1
  .
  .
  .
```
  
create your finite state machine from this file (creates and returns a FSM)

``my_fsm = FSM.fromFile(path_to_file)``

Run it on a string

``res = my_fsm.run(string)``

minimize it (return the minimized equivalent FSM)

``minimized_fsm = fsm.minimize()``

display them (requires matplotlib)

``my_fsm.display()``

``minimized_fsm.display()``

write it to file:

``minimized_fsm.toFile(path_to_file)``

Example finite state machine:

```
states: start saw0 saw00 
init: start
final: saw00 
start saw0 0
start start 1
saw0 saw00 0
saw0 start 1
saw00 saw00 0
saw00 start 1
```

![Visualizing the above finite state machine](/ex_graph2.png "Requires matplotlib")

The filled line is for 0, and the dotted line is for 1 and dges from node 1 to node 2 have to same color as node 1.

#### Good luck! Any contributions are welcome.
  
