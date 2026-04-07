import naer2 as np


# Option 1
@np.edge
class stdp_synapse_timestamp:
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, time, source, destination):
        self.weight = self.LTP(self.weight, source.potential, source.time, time)
        self.weight = self.LTD(self.weight, source.time, time)
        # extend lif_ed here with additional attributes for timestamp buffer

    def backward(self, time, source, destination):
        self.weight = self.LTP(self.weight, source.potential, source.time, time)
        self.weight = self.LTD(self.weight, source.time, time)
        # extend lif_ed here with additional attributes for timestamp buffer


# load all neurons of group
# for each neuron
#   update
# load messages with sources
# for each message
#   load connections and targets of message
#   for each connection
#     update synapse (transmit message forwards)
#     receive
# whenever a send is made: neuron send + all incoming connections back-transmit

# OBSERVATION: since the synapse can add attributes to source and destination for buffers, there is no need to specify which type they are!
# This STDP Synapse would work with ANY type, and only checking at transpilation that the type there matches is needed.
# The interaction only comes from the neuron code needing to know the weight!
# The synapse code does not need to know anything about the neuron code in this case!

# enforce back-transmission cannot change self attributes!
# if it does, a backwards search is needed, slowing computation, so its forbidden for now
