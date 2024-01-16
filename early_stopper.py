class EarlyStopper:

    def __init__(self, patience, min_delta):
        """
        Initialise the early stopper object.

        Arguments:
            - patience {int} -- The number of epochs the validation error must be above the threshold in order to trigger early stopping.
            - min_delta {int} -- The value used to determine the height of the threshold.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.min_loss = float('inf')
    
    def reset(self):
        """
        Reset the early stopper object for a new run.
        """
        self.count = 0
        self.min_loss = float('inf')
    
    def stop(self, loss):
        """
        Determine whether the loss is high enough to trigger early stopping.

        Arguments:
            - loss {float} -- the current validation loss.
        
        Returns a boolean value (whether or not early stopping is triggered).
        """
        if (loss < self.min_loss): # Check to see if we update minimum loss so far
            self.min_loss = loss
            self.count = 0
        elif loss >= self.min_loss + self.min_delta: # If we exceed the threshold
            self.count += 1
            if self.count == self.patience: # If we have exceeded the threshold for enough cycles
                return True
        return False