# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:40:12 2019

@author: Danish
"""


# Create a class that will do the batching for the algorithm
class BatchMaker():
    
    """ This class contains three functions. This class will be an iterator. An iterator in 
        Python is a class with methods next() and iter() and is used with loops. Generally we
        use loops that iterate for a number of observations like {for i in range(y):}. We can 
        instead use an iterator, and write: {for input in data}, where data is the instance of
        class. Then we proceed as usual and include the operations we want the loop to perform.
        With batching we want to take batches one after the other and calculate the loss for each 
        batch. This class will split our input and target data into batches whatever the batch size
        we will set. We will take input batches and target batches from the class until the data in
        the object (the object which will be associated to this class) is exhausted at each iteration.

        __init__(): This class performs the batching. The init() method loads the data form the .npz
                    files that we have created in the preprocessig part. 
        __next__(): The next method() loads the next batch from the data that we have loaded.
        __iter__(): The iter() method tells that the class is iterable. Knowing this once a variable 
                    is an instance of a class when included in the loop it will load the first batch 
                    then iterate over the data set taking one batch after the other until the dataset 
                    is exhausted."""
    
    # Dataset is a mandatory arugment, while the batch_size is optional
    # If you don't input batch_size, it will automatically take the value: None
    def __init__(self, inputs, batch_size = None):
    
        # The dataset that loads is one of "train", "validation", "test".
        # e.g. if I call this class with x(dataset,5), it will transform data to a batch size of 5.
        
        # Two variables that take the values of the inputs and the targets. Inputs are floats, targets are integers
        self.inputs = inputs
        
        # Counts the batch number, given the size you feed it later
        # If the batch size is None, we are either validating or testing, so we want to take the data in a single batch
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size
    
    # A method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()
            
        # You slice the dataset in batches and then the "next" function loads them one after the other
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        self.curr_batch += 1
        
        # The function will return the inputs batch and the one-hot encoded targets
        return inputs_batch
    
        
    # A method needed for iterating over the batches, as we will put them in a loop
    # This tells Python that the class we're defining is iterable, i.e. that we can use it like:
    # for input, output in data: 
        # do things
    # An iterator in Python is a class with a method __next__ that defines exactly how to iterate through its objects
    def __iter__(self):
        return self