from mpi4py import MPI
import re
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
group = comm.Get_group()
workerCount = size - 1

### Master process ###
if rank == 0:

    # Command line arguments
    args = {'input_file': None, 'merge_method': None, 'test_file': None}
    # Read command line arguments
    option=''
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            if option != '':
                raise Exception('Error: ' + arg + 'is not a valid value for --' + option)
            else:
                option = arg.lstrip('--')
                if args[option] != None:
                    raise Exception('Error: Duplicate argument: --' + option)
        else:
            if option == '':
                raise Exception('Error: ' + arg + ' has no key.')
            elif option not in args:
                raise Exception('Error: ' + option + 'is not a valid key.')
            else:
                args[option] = arg
                option = ''

    # If one of the mandatory arguments are not given:
    if None in args:
        raise Exception('Error: Please make sure that you provide --input_file, --merge_method and --test_file!')

    # Open input file
    text_file = open(args['input_file'])
    bulk_data = text_file.read()    #read whole file to a string
    text_file.close()               # close file
    
    sentences = re.split(r'\n+', bulk_data) # Get the list of sentences

    parititonSize = len(sentences) // workerCount
    rem = len(sentences) % workerCount # remainder

    partitions = list() # Contains distributed data

    # Distribute data evenly to workers
    for i in range(workerCount):
        partitions.append(sentences[(i* parititonSize):((i+1) * parititonSize)])
    # Distribute the remainders
    for i in range(rem):
        partitions[i].append(sentences[parititonSize * workerCount + i])
    
    # Sending data to workers
    for i, partition in enumerate(partitions):
        comm.send(partition, dest = i+1, tag = 0)
        comm.send(args['merge_method'], dest = i+1, tag = 1)
    
    ### Requirement 4:

    # Total unigram and bigram counts
    totalUnigramCount = dict()
    totalBigramCount = dict()

    # Case 1
    if args['merge_method'] == 'MASTER':

        for worker in range(1, workerCount + 1):
            # Receive unigram and bigram counts from a worker
            unigramCount = comm.recv(source = worker, tag = 2 * worker)
            bigramCount  = comm.recv(source = worker, tag = 2 * worker + 1)

            # Merge unigrams counted by the worker
            for unigram, count in unigramCount.items():
                totalUnigramCount[unigram] = totalUnigramCount.get(unigram, 0) + count
            
            # Merge bigrams counted by the worker
            for bigram, count in  bigramCount.items():
                totalBigramCount[bigram] = totalUnigramCount.get(bigram, 0) + count

        ## TODO: Erase these later    
        #print ("TOTAL UNIGRAMS\n",totalUnigramCount)
        #print ("TOTAL BIGRAMS\n", totalBigramCount)

    # Case 2
    elif args['merge_method'] == 'WORKERS':
        # TODO: get unigram and bigram data from last worker (Send the unigram data with tag 2, bigram data with tag 3)
        totalUnigramCount = comm.recv(source = workerCount, tag = 2 * workerCount)
        totalBigramCount  = comm.recv(source = workerCount, tag = 2 * workerCount + 1)
    
    else:
        raise Exception('Error: Unknown merge method.')

    # TODO: Requiement 4 (Calculating conditional probabilities)

    text_file = open(args['test_file']) 
        
    for line in text_file:
        a = line.split(" ")[0]
        b = line.split(" ")[1]
        denominator = totalUnigramCount[a]
        nominator = totalBigramCount[line]
        cond_prob = nominator / denominator
        print ("P(" + b + "|" + a + ") =" + cond_prob)

    text_file.close()   

### Worker process ###
else:
    # Receive data from master
    sentences = comm.recv(source = 0, tag = 0)
    merge_method = comm.recv(source = 0, tag = 1)

    # Print the rank and the number of sentences (Requirement 2)
    print("Rank:", rank, "Number of sentences:", len(sentences))

    unigramCount = dict()
    bigramCount = dict()

    # For each sentence in sentence list
    for sentence in sentences:

        # List of unigrams
        unigramsOfSentence = re.findall(r'<s>|</s>|\w+', sentence)

        # Add unigrams to dictionary
        for unigram in unigramsOfSentence:
            unigramCount[unigram] = unigramCount.get(unigram, 0) + 1
        
        # List of bigrams
        bigramsofSentence = list()
        for i in range(len(unigramsOfSentence) - 1):
            bigram = unigramsOfSentence[i] + " " + unigramsOfSentence[i+1]
            bigramCount[bigram] = bigramCount.get(bigram, 0) + 1
    
    # Merge data

    # Case 1
    if merge_method == 'MASTER':
        # Send data to master
        comm.send(obj = unigramCount, dest = 0, tag = 2 * rank)
        comm.send(obj = bigramCount,  dest = 0, tag = 2 * rank + 1)
        
    # Case 2 
    elif merge_method == 'WORKERS':

        # TODO: get data from previous worker (don't do this if rank == 1)
        if(rank > 1):
            previousWorker = rank - 1
            prevUnigramCount = comm.recv(source = previousWorker, tag = 2 * previousWorker)
            prevBigramCount  = comm.recv(source = previousWorker, tag = 2 * previousWorker + 1)

        # TODO: merge this worker's data and the data obtained from the previous worker
        for unigram, count in unigramCount.items():
            unigramCount[unigram] = prevUnigramCount.get(unigram, 0) + count
        
        # Merge bigrams counted by the worker
        for bigram, count in  bigramCount.items():
            unigramCount[bigram] = prevBigramCount.get(bigram, 0) + count

        # TODO: send merged data to next worker (or to master if rank == workerCount)
        destination = 0 if rank == workerCount else (rank + 1)
        comm.send(obj = unigramCount, dest = destination, tag = 2 * rank)
        comm.send(obj = bigramCount,  dest = destination, tag = 2 * rank + 1)

    else:
        raise Exception('Error: Unknown merge method.')
