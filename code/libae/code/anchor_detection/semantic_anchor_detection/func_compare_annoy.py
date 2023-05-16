from annoy import AnnoyIndex
import random, tqdm


def annoy_test():
    f = 64  # Length of item vector that will be indexed

    # t = AnnoyIndex(f, 'angular')
    # v_list = []
    # for i in tqdm.tqdm(range(100000)):
    #     v_list.append([random.gauss(0, 1) for z in range(f)])
        
    # for i in tqdm.tqdm(range(100000)):
    #     t.add_item(i, v_list[i])

    # t.build(10) # 10 trees
    
    # t.save('test.ann')

    # ...

    u = AnnoyIndex(f, 'angular')

    u.load('test.ann') # super fast, will just mmap the file
    query_list = []
    for i in tqdm.tqdm(range(100)):
        query_list.append([random.gauss(0, 1) for z in range(f)])
        
    # for i in tqdm.tqdm(range(100000, 100100)):
    #     u.add_item(i, v_list[i])


    

    for query_vec in query_list:
        print(u.get_nns_by_vector(query_vec, 1000, include_distances=True)) # will find the 1000 nearest neighbors
        
        
annoy_test()