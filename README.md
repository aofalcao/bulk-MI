# bulk-MI
## Source Implementations for bulk computing of Mutual Information for large binary datasets

This has the source code for the [ArXiv paper](https://arxiv.org/abs/2411.19702) where the principles and test results are enumerated

## Mode of use

Please take note of the sample code in `tester.py` where several examples are performed. 

This is the gist: The main idea is the submission of a Binary data object for which one want to compute the mutual information between columns. As an example:
```
D=np.array([[0,1,0],
            [1,0,1],
            [1,1,0],
            [1,0,0],
            [0,0,0],
            [0,1,1],
            [1,0,0],
            [0,0,0]])
```

The main entry point should be `bulk_MI(D)`, and the result is an array with the computed mutual information. D can be:
* A numpy array
* A scipy sparse matrix 
* A torch tensor

The code can then be executed this way:

```
import bulkMI

mi =  bulkMI.bulk_MI(D)
print(mi)
```

The result for the above data should be

```
[[1.    0.049 0.   ]
 [0.049 1.    0.016]
 [0.    0.016 1.   ]]
```

`bulkMI` also provides two other entry points that are mostly for performance comparison:

* `pairwise_MI(D)` - this will use the slow pairwise scikit-learn implementation. It's slow but verifiably correct. It may be useful to check results (**Note**: scikit-learn needs to be installed, or it will return an empty array)
* `bulk_MI_base(D)` - The base implementation without gram compute optimization. Orders of magnitude faster than scikit-learn, but still slower than the optimized code of `bulk_MI`

Both these functiuons return an array with the computed Mutual Information for all rows as above
