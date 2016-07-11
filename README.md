# humfilt

Algorithmically determine if a picture contains people.


## Running the Example

To run the program using a few example images, use the below command. This will iterate over all images in the [`tests`](./tests) directory, and the detected images will be copied into the `images-with-people` directory. The directory will be created if it does not already exist.

```
python humfilt.py -o images-with-people tests/
```


To see the full list of supported options, envoke the program with the `--help` flag.

```
python humfilt.py --help
```

## Report

More details about the program's implementation can be found in [`report.pdf`](./report.pdf).
