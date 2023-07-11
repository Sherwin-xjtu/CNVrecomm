# CNVrecom

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/Sherwin-xjtu/CNVrecom/edit/master/README.md)

CNVrecom: CNVrecommender: An online service to recommend suitable tools for Copy Number Variation Detection!

CNVrecom is a copy number variation (CNV) detection software recommendation algorithm developed based on meta-learning, meta-paths, and deep graph neural network technologies! 

## Table of Contents

- [Features](#features)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Features



## Background



## Install
Uncompress the installation zip:

    $ cd /my/install/dir/
    $ unzip /path/to/CNVrecom.zip
    

## Usage


```sh
$ python ~/CNVrecom/CNVrecom-main.py -c /home/cnv_user/software/CNVrecom/data/NA12878.recal.test.new.sorted.bam -n /home/cnv_user/software/CNVrecom/data/NA12878n.recal.test.new.sorted.bam -t /home/cnv_user/software/CNVrecom/data/chip.bed -p 0 -l 101 -d 100 -o /home/cnv_user/CNVrecomReport.tsv

#parameters

-c case.bam. Input the case sample's bam file, which needs to be sorted and indexed.

-n control.bam. Input the case sample's bam file, which needs to be sorted and indexed.

-t chip.bed. Input Sample Sequencing Capture Interval (bed file).

-p purity. e.g.'-p 0.6' indicates that the input tumor purity is 0.6. Note: The tumor purity of the input sample is required. It is recommended to use the ABSOLUTE tool to estimate the tumor purity of the sample.

-l read length. e.g. '-l 150' indicates that the input read length is 150bp. Note: This can be obtained using tools like samtools or pysam.

-d sampleDepth. e.g. '-d 100' indicates that the input sample's average depth is 100X. Note: This can be obtained using tools like samtools or pysam.
```


## Maintainers

[@Sherwin](https://github.com/Sherwin-xjtu).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/Sherwin-xjtu/CNVrecom/issues/new) or submit PRs.

## License

[MIT](LICENSE) © Sherwin




