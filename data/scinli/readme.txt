This directory contains the benchmark train, test and development splits of the SciNLI dataset proposed in the ACL 2022 paper titled "SciNLI: A Corpus for Natural Language Inference on Scientific Text."

If you use this dataset, please cite our paper:

    Mobashir Sadat and Cornelia Caragea. 2022.
      SciNLI: A Corpus for Natural Language Inference on Scientific Text. 
      Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

    @inproceedings{sadat-caragea-2022-SciNLI,
        title = "SciNLI: A Corpus for Natural Language Inference on Scientific Text",
        author = "Sadat, Mobashir  and
          Caragea, Cornelia",
        booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        year = "2022",
        address = "Dublin, Ireland",
        publisher = "Association for Computational Linguistics",
    }

contact: msadat3@uic.edu, sadat.mobashir@gmail.com


###Project page###
Following page contains the code for loading the dataset and running experiments for baseline models: https://github.com/msadat3/SciNLI


###Files###
=> train.csv, test.csv and dev.csv contain the training, testing and development data, respectively. Each file has three columns: 
    * 'id': a unique id for each sample
	* 'sentence1': the premise of each sample
	* 'sentence2': the hypothesis of each sample
	* 'label': corresponding label representing the semantic relation between the premise and hypothesis. 


=> train.jsonl, test.jsonl and dev.jsonl contain the same data as the CSV files but they are formatted in a json format similar to SNLI and MNLI. Precisely, each line is a json dictionary where the keys are 'id', 'sentence1', 'sentence2' and 'label' with the id, premise, hypothesis and the label as the values.


###Data Source###
All sentence pairs are extracted from papers on NLP and computational linguistics available in the ACL Anthology, published between 2000 and 2020. The data source can be cited as follows:

    @inproceedings{bird2008acl,
      title={{The ACL Anthology Reference Corpus: A Reference Dataset for Bibliographic Research in Computational Linguistics}},
      author={Bird, Steven and Dale, Robert and Dorr, Bonnie J. and Gibson, Bryan and Joseph, Mark T. and Kan, Min-Yen and Lee, Dongwon and Powley, Brett and Radev, Dragomir R. and Tan, Yee Fan},
      booktitle={Proc. of the 6th International Conference on Language Resources and Evaluation Conference (LREC’08)},
      pages={1755--1759},
      year={2008}}

    @inproceedings{radev-etal-2009-acl,
        title = "The {ACL} {A}nthology Network",
        author = "Radev, Dragomir R.  and
          Muthukrishnan, Pradeep  and
          Qazvinian, Vahed",
        booktitle = "Proceedings of the 2009 Workshop on Text and Citation Analysis for Scholarly Digital Libraries ({NLPIR}4{DL})",
        month = aug,
        year = "2009",
        address = "Suntec City, Singapore",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/W09-3607",
        pages = "54--61",
    }


The sentence pairs in the training set are automatically annotated with distant supervision. The test and dev sets were human annotated. 

###Labels###
Each sentence pair is assigned to one of the following classes: entailment, contradiction, reasoning, neutral.

###Dataset Size###
Train: 101,412 - automatically annotated
Test: 4,000 - human annotated
Dev: 2,000 - human annotated
Total: 107,412

