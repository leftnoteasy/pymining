Objective
---------
This is a platform writing in Python that can use variant data-mining algorithms to get results from a source (like matrix, text documents).
Algorithms can using xml configuration to make them run one-by-one. E.g. at first, we may run PCA(principle components analysis) for feature selection, then we may run random forest for classification. 
Now, algorithms are mainly design for tasks can be done in a single computer, good scalability of the architecture allows you in a very short period of time to complete the algorithm you want, and use it in your project(believe me, it's faster, better, and easier than Weka). The another important feature is this platfrom can support text classification or clustering operation very good.


Get start
---------
*Just write code like this, you will get amazing result (a naive-bayes training and testing),*

    # load configuratuon from xml file
    config = Configuration.FromFile("conf/test.xml")
    GlobalInfo.Init(config, "__global__")

    # init module that can create matrix from text file
    txt2mat = Text2Matrix(config, "__matrix__")

    # create matrix for training (with tag) from a text file "train.txt"
    [trainx, trainy] = txt2mat.CreateTrainMatrix("data/train.txt")

    # create a chisquare filter from config file
    chiFilter = ChiSquareFilter(config, "__filter__")

    # get filter model from training matrix
    chiFilter.TrainFilter(trainx, trainy)

    # filter training matrix
    [trainx, trainy] = chiFilter.MatrixFilter(trainx, trainy)

    # train naive bayes model
    nbModel = NaiveBayes(config, "naive_bayes")
    nbModel.Train(trainx, trainy)

    # create matrix for test
    [testx, testy] = txt2mat.CreatePredictMatrix("data/test.txt")

    # using chisquare filter do filtering
    [testx, testy] = chiFilter.MatrixFilter(testx, testy)

    # test matrix and get result (save in resultY) and precision
    [resultY, precision] = nbModel.Test(testx, testy)

    print precision


And you need define some paramters in a configuration file (It will be loaded by Configuration.FromFile(...xml)).

    <config>
        <__segmenter__>
            <main_dict>dict/dict.main</main_dict>
        </__segmenter__>

        <__matrix__>
        </__matrix__>

        <__global__>
            <term_to_id>mining/term_to_id</term_to_id>
            <id_to_term>mining/id_to_term</id_to_term>
            <id_to_doc_count>mining/id_to_doc_count</id_to_doc_count>
            <class_to_doc_count>mining/class_to_doc_count</class_to_doc_count>
            <id_to_idf>mining/id_to_idf</id_to_idf>
            <newid_to_id>mining/newid_to_id</newid_to_id>
        </__global__>

        <__filter__>
            <rate>0.2</rate>
            <method>max</method>
            <log_path>mining/filter.log</log_path>
            <model_path>mining/filter.model</model_path>
        </__filter__>

        <naive_bayes>
            <model_path>mining/naive_bayes.model</model_path>
            <log_path>mining/naive_bayes.log</log_path>
        </naive_bayes>

        <twc_naive_bayes>
            <model_path>mining/naive_bayes.model</model_path>
            <log_path>mining/naive_bayes.log</log_path>
        </twc_naive_bayes>
        
        <smo_csvc>
          <model_path>mining/smo_csvc.model</model_path>
          <log_path>mining/smo_csvc.log</log_path>
          <c>100</c>
          <eps>0.001</eps>
          <tolerance>0.000000000001</tolerance>
          <times>50</times>
          <kernel>
            <name>RBF</name>
            <parameters>10</parameters>
          </kernel>
          <cachesize>300</cachesize>
        </smo_csvc>
    </config>

Features
========
*Clustering algorithm*
+ KMeans

*Classification algorithm*
+ Random forest
+ Naive Bayes
+ TWC Naive Bayes
+ SVM

*Feature selector*
+ Chisquare
+ PCA

*Mathematic*
+ Basic operations (like bagging, transpose, etc.)

*Data source support*
+ Matrix (with csv format)
+ Raw text (now only support Chinese, English to be added)

Benchmark
=========
To be add
