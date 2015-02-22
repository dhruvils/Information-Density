#!python

import os
import xml.etree.ElementTree as ET
import operator
import string
import sys
import pickle
from collections import Counter
from math import sqrt

sys.path.append("/home1/c/cis530/hw3/liblinear/python")
from liblinearutil import *

def preprocess(file_list, corenlp_output_dir):
    os.system("(cd /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09; java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:" +\
                   "joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit," +\
                   "pos,lemma,ner,parse -filelist " +file_list +" -outputDirectory " +corenlp_output_dir +")")

def get_all_files(dir):
    absoluteFileList = []
    for dirpath, dirs, files in os.walk(dir):
        absoluteFileList += [ dirpath + '/' + filename for filename in files]
    return absoluteFileList

def extract_top_words(xml_directory):
    word_count = dict()

    #f = open("/home1/c/cis530/hw4/stopwords.txt", 'r')
    #stopword_list = f.read().split('\n')
    #f.close()

    file_list = get_all_files(xml_directory)
    for file_path in file_list:
        #print file_path
        tree = ET.parse(file_path)
        for token in tree.iter('token'):
            w = token[0].text.lower()
            #if w not in stopword_list:
            if w in word_count:
                word_count[w] += 1
            else:
                word_count[w] = 1

    sorted_list = sorted(word_count.items(), key = operator.itemgetter(1), reverse=True)
    return [tuple[0] for tuple in sorted_list[:4000]]

def map_unigrams(xml_filename, top_words):
    output_list = []
    xml_word_list = []
    tree = ET.parse(xml_filename)
    for token in tree.iter('token'):
        xml_word_list.append(token[0].text.lower())
        
    for word in top_words:
        if word in xml_word_list:
            output_list.append(1)
        else:
            output_list.append(0)

    return output_list

def cosine_similarity(x, y):
    n = len(x)
    count = 0
    num = 0
    card_x = 0
    card_y = 0

    for count in range (0, n):
        num += x[count] * y[count]
        card_x += x[count]**2
        card_y += y[count]**2

    if card_x == 0 or card_y == 0:
        cos_sim = 0
    else:
        cos_sim = float(num) / (sqrt(card_x) * sqrt(card_y))

    return cos_sim

def extract_similarity(top_words):
    word_vec_dict = dict()
    f = open('/project/cis/nlp/tools/word2vec/vectors.txt', 'r')
    for line in f:
        word_vec = line.split()
        word = word_vec.pop(0)
        word_vec_dict[word] = [float(x) for x in word_vec] 
    f.close()
    
    outer_dict = dict()
    for word_i in top_words:
        inner_dict = dict()
        for word_j in top_words:
            if word_i in word_vec_dict and word_j in word_vec_dict:
                cos_sim = cosine_similarity(word_vec_dict[word_i], word_vec_dict[word_j])
            else:
                cos_sim = 0.0

            if cos_sim != 0.0:
                inner_dict[word_j] = cos_sim
        outer_dict[word_i] = inner_dict

    return outer_dict

def max_similarity(nonzero_word_list, sim_dict):
    max_sim = 0.0
    for word in nonzero_word_list:
        if word in sim_dict:
            if sim_dict[word] > max_sim:
                max_sim = sim_dict[word]
    return max_sim

def map_expanded_unigrams(xml_filename, top_words, similarity_matrix):
    unigram_vec = map_unigrams(xml_filename, top_words)
    n = len(unigram_vec)
    nonzero_word_list = []
    index = 0
    for index in range(0, n):
        if unigram_vec[index] != 0.0:
            nonzero_word_list.append(top_words[index])

    for index in range(0, n):
        if unigram_vec[index] == 0.0:
            unigram_vec[index] = max_similarity(nonzero_word_list, similarity_matrix[top_words[index]])

    return unigram_vec

def extract_top_dependencies(xml_directory):
    dep_count = dict()
    file_list = get_all_files(xml_directory)
    for file_path in file_list:
        #print file_path
        tree = ET.parse(file_path)
        for basic_dep in tree.iter('basic-dependencies'):
            for dep in basic_dep.iter('dep'):
                dep_tuple = (dep.attrib['type'].lower(), dep[0].text.lower(), dep[1].text.lower())
                if dep_tuple in dep_count:
                    dep_count[dep_tuple] += 1
                else:
                    dep_count[dep_tuple] = 1

    sorted_list = sorted(dep_count.items(), key = operator.itemgetter(1), reverse=True)
    return [tuple[0] for tuple in sorted_list[:4000]]

def map_dependencies(xml_filename, dependency_list):
    output_list = []
    xml_dep_list = []
    tree = ET.parse(xml_filename)
    for basic_dep in tree.iter('basic-dependencies'):
        for dep in basic_dep.iter('dep'):
            xml_dep_list.append((dep.attrib['type'].lower(), dep[0].text.lower(), dep[1].text.lower()))

    for dep in dependency_list:
        if dep in xml_dep_list:
            output_list.append(1)
        else:
            output_list.append(0)

    return output_list

class Node:
    def __init__(self):
        self.parent_node = None
        self.value = ""
        self.children = []

    def append_child(self, child):
        self.children.append(child)

class Tree:
    def _init_(self):
        self.root = Node()

    def create_tree(self, parse_string):
        parse_string = string.replace(parse_string, "(", "( ")
        parse_string = string.replace(parse_string, ")", " )")
        parse_list = parse_string.split()
        #print parse_list

        root = Node()
        current = Node()
        parent = Node()
        count = 0
        for val in parse_list:
            #print val, current.value
            if val == "(":
                parent = current
            elif val == ")":
                current = current.parent_node
            else:
                if not parse_list[count + 1] == ")":
                    current = Node()
                    current.value = val
                    parent.append_child(current)
                    current.parent_node = parent
            count += 1
            #print val, current.value, [child.value for child in current.children] 
            #print val, current.value

        root = current.children[0]
        return root

def extract_rules(root, rule_list):
    if root.children:
        rule = root.value
        for child in root.children:
            rule += "_" +child.value
        rule_list.append(rule)

    for child in root.children:
        extract_rules(child, rule_list)

    return rule_list

def extract_prod_rules(xml_directory):
    file_list = get_all_files(xml_directory)
    prod_rule_list = []
    for file_path in file_list:
        tree = ET.parse(file_path)
        for parse_node in tree.iter('parse'):
            parse_string = parse_node.text

        parse_tree = Tree()
        root = parse_tree.create_tree(parse_string)
        prod_rule_list = extract_rules(root, prod_rule_list)

    prod_rule_dict = Counter(prod_rule_list)
    if '' in prod_rule_dict: prod_rule_dict.remove('')

    sorted_list = sorted(prod_rule_dict.items(), key = operator.itemgetter(1), reverse=True)
    return [tuple[0] for tuple in sorted_list[:4000]]

def map_prod_rules(xml_filename, rules_list):
    output_list = []
    xml_prod_rules = []
    tree = ET.parse(xml_filename)
    for parse_node in tree.iter('parse'):
        parse_tree = Tree()
        root = parse_tree.create_tree(parse_node.text)
        xml_prod_rules = extract_rules(root, xml_prod_rules)

    for rule in rules_list:
        if rule in xml_prod_rules:
            output_list.append(1)
        else:
            output_list.append(0)

    return output_list

def process_corpus(xml_dir, top_words, similarity_matrix, top_dependencies, prod_rules):
    #output_filename = "./train_files/"
    output_filename = "test" if "test" in xml_dir else "train"
    output_string = [""] * 5
    file_list = get_all_files(xml_dir)
    for file_path in file_list:
        unigram_vec = map_unigrams(file_path, top_words)
        extended_unigram_vec = map_expanded_unigrams(file_path, top_words, similarity_matrix)
        dependency_vec = map_dependencies(file_path, top_dependencies)
        prod_rule_vec = map_prod_rules(file_path, prod_rules)
        concat_vec = unigram_vec + dependency_vec + prod_rule_vec

        features_list = [
            unigram_vec,
            extended_unigram_vec,
            dependency_vec,
            prod_rule_vec,
            concat_vec
        ]
    
        for index in range(0, len(features_list)):
            output_string[index] += create_output_string(file_path, features_list[index]) +"\n"

    for index in range(0, len(output_string)):
        f = open(output_filename +"_" +str(index + 1) +".txt", 'w')
        f.write(output_string[index])
        f.close()

def create_output_string(file_path, feature_vec):
    file_name = file_path.split('/')[-1]
    output_string = file_name
    index = 0
    for index in xrange(0, len(feature_vec)):
        if feature_vec[index] != 0.0:
            output_string += " " +str(index + 1) +":" +str(feature_vec[index])
    return output_string

def process_liblinear_format(file_type):
    domain_list = ["Computers", "Research", "Finance", "Health"]
    for index in range(1, 6):
        file_path = file_type +"_" +str(index)
        #f = open(file_path +".txt", 'r')
        
        for domain in domain_list:
            f = open(file_path +".txt", 'r')
            output_file_path = file_path +"_" +domain +".txt"
            output_file = open(output_file_path, 'w')
            
            for line in f:
                line_list = line.split()
                #print line_list[0]
                line_list[0] = str(1) if domain in line else str(-1)
                output_file.write(" ".join(line_list) +"\n")
            f.close()

def count_labels(file_name, label):
    f = open(file_name, 'r')
    count = 0
    for line in f:
        if line.startswith(label):
            count += 1
    return count

def run_classifier(train_file, test_file):
    (y, x) = svm_read_problem(train_file)
    label_counts = Counter(y)
    v1 = label_counts[-1] / float(len(y)) 
    v2 = label_counts[1] / float(len(y))
    model = train(y, x, "-s 0 -w1 " +str(v1) +" -w-1 " +str(v2))
    #model = train(y, x, "-s 0")
    
    (test_labels, test_features) = svm_read_problem(test_file)
    (p_labels, p_acc, p_values) = predict(test_labels, test_features, model, "-b 1")

    matchedPos = 0
    numPredPos = 0
    numTruePos = 0
    matchedNeg = 0
    numPredNeg = 0
    numTrueNeg = 0
    index = 0
    while index < len(p_labels):
        if test_labels[index] == 1:
            numTruePos += 1
        else:
            numTrueNeg += 1

        if p_labels[index] == 1:
            numPredPos += 1
            if test_labels[index] == 1:
                matchedPos += 1
        else:
            numPredNeg += 1

        if p_labels[index] == -1 and test_labels[index] == -1:
            matchedNeg += 1
        index += 1

    posPrecision = (matchedPos / float(numPredPos)) if numPredPos > 0 else 0
    posRecall = (matchedPos / float(numTruePos)) if numTruePos > 0 else 0
    negPrecision = (matchedNeg / float(numPredNeg)) if numPredNeg > 0 else 0
    negRecall = (matchedNeg / float(numTrueNeg)) if numTrueNeg > 0 else 0
    posF = ((2 * posRecall * posPrecision) / float(posRecall + posPrecision)) if (posRecall + posPrecision) > 0 else 0
    negF = ((2 * negRecall * negPrecision) / float(negRecall + negPrecision)) if (negRecall + negPrecision) > 0 else 0
    acc = p_acc[0]
    p_res = (posPrecision, posRecall, negPrecision, negRecall, posF, negF, acc)

    index = 0 if model.label[0] == 1 else 1
    p_list = [p_val[index] for p_val in p_values]
    
    return (p_labels, (posPrecision, posRecall, negPrecision, negRecall, posF, negF, acc), p_list)

def write_results():
    #domains = ["Research", "Finance", "Computers", "Health"]
    domains = ["Business"]
    output_string = ""
    for domain in domains:
        for feature in range(1, 6):
            train_file = "train_" +str(feature) +"_" +domain +".txt"
            test_file = "test_" +str(feature) +"_" +domain +".txt"

            (predLabels, (posPrecision, posRecall, negPrecision, negRecall, posF, negF, acc), p_list) = run_classifier(train_file, test_file)
            output_string += str(posPrecision) +" " +str(posRecall) +" " +str(posF) +" " +str(negPrecision) +" " +str(negRecall) +" " +str(negF) +" " +str(acc) +" " +domain +":" +str(feature) +"\n"

    f = open("results.txt", "w")
    f.write(output_string)
    f.close()

def most_common(lst):
    return max(set(lst), key=lst.count)

def call_classify_doc():
    domains = ["Research", "Finance", "Computers", "Health"]
    feature_lists = []
    results_file = open("results.txt", 'r')
    i = 0
    max_list = []
    feature_accs = []
    for line in results_file:
        result_line = line.split()
        acc = float(result_line[-2])
        feature_accs.append(acc)
        if i == 4:
            max_list.append(feature_accs.index(max(feature_accs)))
            feature_accs = []
            i = 0
        else:
            i += 1

    results_file.close()
    feature = most_common(max_list) + 1

    temp = []
    for domain in domains:
        train_file = "train_" +str(feature) +"_" +domain +".txt"
        test_file = "test_" +str(feature) +"_" +domain +".txt"
        (x, y, p_list) = run_classifier(train_file, test_file)
        temp.append(p_list)
    
    feature_list = classify_documents(temp[3], temp[2], temp[0], temp[1])
    
    f = open("test_1.txt", 'r')
    i = 0
    count = 0
    for line in f:
        file_name = line.split()[0]
        if feature_list[i] in file_name:
            count += 1
        i += 1
    f.close()

    per = (count/float(500)) * 100
    f = open("results.txt", 'a')
    f.write(str(per) +"\n")
    f.close()

    return per

def classify_documents(health_prob, computers_prob, research_prob, finance_prob):
    domains = ["Health", "Research", "Computers", "Finance"]
    domain_list = []
    for (hval, cval, rval, fval) in zip(health_prob, computers_prob, research_prob, finance_prob):
        temp = [hval, rval, cval, fval]
        feature_index = temp.index(max(temp))
        domain_list.append(domains[feature_index])
    return domain_list

def main_ec():
    #corpus_dir = "/home1/c/cis530/hw3/data"
    #concat_file = open("concat_corpus.txt", 'w')
    #for file_name in get_all_files(corpus_dir):
    #    concat_file.write(open(file_name, 'r').read() +" ")

    #os.system("/project/cis/nlp/tools/word2vec/word2vec -train concat_corpus.txt -output vectors_corpus.txt -size 200 -sample le-4")

    top_words = pickle.load(open("top_words.p", "r"))
    #print "Unpickling top_words"
    #ec_sim_matrix = extract_similarity_ec(top_words)
    #print "Pickling ec_sim_matrix"
    #pickle.dump(ec_sim_matrix, open("sim_matrix_ec.p", "w"))
    #print "Pickling done for ec_sim"

    #ec_sim_matrix = pickle.load(open("sim_matrix_ec.p", 'r'))
    #xml_test_dir = "/home1/d/dhruvils/homework3/test_corenlp_output_dir"
    #xml_train_dir = "/home1/c/cis530/hw3/xml_data"
    #print "Processing corpus for test"
    #process_corpus_ec(xml_test_dir, top_words, ec_sim_matrix)
    #print "Process corpus for test done! Now starting train"
    #process_corpus_ec(xml_train_dir, top_words, ec_sim_matrix)
    #print "Process corpus for train done"

    #process_liblinear_format_ec("test")
    #process_liblinear_format_ec("train")
    #write_results_ec()
    print call_classify_doc_ec()

def call_classify_doc_ec():
    domains = ["Research", "Finance", "Computers", "Health"]
    feature_lists = []
    results_file = open("extra_credit_results.txt", 'r')
    i = 0
    max_list = []
    feature_accs = []
    for line in results_file:
        result_line = line.split()
        acc = float(result_line[-2])
        feature_accs.append(acc)
        if i == 5:
            max_list.append(feature_accs.index(max(feature_accs)))
            feature_accs = []
            i = 0
        else:
            i += 1

    results_file.close()
    feature = most_common(max_list) + 1
    #FORCING FEATURE 6 AS BEST
    feature = 6

    temp = []
    for domain in domains:
        train_file = "train_" +str(feature) +"_" +domain +".txt"
        test_file = "test_" +str(feature) +"_" +domain +".txt"
        (x, y, p_list) = run_classifier(train_file, test_file)
        temp.append(p_list)

    feature_list = classify_documents(temp[3], temp[2], temp[0], temp[1])

    f = open("test_1.txt", 'r')
    i = 0
    count = 0
    for line in f:
        file_name = line.split()[0]
        if feature_list[i] in file_name:
            count += 1
        i += 1
    f.close()

    per = (count/float(500)) * 100
    f = open("extra_credit_results.txt", 'a')
    f.write(str(per) +"\n")
    f.close()

    return per

def write_results_ec():
    domains = ["Research", "Finance", "Computers", "Health"]
    output_string = ""
    for domain in domains:
        for feature in range(1, 7):
            train_file = "train_" +str(feature) +"_" +domain +".txt"
            test_file = "test_" +str(feature) +"_" +domain +".txt"

            (predLabels, (posPrecision, posRecall, negPrecision, negRecall, posF, negF, acc), p_list) = run_classifier(train_file, test_file)
            output_string += str(posPrecision) +" " +str(posRecall) +" " +str(posF) +" " +str(negPrecision) +" " +str(negRecall) +" " +str(negF) +" " +str(acc) +" " +domain +":" +str(feature) +"\n"

    f = open("extra_credit_results.txt", "w")
    f.write(output_string)
    f.close()


def process_liblinear_format_ec(file_type):
    domain_list = ["Computers", "Research", "Finance", "Health"]
    #for index in range(1, 7):
    file_path = file_type +"_" +str(6)

    for domain in domain_list:
        f = open(file_path +".txt", 'r')
        output_file_path = file_path +"_" +domain +".txt"
        output_file = open(output_file_path, 'w')
        
        for line in f:
            line_list = line.split()
            line_list[0] = str(1) if domain in line else str(-1)
            output_file.write(" ".join(line_list) +"\n")
        f.close()
    
def process_corpus_ec(xml_dir, top_words, ec_sim_matrix):
    #output_filename = "./train_files/" 
    output_filename = "test" if "test" in xml_dir else "train"
    output_string = ""
    file_list = get_all_files(xml_dir)
    for file_path in file_list:
        extended_unigram_vec_ec = map_expanded_unigrams(file_path, top_words, ec_sim_matrix)

        output_string += create_output_string(file_path, extended_unigram_vec_ec) +"\n"

    f = open(output_filename +"_" +str(6) +".txt", 'w')
    f.write(output_string)
    f.close()

def extract_similarity_ec(top_words):
    word_vec_dict = dict()
    f = open('vectors_corpus.txt', 'r')
    for line in f:
        word_vec = line.split()
        word = word_vec.pop(0)
        word_vec_dict[word] = [float(x) for x in word_vec]
    f.close()

    outer_dict = dict()
    for word_i in top_words:
        inner_dict = dict()
        for word_j in top_words:
             if word_i in word_vec_dict and word_j in word_vec_dict:
                cos_sim = cosine_similarity(word_vec_dict[word_i], word_vec_dict[word_j])
            else:
                cos_sim = 0.0

            if cos_sim != 0.0:
                inner_dict[word_j] = cos_sim
        outer_dict[word_i] = inner_dict

    return outer_dict

def ec_feature_set_halfdata(top_words):
    #corpus_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir/"
    #concat_file = open("concat_corpus.txt", 'w')   
    #for file_name in get_all_files(corpus_dir): 
    #    concat_file.write(open(file_name, 'r').read() +" ")
    #os.system("/project/cis/nlp/tools/word2vec/word2vec -train concat_corpus.txt -output vectors_corpus.txt -size 200 -sample le-4")
    #top_words = pickle.load(open("half_top_words.p", "r"))
    #top_words = pickle.load(open("half_top_words_nsw.p", "r"))
    ec_sim_matrix = extract_similarity_ec(top_words)
    return ec_sim_matrix

def create_temp_test_files():
    src_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir/"
    dest_dir = "/home1/d/dhruvils/clproject/temp_test_corenlp_dir"
    train_labels_path = "/home1/c/cis530/project/train_labels.txt"

    f = open("temp_train_labels.txt", 'w')
    temp_test_file = open("temp_test_labels.txt", "w")

    line_count = 1
    for line in open(train_labels_path):
        if line_count % 2 == 0:
            temp_test_file.write(line)
            file_name = src_dir +line.split()[0] +".xml"
            os.system("mv " +file_name +" " +dest_dir)
        else:
            f.write(line)
        line_count += 1
    f.close()

def process_liblinear_train():
    #train_labels_path = "/home1/d/dhruvils/clproject/train_file_list.txt"
    train_labels_path = "/home1/c/cis530/project/train_labels.txt"
    labels_dict = dict()
    for line in open(train_labels_path):
        labels_info = line.split()
        labels_dict[labels_info[0]] = labels_info[1]
    
    for index in xrange(1,13):
        f = open("train_" +str(index) +"_Business.txt", "w")
        train_file = open("train_" +str(index) +".txt")
        for line in train_file:
            output = line.split()
            output[0] = labels_dict[output[0][:-4]]
            #output[0] = labels_dict[output[0]]
            f.write(" ".join(output) +"\n")
        f.close()

def process_liblinear_train_halfdata():
    train_labels_path = "/home1/d/dhruvils/clproject/temp_train_labels.txt"                                 
    #train_labels_path = "/home1/c/cis530/project/train_labels.txt"
    labels_dict = dict()
    for line in open(train_labels_path):
        labels_info = line.split()
        labels_dict[labels_info[0]] = labels_info[1]

    for index in xrange(1,13):
        f = open("half_train_" +str(index) +"_Business.txt", "w")
        train_file = open("half_train_" +str(index) +".txt")
        for line in train_file:
            output = line.split()
            output[0] = labels_dict[output[0][:-4]]
            #output[0] = labels_dict[output[0]]                                                                             
            f.write(" ".join(output) +"\n")
        f.close()

def process_liblinear_test_halfdata():
    train_labels_path = "/home1/d/dhruvils/clproject/temp_test_labels.txt"                        
    labels_dict = dict()                                                                                                           
    for line in open(train_labels_path):                                                                                           
        labels_info = line.split()                                                                                                 
        labels_dict[labels_info[0]] = labels_info[1]                                                                               

    for index in xrange(1,13):
        f = open("half_test_" +str(index) +"_Business.txt", "w")
        test_file = open("half_test_" +str(index) +".txt")
        for line in test_file:
            output = line.split()
            output[0] = labels_dict[output[0][:-4]]                                                                                
            #output[0] = str(1)
            f.write(" ".join(output) +"\n")
        f.close()

def process_liblinear_test():
    #train_labels_path = "/home1/d/dhruvils/clproject/test_file_list.txt"
    #labels_dict = dict()
    #for line in open(train_labels_path):
    #    labels_info = line.split()
    #    labels_dict[labels_info[0]] = labels_info[1]

    for index in xrange(1,13):
        f = open("test_" +str(index) +"_Business.txt", "w")
        test_file = open("test_" +str(index) +".txt")
        for line in test_file:
            output = line.split()
            #output[0] = labels_dict[output[0][:-4]]
            output[0] = str(1)
            f.write(" ".join(output) +"\n")
        f.close()

def run_classifier_proj(train_file, test_file):
    (y, x) = svm_read_problem(train_file)
    label_counts = Counter(y)
    v1 = label_counts[-1] / float(len(y))
    v2 = label_counts[1] / float(len(y))
    model = train(y, x, "-s 0 -w1 " +str(v1) +" -w-1 " +str(v2))
    #model = train(y, x, "-s 0") 

    (test_labels, test_features) = svm_read_problem(test_file)
    (p_labels, p_acc, p_values) = predict(test_labels, test_features, model, "-b 1")

    return p_labels

def calc_acc(p_labels):
    temp_test_label_file = "/home1/d/dhruvils/clproject/temp_test_labels.txt"
    total_count = 0
    count = 0
    index = 0
    for line in open(temp_test_label_file):
        act_label = int(line.split()[1])
        if p_labels[index] == act_label:
            count += 1
        total_count += 1
        index += 1

    return (float(count) / total_count) * 100

def write_test_labels(p_labels):
    f = open("test.txt", "w")
    index = 0
    for line in open("/home1/d/dhruvils/clproject/test_file_list.txt"):
        file_name = line.split("/")[-1][:-1]
        f.write(file_name +" " +str(int(p_labels[index])) +"\n")
        index += 1
    f.close()

def process_corpus_proj(xml_dir, top_words, similarity_matrix, top_dependencies, prod_rules, ec_feature_set, topic_words):
    #output_filename = "./train_files/"                                                                                            
    #output_filename = "half_test" if "test" in xml_dir else "half_train"
    output_filename = "test" if "test" in xml_dir else "train"
    output_string = [""] * 12
    file_list = get_all_files(xml_dir)
    for file_path in file_list:
        unigram_vec = map_unigrams(file_path, top_words)
        extended_unigram_vec = map_expanded_unigrams(file_path, top_words, similarity_matrix)
        dependency_vec = map_dependencies(file_path, top_dependencies)
        prod_rule_vec = map_prod_rules(file_path, prod_rules)
        concat_vec = unigram_vec + dependency_vec + prod_rule_vec
        feature_6 = concat_vec + extended_unigram_vec
        extended_unigram_vec_ec = [] #map_expanded_unigrams(file_path, top_words, ec_feature_set)
        feature_8 = concat_vec + extended_unigram_vec_ec
        feature_9 = unigram_vec + prod_rule_vec
        topic_words_vec = map_topic_words(file_path, topic_words)
        feature_11 = topic_words_vec + concat_vec
        feature_12 = extended_unigram_vec_ec + topic_words_vec

        features_list = [
            unigram_vec,
            extended_unigram_vec,
            dependency_vec,
            prod_rule_vec,
            concat_vec,
            feature_6,
            extended_unigram_vec_ec,
            feature_8,
            feature_9,
            topic_words_vec,
            feature_11,
            feature_12
        ]

        for index in range(0, len(features_list)):
            output_string[index] += create_output_string(file_path, features_list[index]) +"\n"
    
    for index in range(0, len(output_string)):
        f = open(output_filename +"_" +str(index + 1) +".txt", 'w')
        f.write(output_string[index])
        f.close()

def create_ts_files():
    input_dir = "/home1/d/dhruvils/clproject/temp_train_only1"
    output_dir = "/home1/d/dhruvils/clproject/topic_words/"
    tool_dir = "/home1/c/cis530/hw4/TopicWords-v2/"
    my_dir = "/home1/d/dhruvils/clproject/"

    config_base_string = "==== Do not change these values ====\n"+\
                         "stopFilePath = stoplist-smart-sys.txt\n"+\
                         "performStemming = N\n"+\
                         "backgroundCorpusFreqCounts = bgCounts-Giga.txt\n"+\
                         "topicWordCutoff = 0.1\n"+\
                         "\n"+\
                         "==== Directory to compute topic words on ====\n\n"
    config_outputdir_string = "==== Output File ====\n\n"

    #for i in xrange(0, 4):
    #    for j in xrange(0, 10):
    f = open("config_1", "w")
    f.write(config_base_string)
    f.write("inputDir = " +input_dir +"\n\n")
    f.write(config_outputdir_string)
    f.write("outputFile = " +output_dir +"topic_words.ts")
    f.close()

    os.system("(cd " +tool_dir +"; java -Xmx1000m TopicSignatures " +my_dir +"config_1" +")")

def load_topic_words(topic_file, n):
    f = open(topic_file, "r")
    topic_words_dict = dict()

    for line in f:
        line_data = line.split()
        if float(line_data[1]) >= 10.0:
            topic_words_dict[line_data[0]] = float(line_data[1])

    sorted_topic_words = sorted(topic_words_dict.items(), key = operator.itemgetter(1), reverse=True)
    top_n = [tup[0].lower() for tup in sorted_topic_words[:n]]
    remaining_list = [tup[0].lower() for tup in sorted_topic_words[n+1:]]
    return (top_n, remaining_list)

def map_topic_words(xml_filename, topic_words):
    output_list = []
    xml_word_list = []
    tree = ET.parse(xml_filename)
    for token in tree.iter('token'):
        xml_word_list.append(token[0].text.lower())

    for word in topic_words:
        if word in xml_word_list:
            output_list.append(1)
        else:
            output_list.append(0)

    return output_list

def create_only1_trainset():
    #file_list = get_all_files("/home1/d/dhruvils/clproject/train_corenlp_output_dir")
    source_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir"
    dest_dir = "/home1/d/dhruvils/clproject/temp_train_only1"
    for line in open("temp_train_labels.txt", "r"):
        label_list = line.split()
        if label_list[1] == "1":
            source_file = source_dir + "/" +label_list[0] +".xml"
            os.system("cp " +source_file +" " +dest_dir)

def create_only1_trainset_full():
    #file_list = get_all_files("/home1/d/dhruvils/clproject/train_corenlp_output_dir")
    source_dir = "/home1/d/dhruvils/clproject/full_train_corenlp_output_dir"
    dest_dir = "/home1/d/dhruvils/clproject/full_train_only1"
    for line in open("/home1/c/cis530/project/train_labels.txt", "r"):
        label_list = line.split()
        if label_list[1] == "1":
            source_file = source_dir + "/" +label_list[0] +".xml"
            os.system("cp " +source_file +" " +dest_dir)

def create_ts_files_full():
    input_dir = "/home1/d/dhruvils/clproject/full_train_only1"
    output_dir = "/home1/d/dhruvils/clproject/topic_words/"
    tool_dir = "/home1/c/cis530/hw4/TopicWords-v2/"
    my_dir = "/home1/d/dhruvils/clproject/"

    config_base_string = "==== Do not change these values ====\n"+\
                         "stopFilePath = stoplist-smart-sys.txt\n"+\
                         "performStemming = N\n"+\
                         "backgroundCorpusFreqCounts = bgCounts-Giga.txt\n"+\
                         "topicWordCutoff = 0.1\n"+\
                         "\n"+\
                         "==== Directory to compute topic words on ====\n\n"
    config_outputdir_string = "==== Output File ====\n\n"

    #for i in xrange(0, 4):                                                                                                         
    #    for j in xrange(0, 10):                                                                                                    
    f = open("config_1", "w")
    f.write(config_base_string)
    f.write("inputDir = " +input_dir +"\n\n")
    f.write(config_outputdir_string)
    f.write("outputFile = " +output_dir +"topic_words_full.ts")
    f.close()
    os.system("(cd " +tool_dir +"; java -Xmx1000m TopicSignatures " +my_dir +"config_1" +")")


def main():
    #1:
    #train_file_list = get_all_files("/home1/c/cis530/project/train_data")
    #f = open('train_file_list.txt', 'w')
    #for file_path in train_file_list:
    #    f.write(file_path +"\n")
    #f.close()
    #train_file = "/home1/d/dhruvils/clproject/train_file_list.txt"
    #train_output_dir = "/home1/d/dhruvils/clproject/full_train_corenlp_output_dir"
    #preprocess(train_file, train_output_dir)

    #test_file_list = get_all_files("/home1/c/cis530/project/test_data")
    #f = open('test_file_list.txt', 'w')
    #for file_path in test_file_list:
    #    f.write(file_path +"\n")
    #f.close()
    #test_file = "/home1/d/dhruvils/clproject/test_file_list.txt"
    #test_output_dir = "/home1/d/dhruvils/clproject/test_corenlp_output_dir"
    #preprocess(test_file, test_output_dir)

    #2.1:
    #xml_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir"
    #extract_top_words(xml_dir)

    #2.2:
    #xml_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir"
    #top_words = extract_top_words(xml_dir)
    #xml_file = "/home1/d/dhruvils/clproject/test_corenlp_output_dir/2007_05_19_1848295.txt.xml"
    #map_unigrams(xml_file, top_words)

    #3.1:
    #xml_dir = "/home1/c/cis530/hw3/xml_data"
    #top_words = extract_top_words(xml_dir)
    #similarity_matrix = extract_similartiy(top_words)

    #3.2:
    #xml_dir = "/home1/c/cis530/hw3/xml_data"
    #xml_dir = "/home1/d/dhruvils/homework3/testing_data"
    #top_words = extract_top_words(xml_dir)
    #similarity_matrix = extract_similarity(top_words)
    #xml_filename = "/home1/c/cis530/hw3/xml_data/Finance_2005_06_24_1682533.txt.xml"
    #print map_expanded_unigrams(xml_filename, top_words, similarity_matrix)

    #4.1:
    #xml_dir = "/home1/d/dhruvils/homework3/testing_data"
    #print extract_top_dependencies(xml_dir)

    #4.2:
    #xml_dir = "/home1/d/dhruvils/homework3/testing_data"
    #dependency_list = extract_top_dependencies(xml_dir) 
    #xml_filename = "/home1/c/cis530/hw3/xml_data/Finance_2005_06_24_1682533.txt.xml"
    #print map_dependencies(xml_filename, dependency_list)

    #5.1:
    #xml_dir = "/home1/d/dhruvils/homework3/testing_data"
    #print extract_prod_rules(xml_dir)
    
    #5.2:
    #xml_dir = "/home1/d/dhruvils/homework3/testing_data"
    #prod_rule_list = extract_prod_rules(xml_dir)
    #xml_filename = "/home1/c/cis530/hw3/xml_data/Finance_2005_06_24_1682533.txt.xml"
    #print map_prod_rules(xml_filename, prod_rule_list)



    ## PICKLING
    #xml_dir = "/home1/d/dhruvils/clproject/full_train_corenlp_output_dir"
    #top_words = extract_top_words(xml_dir)
    #pickle.dump(top_words, open("top_words.p", "w"))
    #similarity_matrix = extract_similarity(top_words)
    #pickle.dump(similarity_matrix, open("sim.p", "w"))
    #top_dependencies = extract_top_dependencies(xml_dir)
    #pickle.dump(top_dependencies,  open("dep_list.p", "w"))
    #prod_rules = extract_prod_rules(xml_dir)
    #pickle.dump(prod_rules, open("rules.p", "w"))

    # UNPICKLING
    #top_words = pickle.load(open("top_words.p", "r"))
    #similarity_matrix = pickle.load(open("sim.p", "r"))
    #top_dependencies = pickle.load(open("dep_list.p", "r"))
    #prod_rules = pickle.load(open("rules.p","r"))

    #6.2:
    #xml_test_dir = "/home1/d/dhruvils/clproject/test_corenlp_output_dir"
    #xml_train_dir = "/home1/d/dhruvils/clproject/full_train_corenlp_output_dir"
    #process_corpus(xml_test_dir, top_words, similarity_matrix, top_dependencies, prod_rules)
    #process_corpus(xml_train_dir, top_words, similarity_matrix, top_dependencies, prod_rules)

    #6.3:
    #process_liblinear_format("train")
    #process_liblinear_format("test")

    #6.4:
    #write_results()

    #7:
    #print call_classify_doc()

    #8: EXTRA CREDIT:
    #main_ec()

    #CL PROJECT:
    #create_only1_trainset()
    #create_ts_files()
    
    ## PICKLING - HALF DATA                                                                                            
    #xml_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir"
    #top_words = extract_top_words(xml_dir)                                                                                        
    #pickle.dump(top_words, open("half_top_words.p", "w")) 
    #similarity_matrix = extract_similarity(top_words)                                                                             
    #pickle.dump(similarity_matrix, open("half_sim.p", "w"))                                                
    #top_dependencies = extract_top_dependencies(xml_dir)                                                                          
    #pickle.dump(top_dependencies,  open("half_dep_list.p", "w"))                                             
    #prod_rules = extract_prod_rules(xml_dir)                                                                                      
    #pickle.dump(prod_rules, open("half_rules.p", "w"))                                                         
    #(topic_words, remaining_list) = load_topic_words("/home1/d/dhruvils/clproject/topic_words/topic_words.ts", 2000)
    #print topic_words
    #pickle.dump(topic_words, open("half_topic_words.p", "w"))

    # UNPICKLING - HALF DATA                                                                             
    #top_words = pickle.load(open("half_top_words.p", "r"))                                                              
    #similarity_matrix = pickle.load(open("half_sim.p", "r"))                                                     
    #top_dependencies = pickle.load(open("half_dep_list.p", "r"))                                                     
    #prod_rules = pickle.load(open("half_rules.p","r"))
    #topic_words = pickle.load(open("half_topic_words.p", "r"))

    ec_feature_set = ec_feature_set_halfdata()
    #ec_feature_set = pickle.load(open("half_sim_matrix_ec.p", "r"))

    # PICKLING - FULL DATA
    #xml_dir = "/home1/d/dhruvils/clproject/full_train_corenlp_output_dir"
    top_words = extract_top_words(xml_dir)
    #pickle.dump(top_words, open("top_words.p", "w"))
    similarity_matrix = extract_similarity(top_words)
    #pickle.dump(similarity_matrix, open("sim.p", "w"))
    top_dependencies = extract_top_dependencies(xml_dir)
    #pickle.dump(top_dependencies,  open("dep_list.p", "w"))
    prod_rules = extract_prod_rules(xml_dir)
    #pickle.dump(prod_rules, open("rules.p", "w"))
    #create_only1_trainset_full()
    #create_ts_files_full()
    (topic_words, remaining_list) = load_topic_words("/home1/d/dhruvils/clproject/topic_words/topic_words_full.ts", 2000)
    #pickle.dump(topic_words, open("topic_words.p", "w"))

    # UNPICKLING - FULL DATA  
    #top_words = pickle.load(open("top_words.p", "r"))
    #similarity_matrix = pickle.load(open("sim.p", "r"))
    #top_dependencies = pickle.load(open("dep_list.p", "r"))
    #prod_rules = pickle.load(open("rules.p","r"))
    #topic_words = pickle.load(open("topic_words.p", "r"))

    #create_temp_test_files()
    xml_test_dir = "/home1/d/dhruvils/clproject/test_corenlp_output_dir"
    xml_train_dir = "/home1/d/dhruvils/clproject/full_train_corenlp_output_dir"
    process_corpus_proj(xml_test_dir, top_words, similarity_matrix, top_dependencies, prod_rules, ec_feature_set, topic_words)
    process_corpus_proj(xml_train_dir, top_words, similarity_matrix, top_dependencies, prod_rules, ec_feature_set, topic_words)
    
    #xml_test_dir = "/home1/d/dhruvils/clproject/temp_test_corenlp_dir"
    #xml_train_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir"
    #process_corpus_proj(xml_test_dir, top_words, similarity_matrix, top_dependencies, prod_rules, ec_feature_set)
    #process_corpus_proj(xml_train_dir, top_words, similarity_matrix, top_dependencies, prod_rules, ec_feature_set)
    #process_liblinear_train()
    #process_liblinear_test()

    #xml_test_dir = "/home1/d/dhruvils/clproject/temp_test_corenlp_dir"
    #xml_train_dir = "/home1/d/dhruvils/clproject/train_corenlp_output_dir"
    #process_corpus_proj(xml_test_dir, top_words, similarity_matrix, top_dependencies, prod_rules, ec_feature_set, topic_words)
    #process_corpus_proj(xml_train_dir, top_words, similarity_matrix, top_dependencies, prod_rules, ec_feature_set, topic_words)
    process_liblinear_train_halfdata()
    process_liblinear_test_halfdata()

    #for i in xrange(1, 13):
    #    train_file = "half_train_" +str(i) +"_Business.txt"
    #    test_file = "half_test_" +str(i) +"_Business.txt"
    #    print calc_acc(run_classifier_proj(train_file, test_file))

    train_file = "train_5_Business.txt"
    test_file = "test_5_Business.txt"
    write_test_labels(run_classifier_proj(train_file, test_file))

if __name__ == "__main__":
    main()
