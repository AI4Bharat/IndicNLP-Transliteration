"""
A script to calculate accuracy, mAP, F1-score, etc. of transliterations
given the predictions and ground truth as JSONs.

USAGE:
    Example 1: Predictions file format - JSON
    <<script.py>> --gt-json EnHi_dev.json --pred-json Google/EnHi_dev.json [--save-output-csv scores.csv]

    Example 2: Predictions file format - txt
    <<script.py>> --gt-json EnHi_dev.json --in EnHi_dev.en --out Moses_out.hi [--save-output-csv scores.csv]

Source: https://github.com/snukky/news-translit-nmt/blob/master/tools/news_evaluation.py
"""

import os, sys, json, tqdm

def LCS_length(s1, s2):
    '''
    Calculates the length of the longest common subsequence of s1 and s2
    s1 and s2 must be anything iterable
    The implementation is almost copy-pasted from Wikibooks.org
    '''
    m = len(s1)
    n = len(s2)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n+1) for i in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C[m][n]

def f_score(candidate, references):
    '''
    Calculates F-score for the candidate and its best matching reference
    Returns F-score and best matching reference
    '''
    # determine the best matching reference (the one with the shortest ED)
    best_ref = references[0]
    best_ref_lcs = LCS_length(candidate, references[0])
    for ref in references[1:]:
        lcs = LCS_length(candidate, ref)
        if (len(ref) - 2*lcs) < (len(best_ref) - 2*best_ref_lcs):
            best_ref = ref
            best_ref_lcs = lcs

    try:
        precision = float(best_ref_lcs)/float(len(candidate))
        recall = float(best_ref_lcs)/float(len(best_ref))
    except:
        return 0.0, best_ref
    #    import ipdb
    #    ipdb.set_trace()

    if best_ref_lcs:
        return 2*precision*recall/(precision+recall), best_ref
    else:
        return 0.0, best_ref

def mean_average_precision(candidates, references, n):
    '''
    Calculates mean average precision up to n candidates.
    '''

    total = 0.0
    num_correct = 0
    for k in range(n):
        if k < len(candidates) and (candidates[k] in references):
            num_correct += 1
        total += float(num_correct)/float(k+1)

    return total/float(n)

def inverse_rank(candidates, reference):
    '''
    Returns inverse rank of the matching candidate given the reference
    Returns 0 if no match was found.
    '''
    rank = 0
    while (rank < len(candidates)) and (candidates[rank] != reference):
        rank += 1
    if rank == len(candidates):
        return 0.0
    else:
        return 1.0/(rank+1)

def evaluate(input_data, test_data, verbose=True):
    '''
    Evaluates all metrics to save looping over input_data several times
    n is the map-n parameter
    Returns acc, f_score, mrr, map_ref, map_n
    '''
    mrr = {}
    acc = {}
    f = {}
    f_best_match = {}
    #map_n = {}
    map_ref = {}
    #map_sys = {}

    empty_xlits = []

    for src_word in test_data.keys():
        if src_word in input_data and len(input_data[src_word]) > 0:
            candidates = input_data[src_word]
            references = test_data[src_word]

            acc[src_word] = max([int(candidates[0] == ref) for ref in references]) # either 1 or 0

            f[src_word], f_best_match[src_word] = f_score(candidates[0], references)

            mrr[src_word] = max([inverse_rank(candidates, ref) for ref in references])

            #map_n[src_word] = mean_average_precision(candidates, references, n)
            map_ref[src_word] = mean_average_precision(candidates, references, len(references))
            #map_sys[src_word] = mean_average_precision(candidates, references, len(candidates))

        else:
            empty_xlits.append(src_word)
            mrr[src_word] = 0.0
            acc[src_word] = 0.0
            f[src_word] = 0.0
            f_best_match[src_word] = ''
            #map_n[src_word] = 0.0
            map_ref[src_word] = 0.0
            #map_sys[src_word] = 0.0

    if len(empty_xlits) > 0 and verbose:
        print('Warning: No transliterations found for following %d words out of %d:' % (len(empty_xlits), len(test_data.keys())))
        print(empty_xlits)

    return acc, f, f_best_match, mrr, map_ref

def write_details(output_fname, input_data, test_data, acc, f, f_best_match, mrr, map_ref):
    '''
    Writes detailed results to CSV file
    '''
    f_out = open(output_fname, 'w', encoding='utf-8')

    f_out.write('%s\n' % (','.join(['"Source word"', '"First candidate"', '"Top-1"', '"ACC"', '"F-score"', '"Best matching reference"',
    '"MRR"', '"MAP_ref"', '"References"'])))

    for src_word in test_data.keys():
        if src_word in input_data and len(input_data[src_word]) > 0:
            first_candidate = input_data[src_word][0]
        else:
            first_candidate = ''

        f_out.write('%s,%s,%f,%f,%s,%f,%f,%s\n' % (src_word, first_candidate, acc[src_word], f[src_word], f_best_match[src_word], mrr[src_word], map_ref[src_word], '"' + ' | '.join(test_data[src_word]) + '"'))

    f_out.close()

def read_data(gt_json, args):
    file_type = 'json' if args.pred_json else 'txt'
    # Check if files exist
    if not os.path.isfile(gt_json):
        sys.exit('ERROR:', gt_json, 'Not Found')
    # Read test data ground truth
    with open(gt_json, encoding='utf8') as f:
            gt_data = json.load(f)

    # Read Predictions based on file_type
    if file_type.lower() == 'json':
        pred_json = args.pred_json
        if not os.path.isfile(pred_json):
            sys.exit('ERROR:', pred_json, 'Not Found')

        # Read predictions
        with open(pred_json, encoding='utf8') as f:
            pred_data = json.load(f)

        return gt_data, pred_data

    elif file_type.lower() == 'txt':
        # Read predictions from 2 parallel txt files
        infile = args.inp
        outfile = args.out
        if not os.path.isfile(infile):
            sys.exit('ERROR:', infile, 'Not Found')
        if not os.path.isfile(outfile):
            sys.exit('ERROR:', outfile, 'Not Found')

        # Read all the input and output lines into memory
        with open(infile) as f:
            input_lines = f.readlines()
        with open(outfile) as f:
            output_lines = f.readlines()

        if len(input_lines) != len(output_lines):
            sys.exit('ERROR: The number of lines in input (%d) and output (%d) do not match!'
                    % (len(input_lines), len(output_lines)))

        pred_data = {}
        # Assumes one word per line
        for inp, out in zip(input_lines, output_lines):
            inp = inp.strip().replace(' ', '') #Remove spaces
            out = out.strip().replace(' ', '')
            if inp not in pred_data:
                pred_data[inp] = []
            pred_data[inp].append(out)

        if args.save_json:
            # Save as JSON for future reference
            with open(args.save_json, 'w', encoding='utf8') as f:
                json.dump(pred_data, f, ensure_ascii=False, indent=4, sort_keys=True)

        return gt_data, pred_data

    else:
        sys.exit('ERROR: Unrecognized predictions file format:', file_type)

def print_scores(args):
    gt_data, pred_data = read_data(args.gt_json, args)

    acc, f, f_best_match, mrr, map_ref = evaluate(pred_data, gt_data)

    if args.save_output_csv:
        write_details(args.save_output_csv, pred_data, gt_data, acc, f, f_best_match, mrr, map_ref)

    N = len(acc)
    sys.stdout.write('SCORES FOR %d SAMPLES:\n\n' % N)
    sys.stdout.write('ACC:          %f\n' % (float(sum([acc[src_word] for src_word in acc.keys()]))/N))
    sys.stdout.write('Mean F-score: %f\n' % (float(sum([f[src_word] for src_word in f.keys()]))/N))
    sys.stdout.write('MRR:          %f\n' % (float(sum([mrr[src_word] for src_word in mrr.keys()]))/N))
    sys.stdout.write('MAP_ref:      %f\n' % (float(sum([map_ref[src_word] for src_word in map_ref.keys()]))/N))
    #sys.stdout.write('MAP_%d:       %f\n' % (n, float(sum([map_n[src_word] for src_word in map_n.keys()]))/N))
    #sys.stdout.write('MAP_sys:      %f\n' % (float(sum([map_sys[src_word] for src_word in map_sys.keys()]))/N))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-json', type=str, required=True)
    parser.add_argument('--pred-json', type=str)
    parser.add_argument('--inp', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--save-output-csv', type=str)
    print_scores(parser.parse_args())