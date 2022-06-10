'''
Perform concatenation adversarial attack on 
GEC system, with aim of finding universal adversarial phrase
that minimises average number of edits between original and 
predicted gec sentence and ensures attack phrase keeps perplexity below detection threshold.
'''
import sys
import os
import argparse
import torch
from utils.helpers import read_lines_with_id
import json
from datetime import date
from utils.perplexity import perplexity
from statistics import mean
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from gector.gec_model import GecBERTModel

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_avg(model, sentences, attack_phrase):
    edit_counts = []
    for sent in sentences:
        sent_attack = sent + ' ' + attack_phrase
        _, cnt = model.handle_batch([sent_attack.split()])
        edit_counts.append(cnt)
    return mean(edit_counts)

def is_perp_less_than_thresh(sentences, attack_phrase, thresh):
    '''
        Return True if the average dataset perplexity is less than threshold
    '''
    perp_tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
    perp_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    perps = []
    for sent in sentences:
        sent = sent + ' ' + attack_phrase
        try:
            perp = perplexity(sent, perp_tokenizer, perp_model)
            perps.append(min(perp, 1000))
        except:
            continue
        # import pdb; pdb.set_trace()
    avg_perp = mean(perps)
    if avg_perp < thresh:
        return True
    return False

if __name__ == "__main__":

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--input_file',
                        help='Path to the data file',
                        required=True)
    parser.add_argument('--log',
                        type=str,
                        help='Specify txt file to log iteratively better words',
                        required=True)
    parser.add_argument('--asr_vocab',
                        type=str,
                        help='ASR vocab file',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the vocab file.',
                        default='data/output_vocabulary')
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--num_points',
                        type=int,
                        help='Number of training data points to consider',
                        default=1000)
    parser.add_argument('--search_size',
                        type=int,
                        help='Number of words to check',
                        default=400)
    parser.add_argument('--start',
                        type=int,
                        help='Vocab batch number',
                        default=0)
    parser.add_argument('--perp_thresh',
                        type=float,
                        help='Perplexity Detector threshold',
                        default=0)
    parser.add_argument('--seed',
                        type=int,
                        help='reproducibility',
                        default=1)
    parser.add_argument('--prev_attack',
                        type=str,
                        help='attack phrase to concatenate from before',
                        default='')
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/uni_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # load model
    model = GecBERTModel(vocab_path=args.vocab_path,
                            model_paths=args.model_path,
                            max_len=args.max_len, min_len=args.min_len,
                            iterations=args.iteration_count,
                            min_error_probability=args.min_error_probability,
                            lowercase_tokens=args.lowercase_tokens,
                            model_name=args.transformer_model,
                            special_tokens_fix=args.special_tokens_fix,
                            log=False,
                            confidence=args.additional_confidence,
                            del_confidence=args.additional_del_confidence,
                            is_ensemble=args.is_ensemble,
                            weigths=args.weights)

    # Load input sentences
    _, sentences = read_lines_with_id(args.input_file, do_random=True, num=args.num_points)

    # Get list of words to try
    with open(args.asr_vocab, 'r') as f:
        test_words = json.loads(f.read())
    test_words = [str(word).lower() for word in test_words]

    # Keep only selected batch of words
    start_index = args.start*args.search_size
    test_words = test_words[start_index:start_index+args.search_size]

    # Add blank word at beginning of list
    # test_words = ['']+test_words

    # Initialise empty log file
    with open(args.log, 'w') as f:
        f.write("Logged on "+ str(date.today()))

    best = ('none', 1000)
    for word in test_words:
        attack_phrase = args.prev_attack + ' ' + word + ' .'
        if not is_perp_less_than_thresh(sentences, attack_phrase, args.perp_thresh):
            continue
        edits_avg = get_avg(model, sentences, attack_phrase)
        # print(word, edits_avg) # temp debug

        if edits_avg < best[1]:
            best = (word, edits_avg)
            # Write to log
            with open(args.log, 'a') as f:
                out = '\n'+best[0]+" "+str(best[1])
                f.write(out)