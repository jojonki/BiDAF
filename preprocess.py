import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def lower_list(str_list):
    return [str_var.lower() for str_var in str_list]


def preprocess(fpath_read, fpath_write):
    count = 0
    with open(fpath_read, 'r') as fs:
        data = fs.read()
        data = json.loads(data)
        fpw   = open(fpath_write, 'w')
        for c in tqdm(data['data']):
            title = c['title']
            for i, p in enumerate(c['paragraphs']):
                # context      = p['context'].split(' ')
                # context_char = list(p['context'])
                # context_pos  = {}
                context_label = title + '#' + str(i)
                for qa in p['qas']:
                    question = word_tokenize(qa['question'])

                    a = qa['answers'][0] # TODO multiple labels
                    answer = a['text'].strip()
                    answer_start = int(a['answer_start'])

                    # add '.' here, just because NLTK is not good enough in some cases
                    answer_words = word_tokenize(answer+'.')
                    # answer_words = word_tokenize(answer)
                    if answer_words[-1] == '.':
                        answer_words = answer_words[:-1]
                    else:
                        answer_words = word_tokenize(answer)

                    left_context  = word_tokenize( p['context'][0:answer_start] )
                    right_context = word_tokenize( p['context'][answer_start:] )
                    answer_reproduce = []
                    for i in range(len(answer_words)):
                        assert(i < len(right_context))
                        answer_reproduce.append( right_context[i] )
                    join_a  = ' '.join(answer_words)
                    join_ar = ' '.join(answer_reproduce)

                    if join_a != join_ar: # TODO
                        print('reproduced answers are different.')
                        print(qa['id'])
                        print('join_a:', join_a, '\njoin_ar:', join_ar)
                        count += 1
                        print('current different count:', count)

                    fpw.write(context_label + '\t')
                    fpw.write(' '.join(left_context+right_context) + '\t')
                    fpw.write(' '.join(question) + '\t')

                    answer_seq = []
                    for i in range(len(answer_words)):
                        assert(i < len(right_context))
                        # answer_seq.append(str(len(left_context)+i+1))
                        answer_seq.append(str(len(left_context)+i))
                    if len(answer_seq) == 0:
                        print('join_ar', join_ar)
                        print('join_a', join_a)
                        print('answer:'+answer)
                    assert(len(answer_seq) > 0)
                    fpw.write(' '.join(answer_seq) + '\t')
                    fpw.write(answer + '\n')

        fpw.close()
preprocess('./dataset/train-v1.1.json', './dataset/train.txt')
preprocess('./dataset/dev-v1.1.json', './dataset/dev.txt')
print('SQuAD preprossing finished!')
