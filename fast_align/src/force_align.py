#!/usr/bin/env python

import os
import subprocess
import sys
import threading

# Simplified, non-threadsafe version for force_align.py
# Use the version in realtime for development
class Aligner:

    def __init__(self, fwd_params, fwd_err, rev_params, rev_err, heuristic='grow-diag-final-and'):

        build_root = os.path.dirname(os.path.abspath(__file__))
        fast_align = os.path.join(build_root, 'fast_align')
        atools = os.path.join(build_root, 'atools')

        (fwd_T, fwd_m) = self.read_err(fwd_err)
        (rev_T, rev_m) = self.read_err(rev_err)

        fwd_cmd = [fast_align, '-i', '-', '-d', '-T', fwd_T, '-m', fwd_m, '-f', fwd_params]
        rev_cmd = [fast_align, '-i', '-', '-d', '-T', rev_T, '-m', rev_m, '-f', rev_params, '-r']
        tools_cmd = [atools, '-i', '-', '-j', '-', '-c', heuristic]

        self.fwd_align = popen_io(fwd_cmd)
        self.rev_align = popen_io(rev_cmd)
        self.tools = popen_io(tools_cmd)

    def align(self, line):
        self.fwd_align.stdin.write('{}\n'.format(line))
        self.rev_align.stdin.write('{}\n'.format(line))
        # f words ||| e words ||| links ||| score
        '''
        f = self.fwd_align.stdout.readline().split('|||')[2].strip().split('x')
        fwd_line = f[0]
        fwd_score =float(f[1])
        r = self.rev_align.stdout.readline().split('|||')[2].strip().split('x')
        rev_line = r[0]
        rev_score = float(r[1])
        global_score = (fwd_score+rev_score)/2
        '''
        fwd_line_score = self.fwd_align.stdout.readline().split('|||')[2:]
        try:
            fwd_line, fwd_score = fwd_line_score[0].strip(), float(fwd_line_score[1].strip())
        except ValueError:
            print("Value Error")
            fwd_line = ""
            fwd_score = 0
        except:
            print("Unknown Error")
            fwd_line = ""
            fwd_score = 0


        rev_line_score = self.rev_align.stdout.readline().split('|||')[2:]
        try:
            rev_line, rev_score = rev_line_score[0].strip(), float(rev_line_score[1].strip())
        except ValueError:
            print("Value Error")
            rev_line = ""
            rev_score = 0
        except:
            print("Unknown Error")
            rev_line = ""
            rev_score = 0

        global_score = (fwd_score + rev_score)/2
        
        if fwd_line == "" or rev_line == "":
            return "", 0.0
        else:
            self.tools.stdin.write('{}\n'.format(fwd_line))
            self.tools.stdin.write('{}\n'.format(rev_line))

            al_line = self.tools.stdout.readline().strip()
            return al_line, global_score
 
    def close(self):
        self.fwd_align.stdin.close()
        self.fwd_align.wait()
        self.rev_align.stdin.close()
        self.rev_align.wait()
        self.tools.stdin.close()
        self.tools.wait()

    def read_err(self, err):
        (T, m) = ('', '')
        for line in open(err):
            # expected target length = source length * N
            if 'expected target length' in line:
                m = line.split()[-1]
            # final tension: N
            elif 'final tension' in line:
                T = line.split()[-1]
        return (T, m)

def popen_io(cmd):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize = 1, universal_newlines= True, encoding="UTF-8")
    def consume(s):
        for _ in s:
            pass
    threading.Thread(target=consume, args=(p.stderr,)).start()
    return p

def main():
    print("force_align.py main function")

    if len(sys.argv[1:]) < 4:
        sys.stderr.write('run:\n')
        sys.stderr.write('  fast_align -i corpus.f-e -d -v -o -p fwd_params >fwd_align 2>fwd_err\n')
        sys.stderr.write('  fast_align -i corpus.f-e -r -d -v -o -p rev_params >rev_align 2>rev_err\n')
        sys.stderr.write('\n')
        sys.stderr.write('then run:\n')
        sys.stderr.write('  {} fwd_params fwd_err rev_params rev_err [heuristic] <in.f-e >out.f-e.gdfa\n'.format(sys.argv[0]))
        sys.stderr.write('\n')
        sys.stderr.write('where heuristic is one of: (intersect union grow-diag grow-diag-final grow-diag-final-and) default=grow-diag-final-and\n')
        sys.exit(2)

    aligner = Aligner(*sys.argv[1:])

    count = 0
    while True:
        line = sys.stdin.readline()
        #if count == 0:
        #    print("Read line: ", line)
        if not line:
            break
        # sys.stdout.write('{}\n'.format(aligner.align(line.strip())))
        sys.stdout.write('{} | {}\n'.format(aligner.align(line.strip())[0],aligner.align(line.strip())[1]))
        sys.stdout.flush()
        count = count + 1

    aligner.close()

def run(input: str, fwd_params, fwd_err, rev_params, rev_err, heuristic):
    print("run function")
    aligner = Aligner(fwd_params, fwd_err, rev_params, rev_err, heuristic)
    output = ""

    count = 0
    input_list = input.split("\n")
    for line in input_list:
        #if count == 0:
        #    print("Read line: ", line)
        if not line:
            break
        # sys.stdout.write('{}\n'.format(aligner.align(line.strip())))
        output += ('{} | {}\n'.format(aligner.align(line.strip())[0],aligner.align(line.strip())[1]))
        
        count = count + 1

    aligner.close()
    return output
    
if __name__ == '__main__':
    main()


