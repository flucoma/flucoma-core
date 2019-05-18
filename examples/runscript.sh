# for i in {0..3}; do ../build/examples/test_envseg /Users/pa/Documents/documents@hudd/research/projects/fluid\ corpus\ navigation/research/ampseg/test-320.wav 250 2205 1 2205 221 -20 144 -144 0 0 0 -20 0 0 0 1 ${i}; mv out.wav out-${i}.wav; done

for i in {0..3}; do ../build/examples/test_envseg "/Users/pa/Documents/documents@hudd/research/projects/fluid corpus navigation/research/fluid_decomposition/AudioFiles/Nicol-LoopE-M.wav" 40 441 10 441 441 -40 13 4 0 0 0 -70 0 0 0 1 ${i}; mv out.wav out-${i}.wav; done
