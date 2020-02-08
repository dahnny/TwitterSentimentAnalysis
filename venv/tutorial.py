freq = {}
line = input()
for word in line.split():
    freq[word]=(freq.get(word,0)+1)
print (freq)
    
