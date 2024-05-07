words = []
file = open("song.txt")
for line in file:
    current_words = line.split()
    for word in current_words:
        word.lower()
        words.append(word)
file.close()

dictionary = dict()

for word in words:
    dictionary[word] = words.count(word)

count = 0
for word in dictionary:
        if dictionary[word] == 1:
            count += 1
            print(word)

print("Ima", count, "rijeci koje se pojavljuju jednom.")