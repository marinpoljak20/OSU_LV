def prosjecan_broj_rijeci(messages):
    count = 0
    length = 0
    for message in messages:
        length += len(message)
        words = message.split()
        count += len(words)
    return length / count

hams = []
spams = []

file = open("SMSSpamCollection.txt")
for line in file:
    line = line.rsplit("	")
    if line[0] == "ham":
        hams.append(line[1])
    else:
        spams.append(line[1])

print("Prosjecan broj rijeci HAM:", prosjecan_broj_rijeci(hams))
print("Prosjecan broj rijeci SPAM:", prosjecan_broj_rijeci(spams))

count = 0
for spam in spams:
    if spam.endswith('!'):
        count += 1

print(count, "SPAM poruka zavrsava usklicnikom.")