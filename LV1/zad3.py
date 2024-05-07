def srednja_vrijednost(brojevi):
    zbroj = 0
    for broj in brojevi:
        zbroj += broj
    return zbroj / len(brojevi)


brojevi = []
while True:
    unos = input()
    if(unos == "Done"):
        break
    try:
        broj = int(unos)
    except:
        print("Nije unešen broj.")
        continue
    else:
        brojevi.append(broj)

print("Korisnik je unio", len(brojevi), "brojeva.")
print("Srednja vrijednost:", srednja_vrijednost(brojevi))
print("Min:", min(brojevi))
print("Max:", max(brojevi))
brojevi.sort()
print(brojevi)