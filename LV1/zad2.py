try:
    broj = float(input())
except:
    print("Nije unešen broj.")
else:
    if broj < 0.0 or broj > 1.0:
        print("Broj izvan intervala.")
    elif broj < 0.6:
        print("F")
    elif broj < 0.7:
        print("D")
    elif broj < 0.8:
        print("C")
    elif broj < 0.9:
        print("B")
    else:
        print("A")
