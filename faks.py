'''def total_euro(x, y):
    return float(x)*float(y)
    

x = input('radni sati: ')
y = input('eura/h: ')
ukupno = total_euro(x, y)
print('ukupno: '+ str(ukupno))'''



try:
    x= input('broj izmedu 0 i 1: ')
    x=float(x)
    if(x>1):
        print('broj nije u odgovarajucem intervalu')
    elif(x>= 0.9):
        print('A')
    elif(x>= 0.8):
        print('B')
    elif(x >= 0.7):
        print('C')
    elif(x>= 0.6):
        print('D')
    elif(x< 0.6):
        print('F')
except:
    print('niste unijeli broj: ')

